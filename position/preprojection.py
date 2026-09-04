"""Attention-local sinusoidal injection before the Q/K projections."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Literal

import torch

from position.basis import FrozenFourierBasis
from position.frequency import (
    SINUSOID_FREQUENCY_DEFAULTS,
    normalize_sinusoid_frequency_config,
)
from position.precision import PreserveFP32BuffersMixin


QKPreprojectionMode = Literal[
    "tied_scalar",
    "tied_smooth_amplitude",
    "tied_smooth_polar",
    "split_scalar",
    "split_smooth_polar",
    "split_pair_amplitude",
    "split_pair_polar",
]
QK_PREPROJECTION_MODES = {
    "tied_scalar",
    "tied_smooth_amplitude",
    "tied_smooth_polar",
    "split_scalar",
    "split_smooth_polar",
    "split_pair_amplitude",
    "split_pair_polar",
}

QK_PREPROJECTION_DEFAULTS = {
    "enabled": False,
    "mode": "tied_scalar",
    "basis_dim": None,
    "theta": None,
    "gate_init": 1.0,
    "learnable_gate": True,
    # Number of low-order DCT modes for smooth spectral amplitude/phase.
    "smooth_rank": 4,
    # The parameterized bank itself lives once on Transformer and is shared by
    # all of these per-layer carrier adapters.
    "frequency": copy.deepcopy(SINUSOID_FREQUENCY_DEFAULTS),
}


@dataclass(frozen=True)
class QKPreprojectionOutput:
    """Separate positional inputs for the Q and K projection branches."""

    q: torch.Tensor
    k: torch.Tensor


def normalize_qk_preprojection_config(
    config: dict | None,
    *,
    model_dim: int,
    rope_theta: float,
    frequency_reference_length: int | None = None,
) -> dict:
    if config is None:
        config = {}
    if not isinstance(config, dict):
        raise TypeError("qk_preprojection must be an object")
    unknown = set(config) - set(QK_PREPROJECTION_DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown qk_preprojection keys: {sorted(unknown)}")
    normalized = copy.deepcopy(QK_PREPROJECTION_DEFAULTS)
    normalized.update(config)
    if not isinstance(normalized["enabled"], bool):
        raise TypeError("qk_preprojection.enabled must be a boolean")
    mode = normalized["mode"]
    if mode not in QK_PREPROJECTION_MODES:
        raise ValueError(
            "qk_preprojection.mode must be one of "
            f"{sorted(QK_PREPROJECTION_MODES)}, got {mode!r}"
        )
    basis_dim = normalized["basis_dim"]
    basis_dim = model_dim if basis_dim is None else int(basis_dim)
    if basis_dim != model_dim:
        raise ValueError(
            "qk_preprojection currently requires basis_dim=model_dim; this "
            "keeps the position projection tied exactly to W_q/W_k"
        )
    if basis_dim <= 0 or basis_dim % 2:
        raise ValueError("qk_preprojection requires a positive even model_dim")
    normalized["basis_dim"] = basis_dim
    theta = rope_theta if normalized["theta"] is None else float(normalized["theta"])
    if not math.isfinite(theta) or theta <= 0:
        raise ValueError("qk_preprojection.theta must be finite and positive")
    normalized["theta"] = theta
    gate_init = normalized["gate_init"]
    if isinstance(gate_init, bool) or not isinstance(gate_init, (int, float)):
        raise TypeError("qk_preprojection.gate_init must be a number")
    normalized["gate_init"] = float(gate_init)
    if not math.isfinite(normalized["gate_init"]):
        raise ValueError("qk_preprojection.gate_init must be finite")
    if not isinstance(normalized["learnable_gate"], bool):
        raise TypeError("qk_preprojection.learnable_gate must be a boolean")
    smooth_rank = normalized["smooth_rank"]
    if isinstance(smooth_rank, bool) or not isinstance(smooth_rank, int):
        raise TypeError("qk_preprojection.smooth_rank must be an integer")
    if smooth_rank <= 0:
        raise ValueError("qk_preprojection.smooth_rank must be positive")
    if "smooth" in mode and smooth_rank >= basis_dim // 2:
        raise ValueError(
            "qk_preprojection.smooth_rank must be smaller than the number "
            "of Fourier pairs"
        )
    normalized["smooth_rank"] = smooth_rank
    normalized["frequency"] = normalize_sinusoid_frequency_config(
        normalized["frequency"],
        default_reference_length=frequency_reference_length,
    )
    if not normalized["enabled"] and normalized["frequency"]["mode"] != "fixed":
        raise ValueError(
            "qk_preprojection.frequency can be learned only when "
            "qk_preprojection.enabled=true"
        )
    return normalized


def _orthonormal_zero_sum_basis(size: int) -> torch.Tensor:
    """Return a Helmert basis for the zero-mean subspace of R^size."""
    if size <= 1:
        return torch.empty((size, 0), dtype=torch.float32)
    basis = torch.zeros((size, size - 1), dtype=torch.float32)
    for column in range(size - 1):
        count = column + 1
        denominator = math.sqrt(count * (count + 1))
        basis[:count, column] = 1.0 / denominator
        basis[count, column] = -count / denominator
    return basis


def _unit_rms_cosine_basis(
    size: int,
    *,
    start_mode: int,
    count: int,
) -> torch.Tensor:
    """Return orthogonal, unit-RMS DCT-II modes over log-frequency index.

    Unit-RMS scaling gives each coordinate a width-independent functional
    meaning. An L2-normalized column would shrink its per-band forward effect
    as ``1/sqrt(pair_dim)``; Adam's gradient normalization would not undo that
    forward Jacobian.
    """
    if count <= 0:
        return torch.empty((size, 0), dtype=torch.float32)
    index = torch.arange(size, dtype=torch.float32)[:, None] + 0.5
    modes = torch.arange(
        start_mode,
        start_mode + count,
        dtype=torch.float32,
    )[None, :]
    basis = torch.cos(math.pi * index * modes / size)
    scale = torch.full((count,), math.sqrt(2.0), dtype=torch.float32)
    if start_mode == 0:
        scale[0] = 1.0
    return basis * scale


class QKPreprojectionPosition(PreserveFP32BuffersMixin, torch.nn.Module):
    """Produce sinusoidal inputs added only before the Q and K projections.

    ``tied_scalar`` is the original implementation. Smooth modes use a few
    low-order DCT modes over the uniformly spaced log-frequency index. The
    amplitude basis excludes the constant mode so the global gate remains
    identifiable; the phase basis includes it. Split modes give Q and K
    separate transforms. Pairwise modes retain the historical fully free
    spectral controls.

    For pair ``i`` and branch ``b`` the pairwise modes apply

    ``A_i^b = g_b * exp(delta_i^b) * R(phi_i^b)``.

    ``delta`` lives in an explicit zero-sum subspace, so its spectral factor
    has geometric mean one and cannot duplicate the global gain.
    """

    _fp32_buffer_names = (
        "zero_sum_basis",
        "smooth_amplitude_basis",
        "smooth_phase_basis",
        "fixed_gate",
        "fixed_q_gate",
        "fixed_k_gate",
    )
    _fp32_parameter_names = (
        "gate",
        "q_gate",
        "k_gate",
        "log_amplitude_coordinates",
        "q_log_amplitude_coordinates",
        "k_log_amplitude_coordinates",
        "phase_coordinates",
        "q_phase",
        "k_phase",
    )

    def __init__(self, config: dict, *, model_dim: int, extent: int) -> None:
        super().__init__()
        self.config = copy.deepcopy(config)
        self.model_dim = model_dim
        self.pair_dim = model_dim // 2
        self.extent = extent
        self.mode: QKPreprojectionMode = config["mode"]
        self.smooth_rank = int(config["smooth_rank"])
        self.basis = FrozenFourierBasis(
            extent=extent,
            basis_dim=model_dim,
            theta=float(config["theta"]),
        )
        self.register_buffer(
            "zero_sum_basis",
            _orthonormal_zero_sum_basis(self.pair_dim),
            persistent=False,
        )
        self.register_buffer(
            "smooth_amplitude_basis",
            _unit_rms_cosine_basis(
                self.pair_dim,
                start_mode=1,
                count=self.smooth_rank,
            ),
            persistent=False,
        )
        self.register_buffer(
            "smooth_phase_basis",
            _unit_rms_cosine_basis(
                self.pair_dim,
                start_mode=0,
                count=self.smooth_rank,
            ),
            persistent=False,
        )

        gate = torch.tensor(float(config["gate_init"]), dtype=torch.float32)
        self.register_parameter("gate", None)
        self.register_parameter("q_gate", None)
        self.register_parameter("k_gate", None)
        self.register_buffer("fixed_gate", None)
        self.register_buffer("fixed_q_gate", None)
        self.register_buffer("fixed_k_gate", None)
        if self.mode.startswith("tied_"):
            if config["learnable_gate"]:
                self.gate = torch.nn.Parameter(gate)
            else:
                self.fixed_gate = gate
        elif config["learnable_gate"]:
            self.q_gate = torch.nn.Parameter(gate.clone())
            self.k_gate = torch.nn.Parameter(gate.clone())
        else:
            self.fixed_q_gate = gate.clone()
            self.fixed_k_gate = gate.clone()

        self.register_parameter("log_amplitude_coordinates", None)
        self.register_parameter("q_log_amplitude_coordinates", None)
        self.register_parameter("k_log_amplitude_coordinates", None)
        if self.mode in {"tied_smooth_amplitude", "tied_smooth_polar"}:
            self.log_amplitude_coordinates = torch.nn.Parameter(
                torch.zeros(self.smooth_rank, dtype=torch.float32)
            )
        elif self.mode in {"split_pair_amplitude", "split_pair_polar"}:
            coordinate_dim = max(self.pair_dim - 1, 0)
            self.q_log_amplitude_coordinates = torch.nn.Parameter(
                torch.zeros(coordinate_dim, dtype=torch.float32)
            )
            self.k_log_amplitude_coordinates = torch.nn.Parameter(
                torch.zeros(coordinate_dim, dtype=torch.float32)
            )
        elif self.mode == "split_smooth_polar":
            self.q_log_amplitude_coordinates = torch.nn.Parameter(
                torch.zeros(self.smooth_rank, dtype=torch.float32)
            )
            self.k_log_amplitude_coordinates = torch.nn.Parameter(
                torch.zeros(self.smooth_rank, dtype=torch.float32)
            )

        self.register_parameter("phase_coordinates", None)
        self.register_parameter("q_phase", None)
        self.register_parameter("k_phase", None)
        if self.mode == "tied_smooth_polar":
            self.phase_coordinates = torch.nn.Parameter(
                torch.zeros(self.smooth_rank, dtype=torch.float32)
            )
        elif self.mode == "split_pair_polar":
            self.q_phase = torch.nn.Parameter(
                torch.zeros(self.pair_dim, dtype=torch.float32)
            )
            self.k_phase = torch.nn.Parameter(
                torch.zeros(self.pair_dim, dtype=torch.float32)
            )
        elif self.mode == "split_smooth_polar":
            self.q_phase = torch.nn.Parameter(
                torch.zeros(self.smooth_rank, dtype=torch.float32)
            )
            self.k_phase = torch.nn.Parameter(
                torch.zeros(self.smooth_rank, dtype=torch.float32)
            )

    def _apply(self, fn, recurse: bool = True):
        # Static phase and log-amplitude are tiny parameter sets whose numerical
        # meaning is angle/scale, not an activation dtype. Preserve their fp32
        # values across module-wide half/bfloat16 casts, while still following
        # device moves. Parameter identity is retained for optimizer safety.
        originals = {
            name: parameter.detach().float().clone()
            for name in self._fp32_parameter_names
            if (parameter := self._parameters.get(name)) is not None
        }
        module = super()._apply(fn, recurse=recurse)
        for name, original in originals.items():
            parameter = self._parameters[name]
            parameter.data = original.to(
                device=parameter.device,
                dtype=torch.float32,
            )
            if parameter.grad is not None:
                parameter.grad.data = parameter.grad.detach().to(
                    device=parameter.device,
                    dtype=torch.float32,
                )
        return module

    def gate_values(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.mode.startswith("tied_"):
            value = self.gate if self.gate is not None else self.fixed_gate
            return value, value
        q_value = self.q_gate if self.q_gate is not None else self.fixed_q_gate
        k_value = self.k_gate if self.k_gate is not None else self.fixed_k_gate
        return q_value, k_value

    def gate_value(self) -> torch.Tensor:
        """Return the historical scalar gate for the default tied mode."""
        if self.mode != "tied_scalar":
            raise RuntimeError("gate_value() is defined only for tied_scalar mode")
        return self.gate_values()[0]

    def log_amplitude_deltas(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.log_amplitude_coordinates is not None:
            delta = (
                self.smooth_amplitude_basis
                @ self.log_amplitude_coordinates.float()
            )
            return delta, delta
        if self.q_log_amplitude_coordinates is None:
            zero = self.zero_sum_basis.new_zeros(self.pair_dim)
            return zero, zero
        basis = (
            self.smooth_amplitude_basis
            if self.mode == "split_smooth_polar"
            else self.zero_sum_basis
        )
        return (
            basis @ self.q_log_amplitude_coordinates.float(),
            basis @ self.k_log_amplitude_coordinates.float(),
        )

    def phase_values(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.phase_coordinates is not None:
            phase = self.smooth_phase_basis @ self.phase_coordinates.float()
            return phase, phase
        if self.q_phase is None:
            zero = self.zero_sum_basis.new_zeros(self.pair_dim)
            return zero, zero
        if self.mode == "split_smooth_polar":
            return (
                self.smooth_phase_basis @ self.q_phase.float(),
                self.smooth_phase_basis @ self.k_phase.float(),
            )
        return self.q_phase.float(), self.k_phase.float()

    @staticmethod
    def _transform_pairs(
        basis: torch.Tensor,
        *,
        gain: torch.Tensor,
        log_amplitude_delta: torch.Tensor,
        phase: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        pairs = basis.float().reshape(basis.shape[0], -1, 2)
        amplitude = gain.float() * log_amplitude_delta.float().exp()
        phase_fp32 = phase.float()
        phase_cos = phase_fp32.cos()
        phase_sin = phase_fp32.sin()
        real, imag = pairs.unbind(dim=-1)
        rotated = torch.stack(
            (
                real * phase_cos - imag * phase_sin,
                real * phase_sin + imag * phase_cos,
            ),
            dim=-1,
        )
        return (rotated * amplitude[None, :, None]).flatten(-2).to(dtype=dtype)

    def forward(
        self,
        length: int,
        *,
        dtype: torch.dtype,
        basis_override: torch.Tensor | None = None,
    ) -> QKPreprojectionOutput:
        q_gate, k_gate = self.gate_values()
        if basis_override is not None:
            if (
                basis_override.ndim != 2
                or basis_override.shape[0] < length
                or basis_override.shape[1] != self.model_dim
            ):
                raise ValueError(
                    "basis_override must have shape "
                    f"[at least {length},{self.model_dim}], got "
                    f"{list(basis_override.shape)}"
                )
            basis_fp32 = basis_override[:length].float()
        else:
            basis_fp32 = self.basis(length)
        if self.mode in {"tied_scalar", "split_scalar"}:
            basis = basis_fp32.to(dtype=dtype)
            q = basis * q_gate.to(dtype=dtype)
            k = basis * k_gate.to(dtype=dtype)
            return QKPreprojectionOutput(q=q, k=k)

        # Keep the stored carrier and phase/amplitude transforms in fp32; cast
        # only the completed coefficients used by the model-width branches.
        basis = basis_fp32
        q_delta, k_delta = self.log_amplitude_deltas()
        q_phase, k_phase = self.phase_values()
        return QKPreprojectionOutput(
            q=self._transform_pairs(
                basis,
                gain=q_gate,
                log_amplitude_delta=q_delta,
                phase=q_phase,
                dtype=dtype,
            ),
            k=self._transform_pairs(
                basis,
                gain=k_gate,
                log_amplitude_delta=k_delta,
                phase=k_phase,
                dtype=dtype,
            ),
        )

    def reset_output_parameters(self) -> None:
        gate_init = float(self.config["gate_init"])
        with torch.no_grad():
            for gate in (self.gate, self.q_gate, self.k_gate):
                if gate is not None:
                    gate.fill_(gate_init)
            for parameter in (
                self.q_log_amplitude_coordinates,
                self.k_log_amplitude_coordinates,
                self.log_amplitude_coordinates,
                self.q_phase,
                self.k_phase,
                self.phase_coordinates,
            ):
                if parameter is not None:
                    parameter.zero_()
