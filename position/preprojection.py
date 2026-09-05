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
    "tied_smooth_direct_amplitude",
]
QK_PREPROJECTION_MODES = {
    "tied_scalar",
    "tied_smooth_amplitude",
    "tied_smooth_direct_amplitude",
}
QK_PREPROJECTION_REMOVED_MODES = {
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
    # Number of low-order DCT modes for smooth spectral amplitude.
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
    if mode in QK_PREPROJECTION_REMOVED_MODES:
        if normalized["enabled"]:
            raise ValueError(
                f"qk_preprojection.mode={mode!r} was removed from the active "
                "runtime after the Phase 33/35 null results; use "
                "'tied_scalar', 'tied_smooth_amplitude', or "
                "'tied_smooth_direct_amplitude', or recover the historical "
                "implementation from git history"
            )
        # Disabled archival blocks have no model effect. Canonicalize them so
        # old resolved configs remain understandable without dormant runtime
        # branches.
        mode = "tied_scalar"
        normalized["mode"] = mode
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


def _smooth_amplitude_basis(size: int, count: int) -> torch.Tensor:
    """Return zero-mean, unit-RMS DCT-II modes over log-frequency index.

    Unit-RMS scaling gives each coordinate a width-independent functional
    meaning. An L2-normalized column would shrink its per-band forward effect
    as ``1/sqrt(pair_dim)``; Adam's gradient normalization would not undo that
    forward Jacobian.
    """
    if count <= 0:
        return torch.empty((size, 0), dtype=torch.float32)
    index = torch.arange(size, dtype=torch.float32)[:, None] + 0.5
    # Start at one: the omitted constant mode belongs to the scalar gate.
    modes = torch.arange(1, count + 1, dtype=torch.float32)[None, :]
    basis = torch.cos(math.pi * index * modes / size)
    return basis * math.sqrt(2.0)


class QKPreprojectionPosition(PreserveFP32BuffersMixin, torch.nn.Module):
    """Produce sinusoidal inputs added only before the Q and K projections.

    ``tied_scalar`` is the established carrier. The smooth modes add a few
    low-order DCT modes over the uniformly spaced log-frequency index.
    ``tied_smooth_amplitude`` retains the historical exponential map;
    ``tied_smooth_direct_amplitude`` uses the signed affine factor ``1 + Bc``.
    Its basis excludes the constant mode, so the shared global gate remains
    identifiable. Both modes give Q and K exactly the same positional input;
    their existing projections still learn separate reads of it.
    """

    _fp32_buffer_names = (
        "smooth_amplitude_basis",
        "fixed_gate",
    )
    _fp32_parameter_names = (
        "gate",
        "log_amplitude_coordinates",
        "amplitude_coordinates",
    )

    def __init__(self, config: dict, *, model_dim: int, extent: int) -> None:
        super().__init__()
        self.config = copy.deepcopy(config)
        self.model_dim = model_dim
        self.pair_dim = model_dim // 2
        self.extent = extent
        self.mode: QKPreprojectionMode = config["mode"]
        if self.mode not in QK_PREPROJECTION_MODES:
            raise ValueError(
                f"QKPreprojectionPosition received inactive mode {self.mode!r}; "
                "normalize the qk_preprojection config before construction"
            )
        self.smooth_rank = int(config["smooth_rank"])
        self.basis = FrozenFourierBasis(
            extent=extent,
            basis_dim=model_dim,
            theta=float(config["theta"]),
        )
        self.register_buffer(
            "smooth_amplitude_basis",
            _smooth_amplitude_basis(
                self.pair_dim,
                (
                    self.smooth_rank
                    if self.mode in {
                        "tied_smooth_amplitude",
                        "tied_smooth_direct_amplitude",
                    }
                    else 0
                ),
            ),
            persistent=False,
        )

        gate = torch.tensor(float(config["gate_init"]), dtype=torch.float32)
        if config["learnable_gate"]:
            self.gate = torch.nn.Parameter(gate)
            self.register_buffer("fixed_gate", None)
        else:
            self.register_parameter("gate", None)
            self.register_buffer("fixed_gate", gate)

        if self.mode == "tied_smooth_amplitude":
            self.log_amplitude_coordinates = torch.nn.Parameter(
                torch.zeros(self.smooth_rank, dtype=torch.float32)
            )
        else:
            self.register_parameter("log_amplitude_coordinates", None)
        if self.mode == "tied_smooth_direct_amplitude":
            self.amplitude_coordinates = torch.nn.Parameter(
                torch.zeros(self.smooth_rank, dtype=torch.float32)
            )
        else:
            self.register_parameter("amplitude_coordinates", None)

    def _apply(self, fn, recurse: bool = True):
        # Gate and log-amplitude are tiny parameter sets whose numerical
        # meaning is scale, not an activation dtype. Preserve their fp32 values
        # across module-wide half/bfloat16 casts while still following device
        # moves. Parameter identity is retained for optimizer safety.
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
        value = self.gate if self.gate is not None else self.fixed_gate
        return value, value

    def gate_value(self) -> torch.Tensor:
        """Return the historical scalar gate for the default tied mode."""
        if self.mode != "tied_scalar":
            raise RuntimeError("gate_value() is defined only for tied_scalar mode")
        return self.gate_values()[0]

    def log_amplitude_deltas(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.log_amplitude_coordinates is None:
            delta = self.smooth_amplitude_basis.new_zeros(self.pair_dim)
        else:
            delta = (
                self.smooth_amplitude_basis
                @ self.log_amplitude_coordinates.float()
            )
        return delta, delta

    def direct_amplitude_deltas(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.amplitude_coordinates is None:
            delta = self.smooth_amplitude_basis.new_zeros(self.pair_dim)
        else:
            delta = self.smooth_amplitude_basis @ self.amplitude_coordinates.float()
        return delta, delta

    def amplitude_factors(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.log_amplitude_coordinates is not None:
            factor = self.log_amplitude_deltas()[0].exp()
        elif self.amplitude_coordinates is not None:
            factor = 1.0 + self.direct_amplitude_deltas()[0]
        else:
            factor = self.smooth_amplitude_basis.new_ones(self.pair_dim)
        return factor, factor

    @staticmethod
    def _transform_pairs(
        basis: torch.Tensor,
        *,
        gain: torch.Tensor,
        amplitude_factor: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        pairs = basis.float().reshape(basis.shape[0], -1, 2)
        amplitude = gain.float() * amplitude_factor.float()
        return (pairs * amplitude[None, :, None]).flatten(-2).to(dtype=dtype)

    def forward(
        self,
        length: int,
        *,
        dtype: torch.dtype,
        basis_override: torch.Tensor | None = None,
    ) -> QKPreprojectionOutput:
        gate = self.gate_values()[0]
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
        if self.mode == "tied_scalar":
            basis = basis_fp32.to(dtype=dtype)
            positional = basis * gate.to(dtype=dtype)
            return QKPreprojectionOutput(q=positional, k=positional)

        # Keep the stored carrier and amplitude transform in fp32; cast
        # only the completed coefficients used by the model-width branches.
        amplitude_factor = self.amplitude_factors()[0]
        positional = self._transform_pairs(
            basis_fp32,
            gain=gate,
            amplitude_factor=amplitude_factor,
            dtype=dtype,
        )
        return QKPreprojectionOutput(q=positional, k=positional)

    def reset_output_parameters(self) -> None:
        gate_init = float(self.config["gate_init"])
        with torch.no_grad():
            if self.gate is not None:
                self.gate.fill_(gate_init)
            if self.log_amplitude_coordinates is not None:
                self.log_amplitude_coordinates.zero_()
            if self.amplitude_coordinates is not None:
                self.amplitude_coordinates.zero_()
