"""Attention-local sinusoidal injection before the Q/K projections."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Literal

import torch

from position.basis import FrozenFourierBasis
from position.precision import PreserveFP32BuffersMixin


QKPreprojectionMode = Literal["tied_scalar"]
QK_PREPROJECTION_MODES = {"tied_scalar"}

# Historical modes remain recognizable so archived disabled configs normalize
# cleanly and archived enabled configs fail with an actionable message. Their
# implementations and parameters live in git history and the phase reports.
QK_PREPROJECTION_REMOVED_MODES = {
    "tied_smooth_amplitude",
    "tied_smooth_direct_amplitude",
    "tied_smooth_polar",
    "split_scalar",
    "split_smooth_polar",
    "split_pair_amplitude",
    "split_pair_polar",
}
QK_PREPROJECTION_LEGACY_KEYS = {"smooth_rank", "frequency"}

QK_PREPROJECTION_DEFAULTS = {
    "enabled": False,
    "mode": "tied_scalar",
    "basis_dim": None,
    "theta": None,
    "gate_init": 1.0,
    "learnable_gate": True,
}


@dataclass(frozen=True)
class QKPreprojectionOutput:
    """Separate positional inputs for the Q and K projection branches."""

    q: torch.Tensor
    k: torch.Tensor


def _legacy_frequency_mode(config: dict) -> str:
    frequency = config.get("frequency")
    if frequency is None:
        return "fixed"
    if not isinstance(frequency, dict):
        raise TypeError("qk_preprojection.frequency must be an object")
    mode = frequency.get("mode", "fixed")
    if not isinstance(mode, str):
        raise TypeError("qk_preprojection.frequency.mode must be a string")
    return mode


def normalize_qk_preprojection_config(
    config: dict | None,
    *,
    model_dim: int,
    rope_theta: float,
) -> dict:
    """Validate and resolve the surviving tied-scalar carrier.

    ``smooth_rank`` and ``frequency`` are accepted only as compatibility keys
    from archived resolved configs. An enabled historical intervention fails
    explicitly instead of silently changing its meaning.
    """
    if config is None:
        config = {}
    if not isinstance(config, dict):
        raise TypeError("qk_preprojection must be an object")
    allowed = set(QK_PREPROJECTION_DEFAULTS) | QK_PREPROJECTION_LEGACY_KEYS
    unknown = set(config) - allowed
    if unknown:
        raise ValueError(f"Unknown qk_preprojection keys: {sorted(unknown)}")

    normalized = copy.deepcopy(QK_PREPROJECTION_DEFAULTS)
    normalized.update(
        {key: value for key, value in config.items() if key in normalized}
    )
    if not isinstance(normalized["enabled"], bool):
        raise TypeError("qk_preprojection.enabled must be a boolean")

    mode = normalized["mode"]
    frequency_mode = _legacy_frequency_mode(config)
    if normalized["enabled"] and (
        mode in QK_PREPROJECTION_REMOVED_MODES or frequency_mode != "fixed"
    ):
        raise ValueError(
            "Learned pre-Q/K carrier shape/frequency modes were removed after "
            "the Phase 33--37 null confirmations. Use mode='tied_scalar' with "
            "a fixed carrier, or recover the historical implementation from "
            "git history."
        )
    if not normalized["enabled"]:
        # A disabled archival block has no model effect. Canonicalize it so a
        # historical resolved config does not retain dormant active settings.
        mode = "tied_scalar"
        normalized["mode"] = mode
    if mode not in QK_PREPROJECTION_MODES:
        raise ValueError(
            "qk_preprojection.mode must be 'tied_scalar'; historical shape "
            "modes were removed after Phases 33--37"
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
    return normalized


class QKPreprojectionPosition(PreserveFP32BuffersMixin, torch.nn.Module):
    """Add one tied, gated Fourier carrier before the Q and K projections.

    Q and K receive the same positional input, but their existing projection
    matrices learn separate reads. V and the residual stream are untouched.
    """

    _fp32_buffer_names = ("fixed_gate",)
    _fp32_parameter_names = ("gate",)

    def __init__(self, config: dict, *, model_dim: int, extent: int) -> None:
        super().__init__()
        self.config = copy.deepcopy(config)
        self.model_dim = model_dim
        self.extent = extent
        self.mode: QKPreprojectionMode = config["mode"]
        if self.mode not in QK_PREPROJECTION_MODES:
            raise ValueError(
                f"QKPreprojectionPosition received inactive mode {self.mode!r}; "
                "normalize the qk_preprojection config before construction"
            )
        self.basis = FrozenFourierBasis(
            extent=extent,
            basis_dim=model_dim,
            theta=float(config["theta"]),
        )

        gate = torch.tensor(float(config["gate_init"]), dtype=torch.float32)
        if config["learnable_gate"]:
            self.gate = torch.nn.Parameter(gate)
            self.register_buffer("fixed_gate", None)
        else:
            self.register_parameter("gate", None)
            self.register_buffer("fixed_gate", gate)

    def _apply(self, fn, recurse: bool = True):
        # The gate has a scalar functional meaning; keep it in fp32 across
        # module-wide low-precision casts while following device moves.
        original = (
            self.gate.detach().float().clone() if self.gate is not None else None
        )
        module = super()._apply(fn, recurse=recurse)
        if original is not None:
            self.gate.data = original.to(device=self.gate.device, dtype=torch.float32)
            if self.gate.grad is not None:
                self.gate.grad.data = self.gate.grad.detach().to(
                    device=self.gate.device,
                    dtype=torch.float32,
                )
        return module

    def gate_value(self) -> torch.Tensor:
        return self.gate if self.gate is not None else self.fixed_gate

    def gate_values(self) -> tuple[torch.Tensor, torch.Tensor]:
        value = self.gate_value()
        return value, value

    def forward(
        self,
        length: int,
        *,
        dtype: torch.dtype,
    ) -> QKPreprojectionOutput:
        positional = self.basis(length).to(dtype=dtype)
        positional = positional * self.gate_value().to(dtype=dtype)
        return QKPreprojectionOutput(q=positional, k=positional)

    def reset_output_parameters(self) -> None:
        if self.gate is not None:
            with torch.no_grad():
                self.gate.fill_(float(self.config["gate_init"]))
