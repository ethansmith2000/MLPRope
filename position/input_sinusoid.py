"""Minimal one-shot sinusoidal carrier at the residual-stream entrance."""

from __future__ import annotations

import copy
import math

import torch

from position.basis import FrozenFourierBasis
from position.precision import PreserveFP32BuffersMixin


INPUT_SINUSOID_DEFAULTS = {
    "enabled": False,
    "basis_dim": None,
    "theta": None,
    "gate_init": 1.0,
    "learnable_gate": True,
}


def normalize_input_sinusoid_config(
    config: dict | None,
    *,
    model_dim: int,
    rope_theta: float,
) -> dict:
    """Validate the intentionally narrow residual-input control."""
    if config is None:
        config = {}
    if not isinstance(config, dict):
        raise TypeError("input_sinusoid must be an object")
    unknown = set(config) - set(INPUT_SINUSOID_DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown input_sinusoid keys: {sorted(unknown)}")

    normalized = copy.deepcopy(INPUT_SINUSOID_DEFAULTS)
    normalized.update(config)
    if not isinstance(normalized["enabled"], bool):
        raise TypeError("input_sinusoid.enabled must be a boolean")

    basis_dim = normalized["basis_dim"]
    basis_dim = model_dim if basis_dim is None else int(basis_dim)
    if basis_dim != model_dim:
        raise ValueError("input_sinusoid requires basis_dim=model_dim")
    if basis_dim <= 0 or basis_dim % 2:
        raise ValueError("input_sinusoid requires a positive even model_dim")
    normalized["basis_dim"] = basis_dim

    theta = rope_theta if normalized["theta"] is None else float(normalized["theta"])
    if not math.isfinite(theta) or theta <= 0:
        raise ValueError("input_sinusoid.theta must be finite and positive")
    normalized["theta"] = theta

    gate_init = normalized["gate_init"]
    if isinstance(gate_init, bool) or not isinstance(gate_init, (int, float)):
        raise TypeError("input_sinusoid.gate_init must be a number")
    normalized["gate_init"] = float(gate_init)
    if not math.isfinite(normalized["gate_init"]):
        raise ValueError("input_sinusoid.gate_init must be finite")
    if not isinstance(normalized["learnable_gate"], bool):
        raise TypeError("input_sinusoid.learnable_gate must be a boolean")
    return normalized


class InputSinusoidPosition(PreserveFP32BuffersMixin, torch.nn.Module):
    """Return one gated full-width Fourier vector for every input position."""

    _fp32_buffer_names = ("fixed_gate",)
    _fp32_parameter_names = ("gate",)

    def __init__(self, config: dict, *, model_dim: int, extent: int) -> None:
        super().__init__()
        self.config = copy.deepcopy(config)
        self.model_dim = model_dim
        self.extent = extent
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

    def forward(self, length: int, *, dtype: torch.dtype) -> torch.Tensor:
        if length > self.extent:
            raise ValueError(
                f"Sequence length {length} exceeds input-sinusoid extent "
                f"{self.extent}."
            )
        carrier = self.basis(length).to(dtype=dtype)
        return carrier * self.gate_value().to(dtype=dtype)

    def reset_output_parameters(self) -> None:
        if self.gate is not None:
            with torch.no_grad():
                self.gate.fill_(float(self.config["gate_init"]))
