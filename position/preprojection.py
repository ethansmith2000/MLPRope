"""Attention-local sinusoidal injection before the Q/K projections."""

from __future__ import annotations

import copy
import math

import torch

from position.basis import FrozenFourierBasis


QK_PREPROJECTION_DEFAULTS = {
    "enabled": False,
    "basis_dim": None,
    "theta": None,
    "gate_init": 1.0,
    "learnable_gate": True,
}


def normalize_qk_preprojection_config(
    config: dict | None,
    *,
    model_dim: int,
    rope_theta: float,
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
    basis_dim = normalized["basis_dim"]
    basis_dim = model_dim if basis_dim is None else int(basis_dim)
    if basis_dim != model_dim:
        raise ValueError(
            "qk_preprojection currently requires basis_dim=model_dim; this "
            "keeps the position projection tied exactly to W_q/W_k"
        )
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


class QKPreprojectionPosition(torch.nn.Module):
    """Return ``gate * z(p)`` for addition only to the Q/K projection inputs.

    If ``u`` is the block-normalized residual, the resulting query is
    ``W_q(u + gate*z) = W_q u + gate*W_q z``.  This is therefore a tied
    additive carrier rather than a residual-stream positional write.
    """

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

    def gate_value(self) -> torch.Tensor:
        return self.gate if self.gate is not None else self.fixed_gate

    def forward(self, length: int, *, dtype: torch.dtype) -> torch.Tensor:
        basis = self.basis(length, dtype=dtype)
        return basis * self.gate_value().to(dtype=dtype)

    def reset_output_parameters(self) -> None:
        if self.gate is not None:
            with torch.no_grad():
                self.gate.fill_(float(self.config["gate_init"]))
