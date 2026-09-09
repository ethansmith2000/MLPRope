"""One-shot sinusoidal carriers at the residual-stream entrance."""

from __future__ import annotations

import copy
import math

import torch
import torch.nn.functional as F

from position.basis import FrozenFourierBasis
from position.precision import PreserveFP32BuffersMixin


INPUT_SINUSOID_MODES = {
    "tied_scalar",
    "low_rank_linear_residual",
    "dense_linear_residual",
    "residual_mlp",
    "per_pair_amplitude",
}

INPUT_SINUSOID_DEFAULTS = {
    "enabled": False,
    "mode": "tied_scalar",
    "basis_dim": None,
    "theta": None,
    "gate_init": 1.0,
    "learnable_gate": True,
    "rank": 32,
    "hidden_dim": None,
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

    mode = normalized["mode"]
    if not isinstance(mode, str):
        raise TypeError("input_sinusoid.mode must be a string")
    if mode not in INPUT_SINUSOID_MODES:
        raise ValueError(
            "input_sinusoid.mode must be one of "
            f"{sorted(INPUT_SINUSOID_MODES)}"
        )

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
    rank = normalized["rank"]
    if isinstance(rank, bool) or not isinstance(rank, int):
        raise TypeError("input_sinusoid.rank must be an integer")
    if rank <= 0:
        raise ValueError("input_sinusoid.rank must be positive")
    if mode == "low_rank_linear_residual" and rank > model_dim:
        raise ValueError(
            "low-rank input_sinusoid.rank must be no larger than model_dim"
        )
    normalized["rank"] = rank
    hidden_dim = normalized["hidden_dim"]
    hidden_dim = model_dim if hidden_dim is None else hidden_dim
    if isinstance(hidden_dim, bool) or not isinstance(hidden_dim, int):
        raise TypeError("input_sinusoid.hidden_dim must be an integer or null")
    if hidden_dim <= 0:
        raise ValueError("input_sinusoid.hidden_dim must be positive")
    normalized["hidden_dim"] = hidden_dim
    return normalized


class InputSinusoidPosition(PreserveFP32BuffersMixin, torch.nn.Module):
    """Return a learned full-width Fourier vector for every input position.

    ``tied_scalar`` is the historical one-scalar input carrier. The linear and
    MLP modes add a zero-initialized residual around that exact anchor. The
    pair-amplitude mode learns one direct signed coefficient per frequency,
    shared by the corresponding cosine/sine coordinates so it cannot create a
    phase shift implicitly.
    """

    _fp32_buffer_names = ("fixed_gate", "fixed_pair_amplitude")

    def __init__(self, config: dict, *, model_dim: int, extent: int) -> None:
        super().__init__()
        self.config = copy.deepcopy(config)
        self.model_dim = model_dim
        self.extent = extent
        self.mode = config["mode"]
        self.basis = FrozenFourierBasis(
            extent=extent,
            basis_dim=model_dim,
            theta=float(config["theta"]),
        )
        self.down = None
        self.up = None
        self.dense = None
        if self.mode == "low_rank_linear_residual":
            rank = int(config["rank"])
            self.down = torch.nn.Linear(model_dim, rank, bias=False)
            self.up = torch.nn.Linear(rank, model_dim, bias=False)
        elif self.mode == "dense_linear_residual":
            self.dense = torch.nn.Linear(model_dim, model_dim, bias=False)
        elif self.mode == "residual_mlp":
            hidden_dim = int(config["hidden_dim"])
            self.down = torch.nn.Linear(model_dim, hidden_dim, bias=False)
            self.up = torch.nn.Linear(hidden_dim, model_dim, bias=False)

        gate_init = float(config["gate_init"])
        if self.mode == "per_pair_amplitude":
            pair_amplitude = torch.full(
                (model_dim // 2,), gate_init, dtype=torch.float32
            )
            self.register_parameter("gate", None)
            self.register_buffer("fixed_gate", None)
            if config["learnable_gate"]:
                self.pair_amplitude = torch.nn.Parameter(pair_amplitude)
                self.register_buffer("fixed_pair_amplitude", None)
            else:
                self.register_parameter("pair_amplitude", None)
                self.register_buffer("fixed_pair_amplitude", pair_amplitude)
        else:
            gate = torch.tensor(gate_init, dtype=torch.float32)
            self.register_parameter("pair_amplitude", None)
            self.register_buffer("fixed_pair_amplitude", None)
            if config["learnable_gate"]:
                self.gate = torch.nn.Parameter(gate)
                self.register_buffer("fixed_gate", None)
            else:
                self.register_parameter("gate", None)
                self.register_buffer("fixed_gate", gate)

    def _apply(self, fn, recurse: bool = True):
        originals = {
            name: parameter.detach().float().clone()
            for name in ("gate", "pair_amplitude")
            if (parameter := getattr(self, name, None)) is not None
        }
        module = super()._apply(fn, recurse=recurse)
        for name, original in originals.items():
            parameter = getattr(self, name)
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

    def gate_value(self) -> torch.Tensor:
        if self.mode == "per_pair_amplitude":
            return self.amplitude_values()
        return self.gate if self.gate is not None else self.fixed_gate

    def amplitude_values(self) -> torch.Tensor:
        if self.mode == "per_pair_amplitude":
            return (
                self.pair_amplitude
                if self.pair_amplitude is not None
                else self.fixed_pair_amplitude
            )
        return self.gate if self.gate is not None else self.fixed_gate

    def _carrier_components(
        self,
        length: int,
        *,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if length > self.extent:
            raise ValueError(
                f"Sequence length {length} exceeds input-sinusoid extent "
                f"{self.extent}."
            )
        basis = self.basis(length, dtype=dtype)
        if self.mode == "per_pair_amplitude":
            amplitude = self.amplitude_values().to(dtype=dtype)
            carrier = (
                basis.unflatten(-1, (self.model_dim // 2, 2))
                * amplitude[:, None]
            ).flatten(-2)
            return carrier, None

        anchor = basis * self.gate_value().to(dtype=dtype)
        residual = None
        if self.mode == "low_rank_linear_residual":
            residual = self.up(self.down(basis))
        elif self.mode == "dense_linear_residual":
            residual = self.dense(basis)
        elif self.mode == "residual_mlp":
            residual = self.up(F.gelu(self.down(basis)))
        return anchor, residual

    def forward(self, length: int, *, dtype: torch.dtype) -> torch.Tensor:
        anchor, residual = self._carrier_components(length, dtype=dtype)
        carrier = anchor if residual is None else anchor + residual
        return carrier

    @torch.no_grad()
    def diagnostics(self, length: int) -> dict[str, float]:
        anchor, residual = self._carrier_components(length, dtype=torch.float32)
        carrier = anchor if residual is None else anchor + residual
        values = self.amplitude_values().detach().float()
        metrics = {
            "carrier_rms": carrier.detach().float().square().mean().sqrt().item(),
            "amplitude_mean": values.mean().item(),
            "amplitude_rms": values.square().mean().sqrt().item(),
            "amplitude_min": values.min().item(),
            "amplitude_max": values.max().item(),
        }
        if values.numel() == 1:
            metrics["gate"] = values.item()
        if residual is not None:
            anchor_rms = anchor.detach().float().square().mean().sqrt()
            residual_rms = residual.detach().float().square().mean().sqrt()
            metrics["anchor_rms"] = anchor_rms.item()
            metrics["adapter_rms"] = residual_rms.item()
            metrics["adapter_to_anchor_rms_ratio"] = (
                (residual_rms / anchor_rms).item()
                if anchor_rms.item() > 0
                else 0.0
            )
        return metrics

    def reset_output_parameters(self) -> None:
        if self.gate is not None:
            with torch.no_grad():
                self.gate.fill_(float(self.config["gate_init"]))
        if self.pair_amplitude is not None:
            with torch.no_grad():
                self.pair_amplitude.fill_(float(self.config["gate_init"]))
        if self.up is not None:
            with torch.no_grad():
                self.up.weight.zero_()
        if self.dense is not None:
            with torch.no_grad():
                self.dense.weight.zero_()
