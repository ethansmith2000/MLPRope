"""Attention-local sinusoidal adapters around the Q/K projections."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F

from position.basis import FrozenFourierBasis
from position.precision import PreserveFP32BuffersMixin


QKPreprojectionMode = Literal[
    "tied_scalar",
    "low_rank_premap",
    "low_rank_qk_replace",
    "low_rank_qk_residual",
    "low_rank_qk_shared_residual",
    "dense_premap_residual",
    "dense_qk_shared_residual",
    "dense_qk_residual",
    "low_rank_qk_mlp_residual",
]
QK_PREPROJECTION_MODES = {
    "tied_scalar",
    "low_rank_premap",
    "low_rank_qk_replace",
    "low_rank_qk_residual",
    "low_rank_qk_shared_residual",
    "dense_premap_residual",
    "dense_qk_shared_residual",
    "dense_qk_residual",
    "low_rank_qk_mlp_residual",
}
QK_PREPROJECTION_LOW_RANK_MODES = {
    "low_rank_premap",
    "low_rank_qk_replace",
    "low_rank_qk_residual",
    "low_rank_qk_shared_residual",
    "low_rank_qk_mlp_residual",
}

# Analysis-only counterfactuals for a trained projected-space carrier. These
# are runtime attributes, not configuration or state-dict fields, so training
# remains bit-for-bit unchanged unless an evaluator opts in after ``eval()``.
QK_PREPROJECTION_EVALUATION_INTERVENTIONS = {
    "full",
    "direct_mean_only",
    "direct_mean_removed",
    "direct_zero",
    "scalar_zero",
    "all_zero",
}

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
    "rank": 32,
    # Optional rank calibration for the zero-initialized projected-space
    # readout only. The bottleneck and scalar anchor retain the base position LR.
    "readout_lr_multiplier": 1.0,
    "compensate_readout_weight_decay": True,
    # Model-level ablation axes. ``active_layers=None`` means every layer.
    "gate_sharing": "per_layer",
    "active_layers": None,
}


@dataclass(frozen=True)
class QKPreprojectionOutput:
    """Position contributions on either side of the Q/K projections.

    ``*_input`` is added to the normalized residual-stream input before
    ``W_q``/``W_k``. ``*_projected`` is added to the concatenated all-head
    projection output before head splitting, QK normalization, and RoPE.
    """

    q_input: torch.Tensor | None = None
    k_input: torch.Tensor | None = None
    q_projected: torch.Tensor | None = None
    k_projected: torch.Tensor | None = None

    def carrier_tensors(self) -> tuple[torch.Tensor, ...]:
        return tuple(
            value
            for value in (
                self.q_input,
                self.k_input,
                self.q_projected,
                self.k_projected,
            )
            if value is not None
        )


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
    """Validate and resolve static pre-Q/K sinusoidal adapters.

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
            "the Phase 33--37 null confirmations. Use the fixed-frequency "
            "tied scalar or low-rank pathway modes, or recover the historical "
            "implementation from git history."
        )
    if not normalized["enabled"]:
        # A disabled archival block has no model effect. Canonicalize it so a
        # historical resolved config does not retain dormant active settings.
        mode = "tied_scalar"
        normalized["mode"] = mode
    if mode not in QK_PREPROJECTION_MODES:
        raise ValueError(
            "qk_preprojection.mode must be one of "
            f"{sorted(QK_PREPROJECTION_MODES)}; historical shape modes were "
            "removed after Phases 33--37"
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
    rank = normalized["rank"]
    if isinstance(rank, bool) or not isinstance(rank, int):
        raise TypeError("qk_preprojection.rank must be an integer")
    if rank <= 0:
        raise ValueError("qk_preprojection.rank must be positive")
    if mode in QK_PREPROJECTION_LOW_RANK_MODES and rank > model_dim:
        raise ValueError(
            "low-rank qk_preprojection.rank must be no larger than model_dim"
        )
    normalized["rank"] = rank
    readout_lr_multiplier = normalized["readout_lr_multiplier"]
    if isinstance(readout_lr_multiplier, bool) or not isinstance(
        readout_lr_multiplier, (int, float)
    ):
        raise TypeError("qk_preprojection.readout_lr_multiplier must be a number")
    readout_lr_multiplier = float(readout_lr_multiplier)
    if not math.isfinite(readout_lr_multiplier) or readout_lr_multiplier <= 0:
        raise ValueError(
            "qk_preprojection.readout_lr_multiplier must be finite and positive"
        )
    normalized["readout_lr_multiplier"] = readout_lr_multiplier
    if not isinstance(normalized["compensate_readout_weight_decay"], bool):
        raise TypeError(
            "qk_preprojection.compensate_readout_weight_decay must be a boolean"
        )
    projected_readout_modes = {
        "low_rank_qk_replace",
        "low_rank_qk_residual",
        "low_rank_qk_shared_residual",
        "low_rank_qk_mlp_residual",
        "dense_qk_shared_residual",
        "dense_qk_residual",
    }
    if readout_lr_multiplier != 1.0 and mode not in projected_readout_modes:
        raise ValueError(
            "qk_preprojection.readout_lr_multiplier may differ from 1 only "
            "for a projected-space Q/K readout mode"
        )
    if mode == "low_rank_qk_replace":
        # This arm is nested at the standard-RoPE baseline: there is no
        # pre-projection identity carrier and therefore no meaningful gate.
        normalized["gate_init"] = 0.0
        normalized["learnable_gate"] = False
        normalized["gate_sharing"] = "per_layer"
    gate_sharing = normalized["gate_sharing"]
    if gate_sharing not in {"per_layer", "global"}:
        raise ValueError(
            "qk_preprojection.gate_sharing must be 'per_layer' or 'global'"
        )
    active_layers = normalized["active_layers"]
    if active_layers is not None:
        if not isinstance(active_layers, list):
            raise TypeError("qk_preprojection.active_layers must be a list or null")
        resolved_layers = []
        for layer_idx in active_layers:
            if isinstance(layer_idx, bool) or not isinstance(layer_idx, int):
                raise TypeError(
                    "qk_preprojection.active_layers must contain only integers"
                )
            if layer_idx < 0:
                raise ValueError(
                    "qk_preprojection.active_layers must be non-negative"
                )
            if layer_idx not in resolved_layers:
                resolved_layers.append(layer_idx)
        normalized["active_layers"] = sorted(resolved_layers)
    return normalized


class QKPreprojectionPosition(PreserveFP32BuffersMixin, torch.nn.Module):
    """Add a static Fourier carrier around the Q and K projections.

    The low-rank modes use a shared ``model_dim -> rank`` positional trunk.
    Premap modes return one model-space residual before W_q/W_k. Q/K modes
    provide shared or separate projected-space readouts. Dense and nonlinear
    variants preserve the same exact scalar anchor at initialization. V and
    the residual stream are untouched.
    """

    _fp32_buffer_names = ("fixed_gate",)
    _fp32_parameter_names = ("gate",)

    def __init__(self, config: dict, *, model_dim: int, extent: int) -> None:
        super().__init__()
        self.config = copy.deepcopy(config)
        self.model_dim = model_dim
        self.extent = extent
        self.mode: QKPreprojectionMode = config["mode"]
        self.evaluation_intervention = "full"
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

        self.down = None
        self.shared_up = None
        self.q_up = None
        self.k_up = None
        if self.mode in QK_PREPROJECTION_LOW_RANK_MODES:
            rank = int(config["rank"])
            self.down = torch.nn.Linear(model_dim, rank, bias=False)
            if self.mode in {"low_rank_premap", "low_rank_qk_shared_residual"}:
                self.shared_up = torch.nn.Linear(rank, model_dim, bias=False)
            else:
                self.q_up = torch.nn.Linear(rank, model_dim, bias=False)
                self.k_up = torch.nn.Linear(rank, model_dim, bias=False)
        elif self.mode == "dense_premap_residual":
            self.shared_up = torch.nn.Linear(model_dim, model_dim, bias=False)
        elif self.mode == "dense_qk_shared_residual":
            self.shared_up = torch.nn.Linear(model_dim, model_dim, bias=False)
        elif self.mode == "dense_qk_residual":
            self.q_up = torch.nn.Linear(model_dim, model_dim, bias=False)
            self.k_up = torch.nn.Linear(model_dim, model_dim, bias=False)

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

    def set_evaluation_intervention(self, intervention: str) -> None:
        """Select a trained-carrier counterfactual for evaluation only.

        ``direct_mean_only`` broadcasts the positional mean of each direct
        Q/K readout; ``direct_mean_removed`` retains only its centered
        position-varying component. The scalar interventions act on the
        model-space sinusoidal anchor. No parameter or persistent buffer is
        modified.
        """
        if intervention not in QK_PREPROJECTION_EVALUATION_INTERVENTIONS:
            raise ValueError(
                "Unknown Q/K preprojection evaluation intervention "
                f"{intervention!r}; expected one of "
                f"{sorted(QK_PREPROJECTION_EVALUATION_INTERVENTIONS)}"
            )
        self.evaluation_intervention = intervention

    def _intervene_direct(self, value: torch.Tensor) -> torch.Tensor:
        intervention = self.evaluation_intervention
        if intervention in {"direct_zero", "all_zero"}:
            return torch.zeros_like(value)
        mean = value.mean(dim=0, keepdim=True)
        if intervention == "direct_mean_only":
            return mean.expand_as(value)
        if intervention == "direct_mean_removed":
            return value - mean
        return value

    def forward(
        self,
        length: int,
        *,
        dtype: torch.dtype,
    ) -> QKPreprojectionOutput:
        if self.training and self.evaluation_intervention != "full":
            raise RuntimeError(
                "Q/K preprojection counterfactuals are evaluation-only; "
                "call eval() before using one"
            )
        basis = self.basis(length).to(dtype=dtype)
        anchor = basis * self.gate_value().to(dtype=dtype)
        if self.evaluation_intervention in {"scalar_zero", "all_zero"}:
            anchor = torch.zeros_like(anchor)
        if self.mode == "tied_scalar":
            return QKPreprojectionOutput(q_input=anchor, k_input=anchor)

        if self.mode == "dense_premap_residual":
            positional = anchor + self.shared_up(basis)
            return QKPreprojectionOutput(q_input=positional, k_input=positional)

        if self.mode == "dense_qk_shared_residual":
            direct = self._intervene_direct(self.shared_up(basis))
            return QKPreprojectionOutput(
                q_input=anchor,
                k_input=anchor,
                q_projected=direct,
                k_projected=direct,
            )

        hidden = basis if self.mode == "dense_qk_residual" else self.down(basis)
        if self.mode == "low_rank_premap":
            positional = anchor + self.shared_up(hidden)
            return QKPreprojectionOutput(q_input=positional, k_input=positional)

        if self.mode == "low_rank_qk_shared_residual":
            direct = self._intervene_direct(self.shared_up(hidden))
            return QKPreprojectionOutput(
                q_input=anchor,
                k_input=anchor,
                q_projected=direct,
                k_projected=direct,
            )

        if self.mode == "low_rank_qk_mlp_residual":
            hidden = F.gelu(hidden)

        q_projected = self._intervene_direct(self.q_up(hidden))
        k_projected = self._intervene_direct(self.k_up(hidden))
        if self.mode == "low_rank_qk_replace":
            return QKPreprojectionOutput(
                q_projected=q_projected,
                k_projected=k_projected,
            )
        if self.mode in {
            "low_rank_qk_residual",
            "dense_qk_residual",
            "low_rank_qk_mlp_residual",
        }:
            return QKPreprojectionOutput(
                q_input=anchor,
                k_input=anchor,
                q_projected=q_projected,
                k_projected=k_projected,
            )
        raise AssertionError(f"Unhandled qk_preprojection mode {self.mode!r}")

    def reset_output_parameters(self) -> None:
        if self.gate is not None:
            with torch.no_grad():
                self.gate.fill_(float(self.config["gate_init"]))
        with torch.no_grad():
            for module in (self.shared_up, self.q_up, self.k_up):
                if module is not None:
                    module.weight.zero_()
