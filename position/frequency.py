"""Configuration and forward maps for learned base-RoPE frequencies."""

from __future__ import annotations

import copy
import math

import torch
import torch.nn.functional as F


ROPE_FREQUENCY_DEFAULTS = {
    "mode": "fixed",
    "head_coupling": "shared",
    "parameterization": "exp",
    "log_bound": 1.0,
    "source": "normalized_residual",
    "mapper": "linear",
    "rank": 32,
    "qk_coupling": "shared",
    "phase_bound": 1.0,
    "reference_length": 1024,
}

STATIC_ROPE_FREQUENCY_PARAMETERIZATIONS = {
    "exp",
    "exp_full_ste",
    "softplus",
    "additive",
    "bounded_log",
}
CONTENT_ROPE_FREQUENCY_PARAMETERIZATIONS = {
    "horizon_bounded",
    "phase_residual",
}
ROPE_FREQUENCY_PARAMETERIZATIONS = (
    STATIC_ROPE_FREQUENCY_PARAMETERIZATIONS
    | CONTENT_ROPE_FREQUENCY_PARAMETERIZATIONS
)
ROPE_FREQUENCY_MAPPERS = {
    "linear",
    "low_rank_linear",
    "low_rank_silu",
}

_LEGACY_TO_CONFIG = {
    "fixed": ("fixed", "shared"),
    "layer_shared": ("static", "shared"),
    "layer_head": ("static", "per_head"),
    "content": ("content", "per_head"),
}
_SOFTPLUS_UNIT_BIAS = math.log(math.e - 1.0)


def legacy_rope_frequency_mode(config: dict) -> str:
    """Return the old flat mode spelling for checkpoint/config compatibility."""

    if config["mode"] == "fixed":
        return "fixed"
    if config["mode"] == "content":
        return "content"
    return {
        "shared": "layer_shared",
        "per_head": "layer_head",
    }[config["head_coupling"]]


def normalize_rope_frequency_config(
    config: dict | None = None,
    *,
    legacy_mode: str = "fixed",
) -> dict:
    """Validate the structured frequency config and upgrade a legacy mode.

    A non-fixed legacy mode is considered explicit and must agree with a
    structured config. ``fixed`` remains the neutral default so direct callers
    can supply only the new object.
    """

    if legacy_mode not in _LEGACY_TO_CONFIG:
        raise ValueError(
            "rope_frequency_mode must be 'fixed', 'layer_shared', "
            "'layer_head', or 'content'"
        )
    legacy_config = copy.deepcopy(ROPE_FREQUENCY_DEFAULTS)
    legacy_config["mode"], legacy_config["head_coupling"] = (
        _LEGACY_TO_CONFIG[legacy_mode]
    )
    if config is None:
        return legacy_config
    if not isinstance(config, dict):
        raise TypeError("rope_frequency must be an object")
    unknown = set(config) - set(ROPE_FREQUENCY_DEFAULTS)
    if unknown:
        raise ValueError(
            f"Unknown rope_frequency keys: {sorted(unknown)}"
        )
    normalized = copy.deepcopy(ROPE_FREQUENCY_DEFAULTS)
    normalized.update(config)
    if normalized["mode"] not in {"fixed", "static", "content"}:
        raise ValueError(
            "rope_frequency.mode must be 'fixed', 'static', or 'content'"
        )
    if normalized["head_coupling"] not in {"shared", "per_head"}:
        raise ValueError(
            "rope_frequency.head_coupling must be 'shared' or 'per_head'"
        )
    if normalized["parameterization"] not in ROPE_FREQUENCY_PARAMETERIZATIONS:
        raise ValueError(
            "unsupported rope_frequency.parameterization: "
            f"{normalized['parameterization']!r}"
        )
    log_bound = normalized["log_bound"]
    if isinstance(log_bound, bool) or not isinstance(log_bound, (int, float)):
        raise TypeError("rope_frequency.log_bound must be a number")
    normalized["log_bound"] = float(log_bound)
    if not math.isfinite(normalized["log_bound"]) or normalized["log_bound"] <= 0:
        raise ValueError("rope_frequency.log_bound must be finite and positive")
    if normalized["source"] != "normalized_residual":
        raise ValueError(
            "only rope_frequency.source='normalized_residual' is implemented"
        )
    if normalized["mapper"] not in ROPE_FREQUENCY_MAPPERS:
        raise ValueError(
            f"unsupported rope_frequency.mapper: {normalized['mapper']!r}"
        )
    rank = normalized["rank"]
    if isinstance(rank, bool) or not isinstance(rank, int):
        raise TypeError("rope_frequency.rank must be an integer")
    if rank <= 0:
        raise ValueError("rope_frequency.rank must be positive")
    if normalized["qk_coupling"] != "shared":
        raise ValueError(
            "only rope_frequency.qk_coupling='shared' is implemented"
        )
    for key in ("phase_bound",):
        value = normalized[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"rope_frequency.{key} must be a number")
        normalized[key] = float(value)
        if not math.isfinite(normalized[key]) or normalized[key] <= 0:
            raise ValueError(
                f"rope_frequency.{key} must be finite and positive"
            )
    reference_length = normalized["reference_length"]
    if isinstance(reference_length, bool) or not isinstance(reference_length, int):
        raise TypeError("rope_frequency.reference_length must be an integer")
    if reference_length <= 0:
        raise ValueError("rope_frequency.reference_length must be positive")
    if normalized["mode"] == "fixed":
        if normalized["parameterization"] != "exp":
            raise ValueError(
                "fixed rope_frequency requires parameterization='exp'"
            )
        if normalized["head_coupling"] != "shared":
            raise ValueError(
                "fixed rope_frequency requires head_coupling='shared'"
            )
    elif normalized["mode"] == "static":
        if (
            normalized["parameterization"]
            not in STATIC_ROPE_FREQUENCY_PARAMETERIZATIONS
        ):
            raise ValueError(
                "static rope_frequency requires a static parameterization"
            )
    else:
        if (
            normalized["parameterization"]
            not in CONTENT_ROPE_FREQUENCY_PARAMETERIZATIONS
        ):
            raise ValueError(
                "content rope_frequency requires horizon_bounded or "
                "phase_residual parameterization"
            )
    if legacy_mode != "fixed" and (
        legacy_rope_frequency_mode(normalized) != legacy_mode
    ):
        raise ValueError(
            "rope_frequency conflicts with legacy rope_frequency_mode"
        )
    return normalized


def parameterize_rope_frequencies(
    base: torch.Tensor,
    raw: torch.Tensor,
    config: dict,
) -> torch.Tensor:
    """Map a zero-anchored raw parameter to effective fp32 frequencies."""

    base = base.float()
    raw = raw.float()
    parameterization = config["parameterization"]
    if parameterization == "exp":
        return base * raw.exp()
    if parameterization == "exp_full_ste":
        forward_value = base * raw.exp()
        # Exact exponential forward, identity d(omega)/d(raw) backward. This
        # removes both exp(raw) and base-frequency scaling from the Jacobian.
        return forward_value.detach() + (raw - raw.detach())
    if parameterization == "softplus":
        multiplier = F.softplus(raw + _SOFTPLUS_UNIT_BIAS)
        return base * multiplier
    if parameterization == "additive":
        return base + raw
    if parameterization == "bounded_log":
        log_multiplier = float(config["log_bound"]) * raw.tanh()
        return base * log_multiplier.exp()
    raise AssertionError(f"Unhandled parameterization: {parameterization}")


class RopeFrequencyController(torch.nn.Module):
    """Historical token-local phase controller retained for config provenance.

    Phase-23 closed this local, independently-per-pair family as a material
    result. New dynamic work belongs in :mod:`position.clock`, where one
    bounded positive coordinate is shared across the fixed frequency planes.
    """

    def __init__(
        self,
        *,
        model_dim: int,
        heads: int,
        pair_dim: int,
        config: dict,
    ) -> None:
        super().__init__()
        self.model_dim = model_dim
        self.heads = heads
        self.pair_dim = pair_dim
        self.config = copy.deepcopy(config)
        self.groups = heads if config["head_coupling"] == "per_head" else 1
        output_dim = self.groups * pair_dim
        mapper = config["mapper"]
        if mapper == "linear":
            self.down = None
            self.output = torch.nn.Linear(model_dim, output_dim, bias=True)
        else:
            self.down = torch.nn.Linear(
                model_dim,
                int(config["rank"]),
                bias=True,
            )
            self.output = torch.nn.Linear(
                int(config["rank"]),
                output_dim,
                bias=True,
            )
        self.reset_output_parameters()

    def reset_output_parameters(self) -> None:
        """Restore the exact fixed-RoPE anchor after global initialization."""

        torch.nn.init.zeros_(self.output.weight)
        if self.output.bias is not None:
            torch.nn.init.zeros_(self.output.bias)

    def raw_output(self, normalized_residual: torch.Tensor) -> torch.Tensor:
        if normalized_residual.ndim != 3:
            raise ValueError(
                "RopeFrequencyController expects normalized residual [B,L,D]"
            )
        if normalized_residual.shape[-1] != self.model_dim:
            raise ValueError(
                "RopeFrequencyController input width does not match model_dim"
            )
        hidden = normalized_residual
        if self.down is not None:
            hidden = self.down(hidden)
            if self.config["mapper"] == "low_rank_silu":
                hidden = F.silu(hidden)
        raw = self.output(hidden)
        batch, length, _ = raw.shape
        raw = raw.view(batch, length, self.groups, self.pair_dim)
        raw = raw.permute(0, 2, 1, 3)
        if self.groups == 1:
            raw = raw.expand(-1, self.heads, -1, -1)
        return raw

    def phase_delta(self, normalized_residual: torch.Tensor) -> torch.Tensor:
        raw = self.raw_output(normalized_residual).float()
        bounded = float(self.config["phase_bound"]) * raw.tanh()
        if self.config["parameterization"] == "phase_residual":
            return bounded
        length = normalized_residual.shape[1]
        positions = torch.arange(
            length,
            device=raw.device,
            dtype=torch.float32,
        )
        position_scale = positions / float(self.config["reference_length"])
        return bounded * position_scale[None, None, :, None]

    @torch.no_grad()
    def diagnostics(self, normalized_residual: torch.Tensor) -> dict[str, float]:
        raw = self.raw_output(normalized_residual).detach().float()
        phase = self.phase_delta(normalized_residual).detach().float()
        token_std = raw.std(dim=2, unbiased=False)
        return {
            "controller_raw_mean": raw.mean().item(),
            "controller_raw_rms": raw.square().mean().sqrt().item(),
            "controller_raw_abs_max": raw.abs().max().item(),
            "controller_token_std_mean": token_std.mean().item(),
            "dynamic_phase_rms": phase.square().mean().sqrt().item(),
            "dynamic_phase_p95": torch.quantile(phase.abs(), 0.95).item(),
            "dynamic_phase_abs_max": phase.abs().max().item(),
        }
