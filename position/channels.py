"""Composable position channels: head pipelines, Q/K coupling, and builders."""

from __future__ import annotations

import copy
import warnings
from dataclasses import dataclass
from typing import Literal

import torch

from position.basis import FrozenFourierBasis
from position.config import (
    channel_theta,
    ensure_channel_v2,
)
from position.mappers import FeatureMapper, build_mapper


class PositionChannel(torch.nn.Module):
    """Marker base class for position-channel parameter accounting and resets."""

    def reset_output_parameters(self) -> None:
        raise NotImplementedError


@dataclass
class QKPositionOutput:
    application: Literal["additive", "rotary"]
    q: torch.Tensor
    k: torch.Tensor


class GroupedLinearReadout(torch.nn.Module):
    """Grouped ``[groups, in, out]`` affine map with controlled initialization."""

    def __init__(
        self,
        groups: int,
        in_dim: int,
        out_dim: int,
        *,
        init: Literal["zeros", "identity"],
    ):
        super().__init__()
        if groups <= 0:
            raise ValueError("groups must be positive.")
        if init == "identity" and in_dim != out_dim:
            raise ValueError("Identity readout requires matching dimensions.")
        self.groups = groups
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.init = init
        self.weight = torch.nn.Parameter(torch.empty(groups, in_dim, out_dim))
        self.bias = torch.nn.Parameter(torch.zeros(groups, out_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.init == "zeros":
            torch.nn.init.zeros_(self.weight)
            torch.nn.init.zeros_(self.bias)
        else:
            with torch.no_grad():
                self.weight.zero_()
                for group in range(self.groups):
                    self.weight[group].copy_(torch.eye(self.in_dim))
            torch.nn.init.zeros_(self.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # features: [groups, length, in] or broadcastable shared-head [1, L, in]
        if features.shape[0] == 1 and self.groups == 1:
            output = torch.einsum("rd,do->ro", features[0], self.weight[0])
            return (output + self.bias[0]).unsqueeze(0)
        output = torch.einsum("grd,gdo->gro", features, self.weight)
        return output + self.bias[:, None, :]


class ScalarCurveReadout(torch.nn.Module):
    """Map ``[heads|1, extent, head_dim]`` features to ``[heads, extent]`` curves."""

    def __init__(self, groups: int, head_dim: int):
        super().__init__()
        self.groups = groups
        self.weight = torch.nn.Parameter(torch.zeros(groups, head_dim))
        self.bias = torch.nn.Parameter(torch.zeros(groups))

    def reset_parameters(self) -> None:
        torch.nn.init.zeros_(self.weight)
        torch.nn.init.zeros_(self.bias)

    def forward(self, features: torch.Tensor, heads: int) -> torch.Tensor:
        if self.groups == 1:
            curve = torch.einsum("rd,d->r", features[0], self.weight[0])
            curve = curve + self.bias[0]
            return curve.unsqueeze(0).expand(heads, -1)
        curves = torch.einsum("hrd,hd->hr", features, self.weight)
        return curves + self.bias[:, None]


class HeadCoupledFeaturePipeline(torch.nn.Module):
    """Frozen Fourier basis + grouped mapper + head layout."""

    def __init__(
        self,
        *,
        heads: int,
        head_dim: int,
        model_dim: int,
        extent: int,
        theta: float,
        head_coupling: str,
        mapper_cfg: dict,
        basis_dim: int,
    ):
        super().__init__()
        if head_coupling not in {
            "shared_head",
            "per_head_independent",
            "per_head_joint",
        }:
            raise ValueError(f"Unknown head_coupling: {head_coupling!r}")
        self.heads = heads
        self.head_dim = head_dim
        self.model_dim = model_dim
        self.extent = extent
        self.head_coupling = head_coupling
        self.basis_dim = basis_dim
        self.groups = heads if head_coupling == "per_head_independent" else 1
        mapper_dim = model_dim if head_coupling == "per_head_joint" else head_dim

        self.basis = FrozenFourierBasis(extent, basis_dim, theta)
        self.mapper = build_mapper(
            kind=mapper_cfg["kind"],
            groups=self.groups,
            input_dim=basis_dim,
            output_dim=mapper_dim,
            residual=mapper_cfg["residual"],
            rank=mapper_cfg["rank"],
            hidden_dim=mapper_cfg["hidden_dim"],
        )

    def forward(
        self,
        length: int | None = None,
        *,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        length = self.extent if length is None else int(length)
        basis = self.basis(length, dtype=dtype)
        grouped = basis.unsqueeze(0).expand(self.groups, -1, -1)
        mapped = self.mapper(grouped)
        if self.head_coupling == "per_head_independent":
            return mapped
        if self.head_coupling == "shared_head":
            return mapped.expand(self.heads, -1, -1)
        return (
            mapped.squeeze(0)
            .reshape(length, self.heads, self.head_dim)
            .permute(1, 0, 2)
            .contiguous()
        )


def _readout_groups(head_coupling: str, heads: int) -> int:
    return 1 if head_coupling == "shared_head" else heads


def _expand_shared_readout(
    output: torch.Tensor,
    head_coupling: str,
    heads: int,
) -> torch.Tensor:
    if head_coupling == "shared_head":
        return output.expand(heads, *output.shape[1:])
    return output


class QKPositionChannel(PositionChannel):
    """Q/K absolute-position channel with configurable coupling and geometry."""

    def __init__(
        self,
        config: dict,
        *,
        heads: int,
        head_dim: int,
        model_dim: int,
        extent: int,
        rope_theta: float,
    ):
        super().__init__()
        self.config = copy.deepcopy(config)
        self.heads = heads
        self.head_dim = head_dim
        self.model_dim = model_dim
        self.extent = extent
        self.application = config["application"]
        self.geometry = config["geometry"]
        self.qk_coupling = config["qk_coupling"]
        self.head_coupling = config["head_coupling"]
        theta = channel_theta(config, rope_theta)
        mapper_cfg = config["mapper"]
        basis_dim = int(config["input"]["basis_dim"])
        readout_groups = _readout_groups(self.head_coupling, heads)

        def make_pipeline() -> HeadCoupledFeaturePipeline:
            return HeadCoupledFeaturePipeline(
                heads=heads,
                head_dim=head_dim,
                model_dim=model_dim,
                extent=extent,
                theta=theta,
                head_coupling=self.head_coupling,
                mapper_cfg=mapper_cfg,
                basis_dim=basis_dim,
            )

        if self.qk_coupling == "separate":
            self.q_pipeline = make_pipeline()
            self.k_pipeline = copy.deepcopy(self.q_pipeline)
            self.pipeline = None
        else:
            self.pipeline = make_pipeline()
            self.q_pipeline = None
            self.k_pipeline = None

        self.q_add_readout = None
        self.k_add_readout = None
        self.phase_head = None
        self.q_phase_head = None
        self.k_phase_head = None

        if self.application == "additive":
            if self.qk_coupling == "shared_trunk_separate_readouts":
                self.q_add_readout = GroupedLinearReadout(
                    readout_groups,
                    head_dim,
                    head_dim,
                    init="identity",
                )
                self.k_add_readout = GroupedLinearReadout(
                    readout_groups,
                    head_dim,
                    head_dim,
                    init="identity",
                )
            elif self.qk_coupling == "separate":
                # No extra readout: independent pipelines supply the addends.
                pass
        elif self.application == "rotary":
            if self.qk_coupling == "shared":
                self.phase_head = GroupedLinearReadout(
                    readout_groups,
                    head_dim,
                    head_dim // 2,
                    init="zeros",
                )
            elif self.qk_coupling == "shared_trunk_separate_readouts":
                self.q_phase_head = GroupedLinearReadout(
                    readout_groups,
                    head_dim,
                    head_dim // 2,
                    init="zeros",
                )
                self.k_phase_head = GroupedLinearReadout(
                    readout_groups,
                    head_dim,
                    head_dim // 2,
                    init="zeros",
                )
            else:
                self.q_phase_head = GroupedLinearReadout(
                    readout_groups,
                    head_dim,
                    head_dim // 2,
                    init="zeros",
                )
                # Identical initialization without shared storage.
                self.k_phase_head = copy.deepcopy(self.q_phase_head)
        else:
            raise ValueError(f"Unsupported Q/K application: {self.application!r}")

    @property
    def uses_multiplicative_rope(self) -> bool:
        return self.application == "rotary"

    def _features_for_readout(self, features: torch.Tensor) -> torch.Tensor:
        if self.head_coupling == "shared_head":
            return features[:1]
        return features

    def _apply_phase_head(
        self,
        features: torch.Tensor,
        head: GroupedLinearReadout,
    ) -> torch.Tensor:
        grouped = self._features_for_readout(features)
        output = head(grouped)
        return _expand_shared_readout(output, self.head_coupling, self.heads)

    def _apply_add_readout(
        self,
        features: torch.Tensor,
        head: GroupedLinearReadout,
    ) -> torch.Tensor:
        grouped = self._features_for_readout(features)
        output = head(grouped)
        return _expand_shared_readout(output, self.head_coupling, self.heads)

    def forward(
        self,
        sequence_length: int,
        *,
        dtype: torch.dtype | None = None,
    ) -> QKPositionOutput:
        if sequence_length > self.extent:
            raise ValueError(
                f"Sequence length {sequence_length} exceeds Q/K position extent "
                f"{self.extent}."
            )
        if self.qk_coupling == "separate":
            q_features = self.q_pipeline(sequence_length, dtype=dtype)
            k_features = self.k_pipeline(sequence_length, dtype=dtype)
        else:
            shared = self.pipeline(sequence_length, dtype=dtype)
            q_features = shared
            k_features = shared

        if self.application == "additive":
            if self.qk_coupling == "shared":
                return QKPositionOutput("additive", q_features, k_features)
            if self.qk_coupling == "shared_trunk_separate_readouts":
                return QKPositionOutput(
                    "additive",
                    self._apply_add_readout(q_features, self.q_add_readout),
                    self._apply_add_readout(k_features, self.k_add_readout),
                )
            return QKPositionOutput("additive", q_features, k_features)

        # rotary / phase
        if self.qk_coupling == "shared":
            phase = self._apply_phase_head(q_features, self.phase_head)
            return QKPositionOutput("rotary", phase, phase)
        if self.qk_coupling == "shared_trunk_separate_readouts":
            return QKPositionOutput(
                "rotary",
                self._apply_phase_head(q_features, self.q_phase_head),
                self._apply_phase_head(k_features, self.k_phase_head),
            )
        return QKPositionOutput(
            "rotary",
            self._apply_phase_head(q_features, self.q_phase_head),
            self._apply_phase_head(k_features, self.k_phase_head),
        )

    def reset_output_parameters(self) -> None:
        for module in (
            self.phase_head,
            self.q_phase_head,
            self.k_phase_head,
            self.q_add_readout,
            self.k_add_readout,
        ):
            if module is not None:
                module.reset_parameters()

    def summarize(
        self,
        sequence_length: int,
        *,
        dtype: torch.dtype | None = None,
        q_ref: torch.Tensor | None = None,
        k_ref: torch.Tensor | None = None,
    ) -> dict[str, float]:
        """Compact finite diagnostics for Q/K channel outputs."""
        output = self.forward(sequence_length, dtype=dtype)
        metrics: dict[str, float] = {}

        def _stats(prefix: str, tensor: torch.Tensor) -> None:
            values = tensor.detach().float()
            metrics[f"{prefix}/mean"] = values.mean().item()
            metrics[f"{prefix}/std"] = values.std().item()
            metrics[f"{prefix}/rms"] = values.pow(2).mean().sqrt().item()
            metrics[f"{prefix}/abs_max"] = values.abs().max().item()

        _stats("q", output.q)
        _stats("k", output.k)
        diff = (output.q - output.k).detach().float()
        metrics["qk_diff_rms"] = diff.pow(2).mean().sqrt().item()
        if output.application == "rotary":
            _stats("phase_q", output.q)
            _stats("phase_k", output.k)
        else:
            _stats("addend_q", output.q)
            _stats("addend_k", output.k)
            if q_ref is not None:
                q_rms = q_ref.detach().float().pow(2).mean().sqrt().clamp_min(1e-12)
                metrics["addend_q_to_q_rms_ratio"] = (
                    output.q.detach().float().pow(2).mean().sqrt() / q_rms
                ).item()
            if k_ref is not None:
                k_rms = k_ref.detach().float().pow(2).mean().sqrt().clamp_min(1e-12)
                metrics["addend_k_to_k_rms_ratio"] = (
                    output.k.detach().float().pow(2).mean().sqrt() / k_rms
                ).item()
        return metrics


class LogitBiasChannel(PositionChannel):
    """Relative-distance scalar logit curves with a fixed ``[heads, extent]`` contract."""

    def __init__(
        self,
        config: dict,
        *,
        heads: int,
        head_dim: int,
        model_dim: int,
        extent: int,
        rope_theta: float,
    ):
        super().__init__()
        self.config = copy.deepcopy(config)
        self.heads = heads
        self.head_dim = head_dim
        self.extent = extent
        self.head_coupling = config["head_coupling"]
        theta = channel_theta(config, rope_theta)
        self.pipeline = HeadCoupledFeaturePipeline(
            heads=heads,
            head_dim=head_dim,
            model_dim=model_dim,
            extent=extent,
            theta=theta,
            head_coupling=self.head_coupling,
            mapper_cfg=config["mapper"],
            basis_dim=int(config["input"]["basis_dim"]),
        )
        self.scalar_head = ScalarCurveReadout(
            _readout_groups(self.head_coupling, heads),
            head_dim,
        )

    def forward(self, *, dtype: torch.dtype | None = None) -> torch.Tensor:
        features = self.pipeline(dtype=dtype)
        if self.head_coupling == "shared_head":
            return self.scalar_head(features[:1], self.heads)
        return self.scalar_head(features, self.heads)

    def reset_output_parameters(self) -> None:
        self.scalar_head.reset_parameters()


def build_qk_position_channel(
    config: dict,
    *,
    heads: int,
    head_dim: int,
    model_dim: int,
    extent: int,
    rope_theta: float,
) -> QKPositionChannel | None:
    resolved = ensure_channel_v2(
        "qk",
        config,
        model_dim=model_dim,
        heads=heads,
        rope_theta=rope_theta,
    )
    if not resolved["enabled"]:
        return None
    return QKPositionChannel(
        resolved,
        heads=heads,
        head_dim=head_dim,
        model_dim=model_dim,
        extent=extent,
        rope_theta=rope_theta,
    )


def build_logit_bias_channel(
    config: dict,
    *,
    heads: int,
    head_dim: int,
    model_dim: int,
    extent: int,
    rope_theta: float,
) -> LogitBiasChannel | None:
    resolved = ensure_channel_v2(
        "logit_bias",
        config,
        model_dim=model_dim,
        heads=heads,
        rope_theta=rope_theta,
    )
    if not resolved["enabled"]:
        return None
    return LogitBiasChannel(
        resolved,
        heads=heads,
        head_dim=head_dim,
        model_dim=model_dim,
        extent=extent,
        rope_theta=rope_theta,
    )


def adapt_legacy_position_state_dict(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Map v1 ``features.*`` / readout parameter names onto the v2 module tree.

    Only shared-coupling layouts are migrated. Incompatible shapes raise.
    """
    remapped: dict[str, torch.Tensor] = {}
    upgraded: list[str] = []

    def map_key(key: str) -> str:
        replacements = (
            (".qk_position.features.", ".qk_position.pipeline.mapper."),
            (".logit_bias.features.", ".logit_bias.pipeline.mapper."),
            (".qk_position.output_weight", ".qk_position.phase_head.weight"),
            (".qk_position.output_bias", ".qk_position.phase_head.bias"),
            (".logit_bias.readout_bias", ".logit_bias.scalar_head.bias"),
            (".logit_bias.readout", ".logit_bias.scalar_head.weight"),
        )
        new_key = key
        for old, new in replacements:
            if old in new_key:
                candidate = new_key.replace(old, new, 1)
                if candidate != new_key:
                    upgraded.append(f"{key} -> {candidate}")
                    new_key = candidate
                    break
        return new_key

    for key, value in state_dict.items():
        # Non-persistent basis buffers should not appear; skip if present.
        if key.endswith(".features.basis") or key.endswith(".pipeline.basis.basis"):
            continue
        remapped[map_key(key)] = value

    if upgraded:
        preview = "; ".join(upgraded[:8])
        extra = "" if len(upgraded) <= 8 else f" (+{len(upgraded) - 8} more)"
        warnings.warn(
            "Adapted legacy position state-dict keys for v2 modules: "
            f"{preview}{extra}",
            stacklevel=2,
        )
    return remapped


def load_position_compatible_state_dict(
    model: torch.nn.Module,
    state_dict: dict[str, torch.Tensor],
    *,
    strict: bool = True,
):
    """Load a possibly-v1 state dict into a v2 model via the adapter."""
    adapted = adapt_legacy_position_state_dict(state_dict)
    return model.load_state_dict(adapted, strict=strict)


def count_position_parameters(model: torch.nn.Module) -> dict[str, int]:
    """Count position parameters by typed modules, deduplicating by identity."""

    def _unique_numel(module: torch.nn.Module | None) -> int:
        if module is None:
            return 0
        seen: set[int] = set()
        total = 0
        for parameter in module.parameters():
            parameter_id = id(parameter)
            if parameter_id in seen:
                continue
            seen.add(parameter_id)
            total += parameter.numel()
        return total

    qk_total = 0
    logit_total = 0
    blocks = getattr(model, "blocks", None)
    if blocks is not None:
        for block in blocks:
            attn = getattr(block, "attn", None)
            if attn is None:
                continue
            qk_total += _unique_numel(getattr(attn, "qk_position", None))
            logit_total += _unique_numel(getattr(attn, "logit_bias", None))
    return {
        "qk_position_params": qk_total,
        "logit_bias_params": logit_total,
        "position_params": qk_total + logit_total,
    }
