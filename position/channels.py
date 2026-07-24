"""Composable position channels: head pipelines, Q/K coupling, and builders."""

from __future__ import annotations

import copy
import math
import warnings
from dataclasses import dataclass
from typing import Literal

import torch

from position.basis import build_position_basis
from position.config import (
    channel_theta,
    ensure_channel_v2,
    normalize_attention_write_config,
    normalize_residual_stream_config,
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
    q_scale: torch.Tensor | None = None
    k_scale: torch.Tensor | None = None


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


class GroupedContentConditioner(torch.nn.Module):
    """Cache-safe local conditioning of a positional output by Q/K content."""

    def __init__(
        self,
        *,
        kind: Literal["local_residual", "content_gate"],
        groups: int,
        content_dim: int,
        output_dim: int,
        hidden_dim: int,
        gate_init: float,
    ):
        super().__init__()
        self.kind = kind
        self.groups = groups
        self.content_dim = content_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.gate_init = float(gate_init)
        if kind == "local_residual":
            self.down = torch.nn.Parameter(
                torch.empty(groups, content_dim + output_dim, hidden_dim)
            )
            self.down_bias = torch.nn.Parameter(torch.zeros(groups, hidden_dim))
            self.up = torch.nn.Parameter(
                torch.zeros(groups, hidden_dim, output_dim)
            )
            self.up_bias = torch.nn.Parameter(torch.zeros(groups, output_dim))
            for weight in self.down:
                torch.nn.init.xavier_normal_(weight)
        elif kind == "content_gate":
            self.gate_weight = torch.nn.Parameter(
                torch.zeros(groups, content_dim, output_dim)
            )
            self.gate_bias = torch.nn.Parameter(torch.zeros(groups, output_dim))
        else:
            raise ValueError(f"Unknown content conditioner kind: {kind!r}")

    def _linear(
        self,
        values: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
    ) -> torch.Tensor:
        if self.groups == 1:
            output = torch.einsum("bhld,do->bhlo", values, weight[0])
            return output + bias[0]
        output = torch.einsum("bhld,hdo->bhlo", values, weight)
        return output + bias[None, :, None, :]

    def forward(
        self,
        base: torch.Tensor,
        content: torch.Tensor,
    ) -> torch.Tensor:
        if content.ndim != 4:
            raise ValueError(
                "Content conditioning expects [batch, heads, sequence, head_dim]"
            )
        if base.ndim == 3:
            base = base.unsqueeze(0).expand(content.shape[0], -1, -1, -1)
        if base.shape[:3] != content.shape[:3]:
            raise ValueError(
                f"Position/content shapes do not align: {tuple(base.shape)} vs "
                f"{tuple(content.shape)}"
            )
        if self.kind == "content_gate":
            gate_delta = self._linear(
                content,
                self.gate_weight,
                self.gate_bias,
            )
            return base * (1.0 + self.gate_init + gate_delta)

        joined = torch.cat((content, base), dim=-1)
        hidden = self._linear(joined, self.down, self.down_bias)
        hidden = torch.nn.functional.gelu(hidden)
        delta = self._linear(hidden, self.up, self.up_bias)
        return base + delta


class HeadCoupledFeaturePipeline(torch.nn.Module):
    """Frozen Fourier basis + grouped mapper + head layout."""

    def __init__(
        self,
        *,
        heads: int,
        head_dim: int,
        model_dim: int,
        extent: int,
        rope_theta: float,
        head_coupling: str,
        mapper_cfg: dict,
        input_cfg: dict,
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
        self.basis_dim = int(input_cfg["basis_dim"])
        self.groups = heads if head_coupling == "per_head_independent" else 1
        mapper_dim = model_dim if head_coupling == "per_head_joint" else head_dim

        theta = (
            rope_theta if input_cfg["theta"] is None else float(input_cfg["theta"])
        )
        self.basis = build_position_basis(
            kind=input_cfg["kind"],
            extent=extent,
            basis_dim=self.basis_dim,
            theta=theta,
            scalars=input_cfg["scalars"],
        )
        self.mapper = build_mapper(
            kind=mapper_cfg["kind"],
            groups=self.groups,
            input_dim=self.basis.output_dim,
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
        mapper_cfg = config["mapper"]
        readout_groups = _readout_groups(self.head_coupling, heads)

        def make_pipeline() -> HeadCoupledFeaturePipeline:
            return HeadCoupledFeaturePipeline(
                heads=heads,
                head_dim=head_dim,
                model_dim=model_dim,
                extent=extent,
                rope_theta=rope_theta,
                head_coupling=self.head_coupling,
                mapper_cfg=mapper_cfg,
                input_cfg=config["input"],
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
        self.amplitude_head = None
        self.q_amplitude_head = None
        self.k_amplitude_head = None
        self.projected_phase_head = None
        self.q_projected_phase_head = None
        self.k_projected_phase_head = None
        self.scale_head = None
        self.q_scale_head = None
        self.k_scale_head = None
        self.conditioner = None
        self.q_conditioner = None
        self.k_conditioner = None
        self.amplitude_conditioner = None
        self.q_amplitude_conditioner = None
        self.k_amplitude_conditioner = None
        self.output_config = config["output"]
        self.conditioning_config = config["conditioning"]

        from position.rotary import build_rope_cache

        base_sin, base_cos = build_rope_cache(extent, head_dim, rope_theta)
        self.register_buffer("base_sin", base_sin, persistent=False)
        self.register_buffer("base_cos", base_cos, persistent=False)

        if self.application == "additive" and self.geometry == "free":
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
        elif self.application == "additive" and self.geometry == "amplitude_phase":
            if self.qk_coupling == "shared":
                self.amplitude_head = GroupedLinearReadout(
                    readout_groups, head_dim, head_dim // 2, init="zeros"
                )
                self.phase_head = GroupedLinearReadout(
                    readout_groups, head_dim, head_dim // 2, init="zeros"
                )
            else:
                self.q_amplitude_head = GroupedLinearReadout(
                    readout_groups, head_dim, head_dim // 2, init="zeros"
                )
                self.q_phase_head = GroupedLinearReadout(
                    readout_groups, head_dim, head_dim // 2, init="zeros"
                )
                self.k_amplitude_head = copy.deepcopy(self.q_amplitude_head)
                self.k_phase_head = copy.deepcopy(self.q_phase_head)
        elif self.application == "rotary" and self.geometry in {
            "phase",
            "scaled_phase",
        }:
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
            if self.geometry == "scaled_phase":
                if self.qk_coupling == "shared":
                    self.scale_head = GroupedLinearReadout(
                        readout_groups, head_dim, head_dim // 2, init="zeros"
                    )
                else:
                    self.q_scale_head = GroupedLinearReadout(
                        readout_groups, head_dim, head_dim // 2, init="zeros"
                    )
                    self.k_scale_head = copy.deepcopy(self.q_scale_head)
        elif self.application == "rotary" and self.geometry == "projected_phase":
            if self.qk_coupling == "shared":
                self.projected_phase_head = GroupedLinearReadout(
                    readout_groups, head_dim, head_dim, init="zeros"
                )
            else:
                self.q_projected_phase_head = GroupedLinearReadout(
                    readout_groups, head_dim, head_dim, init="zeros"
                )
                self.k_projected_phase_head = copy.deepcopy(
                    self.q_projected_phase_head
                )
        else:
            raise ValueError(
                f"Unsupported Q/K application/geometry: "
                f"{self.application!r}/{self.geometry!r}"
            )

        conditioning_kind = self.conditioning_config["kind"]
        if conditioning_kind != "none":
            output_dim = (
                head_dim
                if self.application == "additive"
                and self.geometry == "free"
                else head_dim // 2
            )

            def make_conditioner(
                conditioned_dim: int = output_dim,
            ) -> GroupedContentConditioner:
                return GroupedContentConditioner(
                    kind=conditioning_kind,
                    groups=readout_groups,
                    content_dim=head_dim,
                    output_dim=conditioned_dim,
                    hidden_dim=self.conditioning_config["hidden_dim"],
                    gate_init=self.conditioning_config["gate_init"],
                )

            if self.qk_coupling == "shared":
                self.conditioner = make_conditioner()
            else:
                self.q_conditioner = make_conditioner()
                self.k_conditioner = copy.deepcopy(self.q_conditioner)
            if self.geometry == "amplitude_phase":
                if self.qk_coupling == "shared":
                    self.amplitude_conditioner = make_conditioner(head_dim // 2)
                else:
                    self.q_amplitude_conditioner = make_conditioner(head_dim // 2)
                    self.k_amplitude_conditioner = copy.deepcopy(
                        self.q_amplitude_conditioner
                    )

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

    def _project_to_phase(
        self,
        features: torch.Tensor,
        head: GroupedLinearReadout,
        sequence_length: int,
    ) -> torch.Tensor:
        projected = self._apply_phase_head(features, head)
        half = projected.shape[-1] // 2
        radial_x, radial_y = projected[..., :half], projected[..., half:]
        sin = self.base_sin[:sequence_length].to(projected.dtype)[None, :, :]
        cos = self.base_cos[:sequence_length].to(projected.dtype)[None, :, :]
        return -sin * radial_x + cos * radial_y

    def _amplitude(
        self,
        raw: torch.Tensor,
    ) -> torch.Tensor:
        amplitude_init = self.output_config["amplitude_init"]
        if self.output_config["amplitude_parameterization"] == "signed":
            return raw + amplitude_init
        if amplitude_init == 0:
            offset = raw.new_tensor(-20.0)
        else:
            offset = raw.new_tensor(amplitude_init).expm1().log()
        return torch.nn.functional.softplus(raw + offset)

    def _scale(self, raw: torch.Tensor) -> torch.Tensor:
        scale_init = self.output_config["scale_init"]
        if self.output_config["scale_parameterization"] == "exp":
            return raw.exp() * scale_init
        return raw + scale_init

    def _amplitude_phase_addend(
        self,
        features: torch.Tensor,
        amplitude_head: GroupedLinearReadout,
        phase_head: GroupedLinearReadout,
        sequence_length: int,
        *,
        content: torch.Tensor | None = None,
        amplitude_conditioner: GroupedContentConditioner | None = None,
        phase_conditioner: GroupedContentConditioner | None = None,
    ) -> torch.Tensor:
        amplitude = self._amplitude(
            self._apply_phase_head(features, amplitude_head)
        )
        phase = self._apply_phase_head(features, phase_head)
        if self.conditioning_config["kind"] != "none":
            if content is None:
                raise ValueError(
                    "Amplitude/phase content conditioning requires Q/K content."
                )
            amplitude = amplitude_conditioner(amplitude, content)
            phase = phase_conditioner(phase, content)
        phase = phase * self.output_config["phase_scale"]
        base_sin = self.base_sin[:sequence_length].to(phase.dtype)[None, :, :]
        base_cos = self.base_cos[:sequence_length].to(phase.dtype)[None, :, :]
        delta_sin, delta_cos = phase.sin(), phase.cos()
        sin = base_sin * delta_cos + base_cos * delta_sin
        cos = base_cos * delta_cos - base_sin * delta_sin
        return torch.cat((amplitude * cos, amplitude * sin), dim=-1)

    def _condition_outputs(
        self,
        q_output: torch.Tensor,
        k_output: torch.Tensor,
        q_content: torch.Tensor | None,
        k_content: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.conditioning_config["kind"] == "none":
            return q_output, k_output
        if self.geometry == "amplitude_phase":
            # Conditioning was applied to amplitude/phase latents before
            # synthesis so every pair retains the configured radial geometry.
            return q_output, k_output
        if q_content is None or k_content is None:
            raise ValueError(
                "Q/K content conditioning requires q_content and k_content."
            )
        if self.qk_coupling == "shared":
            return (
                self.conditioner(q_output, q_content),
                self.conditioner(k_output, k_content),
            )
        return (
            self.q_conditioner(q_output, q_content),
            self.k_conditioner(k_output, k_content),
        )

    def forward(
        self,
        sequence_length: int,
        *,
        dtype: torch.dtype | None = None,
        q_content: torch.Tensor | None = None,
        k_content: torch.Tensor | None = None,
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

        q_scale = None
        k_scale = None
        if self.application == "additive" and self.geometry == "free":
            if self.qk_coupling == "shared":
                q_output, k_output = q_features, k_features
            elif self.qk_coupling == "shared_trunk_separate_readouts":
                q_output = self._apply_add_readout(
                    q_features, self.q_add_readout
                )
                k_output = self._apply_add_readout(
                    k_features, self.k_add_readout
                )
            else:
                q_output, k_output = q_features, k_features
        elif self.application == "additive":
            if self.qk_coupling == "shared":
                q_output = self._amplitude_phase_addend(
                    q_features,
                    self.amplitude_head,
                    self.phase_head,
                    sequence_length,
                    content=q_content,
                    amplitude_conditioner=self.amplitude_conditioner,
                    phase_conditioner=self.conditioner,
                )
                k_output = self._amplitude_phase_addend(
                    k_features,
                    self.amplitude_head,
                    self.phase_head,
                    sequence_length,
                    content=k_content,
                    amplitude_conditioner=self.amplitude_conditioner,
                    phase_conditioner=self.conditioner,
                )
            else:
                q_output = self._amplitude_phase_addend(
                    q_features,
                    self.q_amplitude_head,
                    self.q_phase_head,
                    sequence_length,
                    content=q_content,
                    amplitude_conditioner=self.q_amplitude_conditioner,
                    phase_conditioner=self.q_conditioner,
                )
                k_output = self._amplitude_phase_addend(
                    k_features,
                    self.k_amplitude_head,
                    self.k_phase_head,
                    sequence_length,
                    content=k_content,
                    amplitude_conditioner=self.k_amplitude_conditioner,
                    phase_conditioner=self.k_conditioner,
                )
        elif self.geometry == "projected_phase":
            if self.qk_coupling == "shared":
                phase = self._project_to_phase(
                    q_features,
                    self.projected_phase_head,
                    sequence_length,
                )
                q_output, k_output = phase, phase
            else:
                q_output = self._project_to_phase(
                    q_features,
                    self.q_projected_phase_head,
                    sequence_length,
                )
                k_output = self._project_to_phase(
                    k_features,
                    self.k_projected_phase_head,
                    sequence_length,
                )
        else:
            if self.qk_coupling == "shared":
                phase = self._apply_phase_head(q_features, self.phase_head)
                q_output, k_output = phase, phase
            else:
                q_output = self._apply_phase_head(
                    q_features, self.q_phase_head
                )
                k_output = self._apply_phase_head(
                    k_features, self.k_phase_head
                )
            if self.geometry == "scaled_phase":
                if self.qk_coupling == "shared":
                    scale = self._scale(
                        self._apply_phase_head(q_features, self.scale_head)
                    )
                    q_scale, k_scale = scale, scale
                else:
                    q_scale = self._scale(
                        self._apply_phase_head(q_features, self.q_scale_head)
                    )
                    k_scale = self._scale(
                        self._apply_phase_head(k_features, self.k_scale_head)
                    )

        q_output = q_output * self.output_config["phase_scale"] if (
            self.application == "rotary"
        ) else q_output
        k_output = k_output * self.output_config["phase_scale"] if (
            self.application == "rotary"
        ) else k_output
        q_output, k_output = self._condition_outputs(
            q_output,
            k_output,
            q_content,
            k_content,
        )
        return QKPositionOutput(
            self.application,
            q_output,
            k_output,
            q_scale=q_scale,
            k_scale=k_scale,
        )

    def reset_output_parameters(self) -> None:
        for module in (
            self.phase_head,
            self.q_phase_head,
            self.k_phase_head,
            self.q_add_readout,
            self.k_add_readout,
            self.amplitude_head,
            self.q_amplitude_head,
            self.k_amplitude_head,
            self.projected_phase_head,
            self.q_projected_phase_head,
            self.k_projected_phase_head,
            self.scale_head,
            self.q_scale_head,
            self.k_scale_head,
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
        if self.conditioning_config["kind"] == "none":
            output = self.forward(sequence_length, dtype=dtype)
        else:
            parameter = next(self.parameters())
            zero_content = torch.zeros(
                1,
                self.heads,
                sequence_length,
                self.head_dim,
                device=parameter.device,
                dtype=dtype or parameter.dtype,
            )
            output = self.forward(
                sequence_length,
                dtype=dtype,
                q_content=zero_content,
                k_content=zero_content,
            )
        metrics: dict[str, float] = {}

        def _stats(prefix: str, tensor: torch.Tensor) -> None:
            values = tensor.detach().float()
            metrics[f"{prefix}/mean"] = values.mean().item()
            metrics[f"{prefix}/std"] = values.std().item()
            metrics[f"{prefix}/rms"] = values.pow(2).mean().sqrt().item()
            metrics[f"{prefix}/abs_max"] = values.abs().max().item()

        basis_modules = (
            (self.q_pipeline.basis, self.k_pipeline.basis)
            if self.qk_coupling == "separate"
            else (self.pipeline.basis, self.pipeline.basis)
        )
        q_frequency = basis_modules[0].frequencies().detach().float()
        k_frequency = basis_modules[1].frequencies().detach().float()
        metrics["frequency_mean"] = q_frequency.mean().item()
        metrics["frequency_min"] = q_frequency.min().item()
        metrics["frequency_max"] = q_frequency.max().item()
        metrics["qk_frequency_diff_rms"] = (
            (q_frequency - k_frequency).pow(2).mean().sqrt().item()
        )

        _stats("q", output.q)
        _stats("k", output.k)
        diff = (output.q - output.k).detach().float()
        metrics["qk_diff_rms"] = diff.pow(2).mean().sqrt().item()
        if output.application == "rotary":
            _stats("phase_q", output.q)
            _stats("phase_k", output.k)
            if output.q_scale is not None:
                _stats("scale_q", output.q_scale)
                _stats("scale_k", output.k_scale)
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


class InklingRouter(torch.nn.Module):
    """Grouped query router over a bank of relative-distance profiles."""

    def __init__(
        self,
        *,
        groups: int,
        head_dim: int,
        hidden_dim: int,
        num_profiles: int,
    ):
        super().__init__()
        self.groups = groups
        self.down = torch.nn.Parameter(
            torch.empty(groups, head_dim, hidden_dim)
        )
        self.down_bias = torch.nn.Parameter(torch.zeros(groups, hidden_dim))
        self.up = torch.nn.Parameter(
            torch.empty(groups, hidden_dim, num_profiles)
        )
        self.up_bias = torch.nn.Parameter(torch.zeros(groups, num_profiles))
        for weight in self.down:
            torch.nn.init.xavier_normal_(weight)
        for weight in self.up:
            torch.nn.init.xavier_normal_(weight)

    def forward(self, query: torch.Tensor) -> torch.Tensor:
        if self.groups == 1:
            hidden = torch.einsum("bhld,dk->bhlk", query, self.down[0])
            hidden = hidden + self.down_bias[0]
            hidden = torch.nn.functional.gelu(hidden)
            logits = torch.einsum("bhlk,kp->bhlp", hidden, self.up[0])
            return logits + self.up_bias[0]
        hidden = torch.einsum("bhld,hdk->bhlk", query, self.down)
        hidden = hidden + self.down_bias[None, :, None, :]
        hidden = torch.nn.functional.gelu(hidden)
        logits = torch.einsum("bhlk,hkp->bhlp", hidden, self.up)
        return logits + self.up_bias[None, :, None, :]


class InklingProfileBank(torch.nn.Module):
    """Table or CosNet bank mixed per query by ``InklingRouter``."""

    def __init__(
        self,
        *,
        kind: Literal["inkling_table", "inkling_cosnet"],
        groups: int,
        heads: int,
        head_dim: int,
        extent: int,
        config: dict,
    ):
        super().__init__()
        self.kind = kind
        self.groups = groups
        self.heads = heads
        self.extent = extent
        self.num_profiles = config["num_profiles"]
        self.router = InklingRouter(
            groups=groups,
            head_dim=head_dim,
            hidden_dim=config["router_hidden_dim"],
            num_profiles=self.num_profiles,
        )
        self.gate = torch.nn.Parameter(
            torch.full((groups,), float(config["gate_init"]))
        )
        profile_std = config["profile_init_std"]
        if kind == "inkling_table":
            self.profile_table = torch.nn.Parameter(
                torch.empty(groups, self.num_profiles, extent)
            )
            torch.nn.init.normal_(self.profile_table, std=profile_std)
        else:
            num_frequencies = config["num_frequencies"]
            self.amplitude = torch.nn.Parameter(
                torch.empty(groups, self.num_profiles, num_frequencies)
            )
            torch.nn.init.normal_(self.amplitude, std=profile_std)
            initial = torch.linspace(
                1.0 / max(extent, 1),
                1.0,
                num_frequencies,
            ).log()
            self.log_frequency = torch.nn.Parameter(
                initial[None, None, :]
                .expand(groups, self.num_profiles, -1)
                .clone()
            )
            self.phase = torch.nn.Parameter(
                torch.zeros(groups, self.num_profiles, num_frequencies)
            )

    def profiles(
        self,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        if self.kind == "inkling_table":
            return self.profile_table.tanh().to(dtype=dtype)
        distance = torch.arange(
            self.extent,
            device=device,
            dtype=torch.float32,
        )
        frequency = self.log_frequency.float().exp()
        angle = (
            distance[None, None, :, None] * frequency[:, :, None, :]
            + self.phase.float()[:, :, None, :]
        )
        profiles = (
            self.amplitude.float()[:, :, None, :] * angle.cos()
        ).sum(dim=-1) / math.sqrt(self.amplitude.shape[-1])
        return profiles.tanh().to(dtype=dtype)

    def forward(
        self,
        query: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        routing = self.router(query).softmax(dim=-1)
        profiles = self.profiles(dtype=query.dtype, device=query.device)
        if self.groups == 1:
            mixture = torch.einsum(
                "bhlp,pr->bhlr",
                routing,
                profiles[0],
            )
            gate = self.gate[0]
        else:
            mixture = torch.einsum(
                "bhlp,hpr->bhlr",
                routing,
                profiles,
            )
            gate = self.gate[None, :, None, None]
        return mixture * gate, routing


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
        self.pipeline = HeadCoupledFeaturePipeline(
            heads=heads,
            head_dim=head_dim,
            model_dim=model_dim,
            extent=extent,
            rope_theta=rope_theta,
            head_coupling=self.head_coupling,
            mapper_cfg=config["mapper"],
            input_cfg=config["input"],
        )
        self.scalar_head = ScalarCurveReadout(
            _readout_groups(self.head_coupling, heads),
            head_dim,
        )
        conditioning = config["conditioning"]
        self.conditioning_kind = conditioning["kind"]
        self.inkling = None
        self._last_routing_summary: dict[str, torch.Tensor] = {}
        if self.conditioning_kind != "none":
            self.inkling = InklingProfileBank(
                kind=self.conditioning_kind,
                groups=_readout_groups(self.head_coupling, heads),
                heads=heads,
                head_dim=head_dim,
                extent=extent,
                config=conditioning,
            )

    def forward(
        self,
        *,
        dtype: torch.dtype | None = None,
        query: torch.Tensor | None = None,
    ) -> torch.Tensor:
        base = self.base_curves(dtype=dtype)
        if self.inkling is None:
            return base
        if query is None:
            raise ValueError(
                f"{self.conditioning_kind} logit bias requires normalized query content."
            )
        conditional, routing = self.inkling(query)
        with torch.no_grad():
            routing_f = routing.detach().float()
            entropy = -(routing_f.clamp_min(1e-9).log() * routing_f).sum(-1)
            self._last_routing_summary = {
                "routing_entropy_mean": entropy.mean(),
                "routing_max_probability": routing_f.max(dim=-1).values.mean(),
                "inkling_gate_abs_mean": self.inkling.gate.detach().float().abs().mean(),
            }
        return base[None, :, None, :] + conditional

    def base_curves(
        self,
        *,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        features = self.pipeline(dtype=dtype)
        if self.head_coupling == "shared_head":
            return self.scalar_head(features[:1], self.heads)
        return self.scalar_head(features, self.heads)

    def reset_output_parameters(self) -> None:
        self.scalar_head.reset_parameters()

    def routing_summary(self) -> dict[str, float]:
        return {
            key: value.item()
            for key, value in self._last_routing_summary.items()
        }


class ResidualPositionChannel(PositionChannel):
    """Absolute-position writes into the model residual stream."""

    def __init__(
        self,
        config: dict,
        *,
        model_dim: int,
        heads: int,
        extent: int,
        rope_theta: float,
    ):
        super().__init__()
        self.config = copy.deepcopy(config)
        self.model_dim = model_dim
        self.extent = extent
        self.source = config["source"]
        self.gate = torch.nn.Parameter(
            torch.tensor(float(config["gate_init"]))
        )
        if self.source == "learned_absolute":
            self.embedding = torch.nn.Parameter(torch.empty(extent, model_dim))
            torch.nn.init.normal_(self.embedding, std=model_dim ** -0.5)
            self.pipeline = None
        else:
            self.register_parameter("embedding", None)
            self.pipeline = HeadCoupledFeaturePipeline(
                heads=heads,
                head_dim=model_dim // heads,
                model_dim=model_dim,
                extent=extent,
                rope_theta=rope_theta,
                head_coupling="per_head_joint",
                mapper_cfg=config["mapper"],
                input_cfg=config["input"],
            )

    def forward(
        self,
        length: int,
        *,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if length > self.extent:
            raise ValueError(
                f"Residual position length {length} exceeds extent {self.extent}."
            )
        if self.source == "learned_absolute":
            position = self.embedding[:length].to(dtype=dtype)
        else:
            by_head = self.pipeline(length, dtype=dtype)
            position = by_head.permute(1, 0, 2).reshape(length, self.model_dim)
        return position * self.gate.to(dtype=dtype)

    def reset_output_parameters(self) -> None:
        # The configured gate is the explicit initialization contract.
        with torch.no_grad():
            self.gate.fill_(float(self.config["gate_init"]))


def _map_batched_features(
    mapper: FeatureMapper,
    features: torch.Tensor,
    *,
    head_coupling: str,
) -> torch.Tensor:
    """Apply a grouped mapper to ``[B,H,L,I]`` without Python loops."""
    if head_coupling == "per_head_joint":
        batch, heads, length, width = features.shape
        joined = features.permute(0, 2, 1, 3).reshape(
            batch, length, heads * width
        )
        grouped = joined[:, None, :, :]
    else:
        grouped = features

    if mapper.kind == "identity":
        mapped = grouped
    elif mapper.kind == "euclidean_affine":
        scale = mapper.scale
        offset = mapper.offset
        if mapper.groups == 1:
            scale = scale[None, :, None, :]
            offset = offset[None, :, None, :]
        else:
            scale = scale[None, :, None, :]
            offset = offset[None, :, None, :]
        mapped = grouped * (1.0 + scale) + offset
    elif mapper.kind == "linear":
        if head_coupling == "shared_head":
            mapped = torch.einsum(
                "bhli,io->bhlo",
                grouped,
                mapper.weight[0],
            )
        elif mapper.groups == 1:
            mapped = torch.einsum(
                "bgli,gio->bglo",
                grouped,
                mapper.weight,
            )
        else:
            mapped = torch.einsum(
                "bhli,hio->bhlo",
                grouped,
                mapper.weight,
            )
        bias = (
            mapper.bias[0][None, None, None, :]
            if head_coupling == "shared_head"
            else mapper.bias[None, :, None, :]
        )
        mapped = mapped + bias
    else:
        if head_coupling == "shared_head":
            hidden = torch.einsum(
                "bhli,ik->bhlk",
                grouped,
                mapper.down[0],
            )
        elif mapper.groups == 1:
            hidden = torch.einsum(
                "bgli,gik->bglk",
                grouped,
                mapper.down,
            )
        else:
            hidden = torch.einsum(
                "bhli,hik->bhlk",
                grouped,
                mapper.down,
            )
        down_bias = (
            mapper.down_bias[0][None, None, None, :]
            if head_coupling == "shared_head"
            else mapper.down_bias[None, :, None, :]
        )
        hidden = hidden + down_bias
        if mapper.kind in {"bottleneck_mlp", "mlp"}:
            hidden = torch.nn.functional.gelu(hidden)
        if head_coupling == "shared_head":
            branch = torch.einsum(
                "bhlk,ko->bhlo",
                hidden,
                mapper.up[0],
            )
        elif mapper.groups == 1:
            branch = torch.einsum(
                "bglk,gko->bglo",
                hidden,
                mapper.up,
            )
        else:
            branch = torch.einsum(
                "bhlk,hko->bhlo",
                hidden,
                mapper.up,
            )
        up_bias = (
            mapper.up_bias[0][None, None, None, :]
            if head_coupling == "shared_head"
            else mapper.up_bias[None, :, None, :]
        )
        branch = branch + up_bias
        mapped = grouped + branch if mapper.residual else branch

    if head_coupling == "per_head_joint":
        batch, _, length, _ = mapped.shape
        output_dim = mapper.output_dim
        heads = features.shape[1]
        return (
            mapped[:, 0]
            .reshape(batch, length, heads, output_dim // heads)
            .permute(0, 2, 1, 3)
            .contiguous()
        )
    if head_coupling == "shared_head" and mapped.shape[1] == 1:
        return mapped.expand(-1, features.shape[1], -1, -1)
    return mapped


class AttentionPositionWriteChannel(PositionChannel):
    """Write attended key-position or relative-offset summaries to residual."""

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
        self.mode = config["mode"]
        self.head_coupling = config["head_coupling"]
        self.pipeline = HeadCoupledFeaturePipeline(
            heads=heads,
            head_dim=head_dim,
            model_dim=model_dim,
            extent=extent,
            rope_theta=rope_theta,
            head_coupling=self.head_coupling,
            mapper_cfg=config["mapper"],
            input_cfg=config["input"],
        )
        input_width = self.pipeline.basis.output_dim
        if self.mode == "key_position":
            # g(position_j) is mapped before attention, then summed by A_ij.
            self.value_dim = head_dim
        elif self.head_coupling == "per_head_joint":
            if input_width % heads != 0:
                raise ValueError(
                    "per_head_joint attention_write input width must divide heads"
                )
            self.value_dim = input_width // heads
        else:
            self.value_dim = input_width
        if self.mode == "relative_offset" and self.value_dim % 2 != 0:
            raise ValueError(
                "relative_offset attention writes require an even per-head value width"
            )
        gate_groups = _readout_groups(self.head_coupling, heads)
        self.gate = torch.nn.Parameter(
            torch.full((gate_groups,), float(config["gate_init"]))
        )

    def position_values(
        self,
        length: int,
        *,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if self.mode == "key_position":
            return self.pipeline(length, dtype=dtype)
        raw = self.pipeline.basis(length, dtype=dtype)
        if self.head_coupling == "per_head_joint":
            return (
                raw.reshape(length, self.heads, self.value_dim)
                .permute(1, 0, 2)
                .contiguous()
            )
        return raw[None, :, :].expand(self.heads, -1, -1)

    def _relative(
        self,
        summary: torch.Tensor,
        *,
        query_length: int,
    ) -> torch.Tensor:
        query = self.position_values(
            query_length,
            dtype=summary.dtype,
        )[None, :, :, :]
        q_cos, q_sin = query[..., 0::2], query[..., 1::2]
        k_cos, k_sin = summary[..., 0::2], summary[..., 1::2]
        relative_cos = q_cos * k_cos + q_sin * k_sin
        relative_sin = q_sin * k_cos - q_cos * k_sin
        return torch.stack(
            (relative_cos, relative_sin),
            dim=-1,
        ).flatten(-2)

    def forward(self, summary: torch.Tensor) -> torch.Tensor:
        if self.mode == "relative_offset":
            summary = self._relative(
                summary,
                query_length=summary.shape[-2],
            )
            mapped = _map_batched_features(
                self.pipeline.mapper,
                summary,
                head_coupling=self.head_coupling,
            )
        else:
            mapped = summary
        if self.gate.numel() == 1:
            mapped = mapped * self.gate[0]
        else:
            mapped = mapped * self.gate[None, :, None, None]
        return mapped.transpose(1, 2).reshape(
            mapped.shape[0],
            mapped.shape[2],
            self.model_dim,
        )

    def reset_output_parameters(self) -> None:
        with torch.no_grad():
            self.gate.fill_(float(self.config["gate_init"]))


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


def build_residual_position_channel(
    config: dict | None,
    *,
    model_dim: int,
    heads: int,
    extent: int,
    rope_theta: float,
) -> ResidualPositionChannel | None:
    resolved = normalize_residual_stream_config(
        config,
        model_dim=model_dim,
        heads=heads,
        rope_theta=rope_theta,
    )
    if not resolved["enabled"]:
        return None
    return ResidualPositionChannel(
        resolved,
        model_dim=model_dim,
        heads=heads,
        extent=extent,
        rope_theta=rope_theta,
    )


def build_attention_position_write_channel(
    config: dict | None,
    *,
    heads: int,
    head_dim: int,
    model_dim: int,
    extent: int,
    rope_theta: float,
) -> AttentionPositionWriteChannel | None:
    resolved = normalize_attention_write_config(
        config,
        model_dim=model_dim,
        heads=heads,
        rope_theta=rope_theta,
    )
    if not resolved["enabled"]:
        return None
    return AttentionPositionWriteChannel(
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
    attention_write_total = 0
    residual_stream_total = _unique_numel(
        getattr(model, "residual_position_input", None)
    )
    residual_layers = getattr(model, "residual_position_layers", None)
    if residual_layers is not None:
        residual_stream_total += _unique_numel(residual_layers)
    blocks = getattr(model, "blocks", None)
    if blocks is not None:
        for block in blocks:
            attn = getattr(block, "attn", None)
            if attn is None:
                continue
            qk_total += _unique_numel(getattr(attn, "qk_position", None))
            logit_total += _unique_numel(getattr(attn, "logit_bias", None))
            attention_write_total += _unique_numel(
                getattr(attn, "position_write", None)
            )
    position_total = (
        qk_total
        + logit_total
        + attention_write_total
        + residual_stream_total
    )
    return {
        "qk_position_params": qk_total,
        "logit_bias_params": logit_total,
        "attention_write_params": attention_write_total,
        "residual_stream_params": residual_stream_total,
        "position_params": position_total,
    }
