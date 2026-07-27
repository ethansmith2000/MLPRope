"""Composable position channels: head pipelines, Q/K coupling, and builders."""

from __future__ import annotations

import copy
import math
import warnings
from dataclasses import dataclass
from typing import Literal

import torch

from position.autograd import exp_with_identity_grad
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
    q_gain: torch.Tensor | None = None
    k_gain: torch.Tensor | None = None
    q_log_gain_delta: torch.Tensor | None = None
    k_log_gain_delta: torch.Tensor | None = None
    q_hyper_phase_delta: torch.Tensor | None = None
    k_hyper_phase_delta: torch.Tensor | None = None


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
        activation: str = "tanh",
    ):
        super().__init__()
        self.kind = kind
        self.groups = groups
        self.content_dim = content_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.gate_init = float(gate_init)
        self.activation = activation
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
            # Keep the content multiplier in [0, 2]. The original unconstrained
            # affine gate could grow positional addends into the thousands.
            if self.activation == "scaled_sigmoid":
                offset = math.log(self.gate_init / (2.0 - self.gate_init))
                return base * (2.0 * torch.sigmoid(gate_delta + offset))
            gate = torch.tanh(gate_delta + self.gate_init)
            return base * (1.0 + gate)

        joined = torch.cat((content, base), dim=-1)
        hidden = self._linear(joined, self.down, self.down_bias)
        hidden = torch.nn.functional.gelu(hidden)
        delta = self._linear(hidden, self.up, self.up_bias)
        if self.activation == "tanh":
            delta = torch.tanh(delta + self.gate_init)
        elif self.activation == "gelu":
            delta = torch.nn.functional.gelu(delta + self.gate_init)
        elif self.activation == "linear":
            delta = delta + self.gate_init
        return base + delta


class GroupedPhaseRotationConditioner(torch.nn.Module):
    """Content-driven bounded rotations of fixed-radius additive pairs."""

    def __init__(
        self,
        *,
        groups: int,
        content_dim: int,
        pair_dim: int,
        hidden_dim: int,
        target: str,
        coupling: str,
        phase_bound: float,
    ):
        super().__init__()
        self.groups = groups
        self.content_dim = content_dim
        self.pair_dim = pair_dim
        self.hidden_dim = hidden_dim
        self.target = target
        self.coupling = coupling
        self.phase_bound = float(phase_bound)
        self.down = torch.nn.Parameter(
            torch.empty(groups, content_dim, hidden_dim)
        )
        self.down_bias = torch.nn.Parameter(torch.zeros(groups, hidden_dim))
        for weight in self.down:
            torch.nn.init.xavier_normal_(weight)

        self.up = None
        self.up_bias = None
        self.q_up = None
        self.q_up_bias = None
        self.k_up = None
        self.k_up_bias = None
        if coupling == "shared":
            self.up = torch.nn.Parameter(
                torch.zeros(groups, hidden_dim, pair_dim)
            )
            self.up_bias = torch.nn.Parameter(torch.zeros(groups, pair_dim))
        else:
            if target in {"q", "both"}:
                self.q_up = torch.nn.Parameter(
                    torch.zeros(groups, hidden_dim, pair_dim)
                )
                self.q_up_bias = torch.nn.Parameter(
                    torch.zeros(groups, pair_dim)
                )
            if target in {"k", "both"}:
                self.k_up = torch.nn.Parameter(
                    torch.zeros(groups, hidden_dim, pair_dim)
                )
                self.k_up_bias = torch.nn.Parameter(
                    torch.zeros(groups, pair_dim)
                )

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

    def phase(self, content: torch.Tensor, branch: str) -> torch.Tensor:
        if branch not in {"q", "k"}:
            raise ValueError(f"Unknown phase-conditioning branch: {branch!r}")
        if content.ndim != 4:
            raise ValueError(
                "Phase conditioning expects [batch, heads, sequence, content_dim]"
            )
        hidden = torch.nn.functional.gelu(
            self._linear(content, self.down, self.down_bias)
        )
        if self.coupling == "shared":
            up, bias = self.up, self.up_bias
        elif branch == "q":
            up, bias = self.q_up, self.q_up_bias
        else:
            up, bias = self.k_up, self.k_up_bias
        if up is None or bias is None:
            return hidden.new_zeros(*hidden.shape[:-1], self.pair_dim)
        raw = self._linear(hidden, up, bias)
        # ``phase_bound`` is a linear scale, not a saturating bound. Any real
        # phase remains a valid unit-circle rotation after sin/cos synthesis.
        return self.phase_bound * raw

    def reset_output_parameters(self) -> None:
        for parameter in (
            self.up,
            self.up_bias,
            self.q_up,
            self.q_up_bias,
            self.k_up,
            self.k_up_bias,
        ):
            if parameter is not None:
                torch.nn.init.zeros_(parameter)


class GroupedContentActuator(torch.nn.Module):
    """Zero-output content head for anchor-relative phase or gain deltas."""

    def __init__(
        self,
        *,
        groups: int,
        content_dim: int,
        output_dim: int,
        hidden_dim: int,
        target: str,
        coupling: str,
    ):
        super().__init__()
        self.groups = groups
        self.output_dim = output_dim
        self.target = target
        self.coupling = coupling
        self.down = torch.nn.Parameter(
            torch.empty(groups, content_dim, hidden_dim)
        )
        self.down_bias = torch.nn.Parameter(torch.zeros(groups, hidden_dim))
        for weight in self.down:
            torch.nn.init.xavier_normal_(weight)

        self.up = self.up_bias = None
        self.q_up = self.q_up_bias = None
        self.k_up = self.k_up_bias = None
        if coupling == "shared":
            self.up = torch.nn.Parameter(
                torch.zeros(groups, hidden_dim, output_dim)
            )
            self.up_bias = torch.nn.Parameter(torch.zeros(groups, output_dim))
        else:
            if target in {"q", "both"}:
                self.q_up = torch.nn.Parameter(
                    torch.zeros(groups, hidden_dim, output_dim)
                )
                self.q_up_bias = torch.nn.Parameter(
                    torch.zeros(groups, output_dim)
                )
            if target in {"k", "both"}:
                self.k_up = torch.nn.Parameter(
                    torch.zeros(groups, hidden_dim, output_dim)
                )
                self.k_up_bias = torch.nn.Parameter(
                    torch.zeros(groups, output_dim)
                )

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

    def forward(self, content: torch.Tensor, branch: str) -> torch.Tensor:
        if branch not in {"q", "k"}:
            raise ValueError(f"Unknown content-actuator branch: {branch!r}")
        hidden = torch.nn.functional.gelu(
            self._linear(content, self.down, self.down_bias)
        )
        if branch not in ({self.target} if self.target != "both" else {"q", "k"}):
            return hidden.new_zeros(*hidden.shape[:-1], self.output_dim)
        if self.coupling == "shared":
            up, bias = self.up, self.up_bias
        elif branch == "q":
            up, bias = self.q_up, self.q_up_bias
        else:
            up, bias = self.k_up, self.k_up_bias
        return self._linear(hidden, up, bias)

    def reset_output_parameters(self) -> None:
        for parameter in (
            self.up,
            self.up_bias,
            self.q_up,
            self.q_up_bias,
            self.k_up,
            self.k_up_bias,
        ):
            if parameter is not None:
                torch.nn.init.zeros_(parameter)


class _GroupedHyperTrunk(torch.nn.Module):
    """Grouped linear or nonlinear token-local hypernetwork trunk."""

    def __init__(
        self,
        *,
        groups: int,
        input_dim: int,
        hidden_dim: int,
        network: str,
    ):
        super().__init__()
        self.groups = groups
        self.input_dim = input_dim
        self.network = network
        if network == "linear":
            self.output_dim = input_dim
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
            return
        projected_dim = hidden_dim if network == "silu_mlp" else 2 * hidden_dim
        self.output_dim = hidden_dim
        self.weight = torch.nn.Parameter(
            torch.empty(groups, input_dim, projected_dim)
        )
        self.bias = torch.nn.Parameter(torch.zeros(groups, projected_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.weight is not None:
            for weight in self.weight:
                torch.nn.init.xavier_normal_(weight)
            torch.nn.init.zeros_(self.bias)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if self.network == "linear":
            return values
        projected = torch.einsum("bgld,gdo->bglo", values, self.weight)
        projected = projected + self.bias[None, :, None, :]
        if self.network == "silu_mlp":
            return torch.nn.functional.silu(projected)
        gate, value = projected.chunk(2, dim=-1)
        return torch.nn.functional.silu(gate) * value


class CarrierHypernetwork(torch.nn.Module):
    """Anchor-relative token-local gain/phase deltas for additive or rotary Q/K."""

    def __init__(
        self,
        *,
        heads: int,
        pair_dim: int,
        content_dim: int,
        position_dim: int,
        config: dict,
    ):
        super().__init__()
        self.heads = heads
        self.pair_dim = pair_dim
        self.content_dim = content_dim
        self.position_dim = position_dim
        self.input_mode = config["input_mode"]
        self.network = config["network"]
        self.components = config["components"]
        self.target = config["target"]
        self.coupling = config["coupling"]
        self.head_coupling = config["head_coupling"]
        self.groups = (
            1 if self.head_coupling == "shared_head" else self.heads
        )
        input_dim = 0
        if self.input_mode in {"content", "content_position"}:
            input_dim += content_dim
        if self.input_mode in {"position", "content_position"}:
            input_dim += position_dim
        output_dim = pair_dim * (
            2 if self.components == "log_gain_phase" else 1
        )

        def make_trunk() -> _GroupedHyperTrunk:
            return _GroupedHyperTrunk(
                groups=self.groups,
                input_dim=input_dim,
                hidden_dim=config["hidden_dim"],
                network=self.network,
            )

        def make_readout(trunk: _GroupedHyperTrunk) -> GroupedLinearReadout:
            return GroupedLinearReadout(
                self.groups,
                trunk.output_dim,
                output_dim,
                init="zeros",
            )

        self.trunk = self.q_trunk = self.k_trunk = None
        self.readout = self.q_readout = self.k_readout = None
        if self.coupling == "shared":
            self.trunk = make_trunk()
            self.readout = make_readout(self.trunk)
        elif self.coupling == "shared_trunk_separate_readouts":
            self.trunk = make_trunk()
            if self.target in {"q", "both"}:
                self.q_readout = make_readout(self.trunk)
            if self.target in {"k", "both"}:
                self.k_readout = make_readout(self.trunk)
        else:
            if self.target in {"q", "both"}:
                self.q_trunk = make_trunk()
                self.q_readout = make_readout(self.q_trunk)
            if self.target in {"k", "both"}:
                self.k_trunk = make_trunk()
                self.k_readout = make_readout(self.k_trunk)

    def _inputs(
        self,
        content: torch.Tensor | None,
        position: torch.Tensor,
    ) -> torch.Tensor:
        if position.ndim != 2:
            raise ValueError("Hypernetwork position input must be [sequence, dim]")
        needs_content = self.input_mode in {"content", "content_position"}
        if needs_content and content is None:
            raise ValueError(
                "Content-input carrier hypernetwork requires dedicated content"
            )
        batch = 1 if content is None else content.shape[0]
        length = position.shape[0]
        pieces = []
        if needs_content:
            if content.ndim != 4 or content.shape[1] != self.heads:
                raise ValueError(
                    "Hypernetwork content must be [batch, heads, sequence, dim]"
                )
            if content.shape[2] != length:
                raise ValueError("Hypernetwork content/position lengths differ")
            pieces.append(
                content[:, :1] if self.groups == 1 else content
            )
        if self.input_mode in {"position", "content_position"}:
            position_values = position.to(
                dtype=(
                    content.dtype
                    if content is not None
                    else position.dtype
                )
            )
            pieces.append(
                position_values[None, None].expand(
                    batch, self.groups, -1, -1
                )
            )
        return pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=-1)

    @staticmethod
    def _read(
        hidden: torch.Tensor,
        readout: GroupedLinearReadout,
    ) -> torch.Tensor:
        output = torch.einsum("bgld,gdo->bglo", hidden, readout.weight)
        return output + readout.bias[None, :, None, :]

    def _branch(
        self,
        values: torch.Tensor,
        branch: str,
    ) -> torch.Tensor:
        if branch not in ({self.target} if self.target != "both" else {"q", "k"}):
            output_dim = self.pair_dim * (
                2 if self.components == "log_gain_phase" else 1
            )
            return values.new_zeros(*values.shape[:-1], output_dim)
        if self.coupling == "shared":
            hidden = self.trunk(values)
            return self._read(hidden, self.readout)
        if self.coupling == "shared_trunk_separate_readouts":
            hidden = self.trunk(values)
            readout = self.q_readout if branch == "q" else self.k_readout
            return self._read(hidden, readout)
        trunk = self.q_trunk if branch == "q" else self.k_trunk
        readout = self.q_readout if branch == "q" else self.k_readout
        return self._read(trunk(values), readout)

    def _expand_heads(self, values: torch.Tensor) -> torch.Tensor:
        if self.groups == 1:
            return values.expand(-1, self.heads, -1, -1)
        return values

    def forward(
        self,
        *,
        q_content: torch.Tensor | None,
        k_content: torch.Tensor | None,
        position: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        q_values = self._inputs(q_content, position)
        k_values = self._inputs(k_content, position)
        if self.coupling == "shared" and self.target == "both":
            q_raw = self._branch(q_values, "q")
            k_raw = q_raw
        else:
            q_raw = self._branch(q_values, "q")
            k_raw = self._branch(k_values, "k")
        q_raw = self._expand_heads(q_raw)
        k_raw = self._expand_heads(k_raw)
        if self.components == "log_gain_phase":
            q_log_gain, q_phase = q_raw.split(self.pair_dim, dim=-1)
            k_log_gain, k_phase = k_raw.split(self.pair_dim, dim=-1)
        else:
            q_phase, k_phase = q_raw, k_raw
            q_log_gain = torch.zeros_like(q_phase)
            k_log_gain = torch.zeros_like(k_phase)
        return q_log_gain, q_phase, k_log_gain, k_phase

    def reset_output_parameters(self) -> None:
        for readout in (
            self.readout,
            self.q_readout,
            self.k_readout,
        ):
            if readout is not None:
                readout.reset_parameters()


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
        self.rope_theta = rope_theta
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
            normalization_extent=input_cfg.get("normalization_extent"),
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
        content_dim: int,
        extent: int,
        rope_theta: float,
    ):
        super().__init__()
        self.config = copy.deepcopy(config)
        self.heads = heads
        self.head_dim = head_dim
        self.model_dim = model_dim
        self.content_dim = content_dim
        self.extent = extent
        self.rope_theta = rope_theta
        self.application = config["application"]
        self.geometry = config["geometry"]
        self.qk_coupling = config["qk_coupling"]
        self.head_coupling = config["head_coupling"]
        self.output_config = config["output"]
        self.learn_amplitude = self.output_config["learn_amplitude"]
        self.learn_phase = self.output_config["learn_phase"]
        self.conditioning_config = config["conditioning"]
        self.fixed_amplitude_phase = (
            self.application == "additive"
            and self.geometry == "amplitude_phase"
            and not self.learn_amplitude
            and not self.learn_phase
        )
        self.fixed_rotary_phase = (
            self.application == "rotary"
            and self.geometry == "phase"
            and not self.learn_phase
        )
        self.fixed_position_pipeline = (
            self.fixed_amplitude_phase or self.fixed_rotary_phase
        )
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

        if self.fixed_position_pipeline:
            self.pipeline = None
            self.q_pipeline = None
            self.k_pipeline = None
        elif self.qk_coupling == "separate":
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
        self.phase_rotation_conditioner = None
        self.content_actuator = None
        self.amplitude_conditioner = None
        self.q_amplitude_conditioner = None
        self.k_amplitude_conditioner = None
        self.carrier_hypernetwork = None
        self.hyper_position_basis = None
        self.additive_gain = None
        self.q_additive_gain = None
        self.k_additive_gain = None

        if (
            self.application == "additive"
            and self.output_config["additive_normalization"] == "rms"
        ):
            gain_init = self.output_config["additive_gain_init"]
            gain_max = self.output_config["additive_gain_max"]
            gain_logit = math.log(gain_init / (gain_max - gain_init))

            def make_gain() -> torch.Tensor:
                value = torch.full((readout_groups, 1, 1), gain_logit)
                if self.output_config["learn_additive_gain"]:
                    return torch.nn.Parameter(value)
                self.register_buffer(
                    f"_fixed_additive_gain_{len(self._buffers)}",
                    value,
                    persistent=True,
                )
                return value

            if self.qk_coupling == "shared":
                self.additive_gain = make_gain()
            else:
                self.q_additive_gain = make_gain()
                self.k_additive_gain = make_gain()

        from position.rotary import build_rope_cache

        base_sin, base_cos = build_rope_cache(extent, head_dim, rope_theta)
        self.register_buffer("base_sin", base_sin, persistent=False)
        self.register_buffer("base_cos", base_cos, persistent=False)

        if self.application == "additive" and self.geometry in {
            "free",
            "pair_normalized",
        }:
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
                if self.learn_amplitude:
                    self.amplitude_head = GroupedLinearReadout(
                        readout_groups, head_dim, head_dim // 2, init="zeros"
                    )
                if self.learn_phase:
                    self.phase_head = GroupedLinearReadout(
                        readout_groups, head_dim, head_dim // 2, init="zeros"
                    )
            else:
                if self.learn_amplitude:
                    self.q_amplitude_head = GroupedLinearReadout(
                        readout_groups, head_dim, head_dim // 2, init="zeros"
                    )
                    self.k_amplitude_head = copy.deepcopy(
                        self.q_amplitude_head
                    )
                if self.learn_phase:
                    self.q_phase_head = GroupedLinearReadout(
                        readout_groups, head_dim, head_dim // 2, init="zeros"
                    )
                    self.k_phase_head = copy.deepcopy(self.q_phase_head)
        elif self.application == "rotary" and self.geometry in {
            "phase",
            "scaled_phase",
        }:
            if not self.learn_phase:
                pass
            elif self.qk_coupling == "shared":
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
        elif self.application == "rotary" and self.geometry in {
            "projected_phase",
            "unit_pair",
        }:
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
        if conditioning_kind == "phase_rotation":
            self.phase_rotation_conditioner = GroupedPhaseRotationConditioner(
                groups=readout_groups,
                content_dim=content_dim,
                pair_dim=head_dim // 2,
                hidden_dim=self.conditioning_config["hidden_dim"],
                target=self.conditioning_config["target"],
                coupling=self.conditioning_config["coupling"],
                phase_bound=self.conditioning_config["phase_bound"],
            )
        elif conditioning_kind in {
            "adaptive_gain",
            "additive_phase",
            "rope_phase",
        }:
            output_dim = (
                1 if conditioning_kind == "adaptive_gain" else head_dim // 2
            )
            self.content_actuator = GroupedContentActuator(
                groups=readout_groups,
                content_dim=content_dim,
                output_dim=output_dim,
                hidden_dim=self.conditioning_config["hidden_dim"],
                target=self.conditioning_config["target"],
                coupling=self.conditioning_config["coupling"],
            )
        elif conditioning_kind == "carrier_hypernetwork":
            input_cfg = config["input"]
            position_dim = int(input_cfg["basis_dim"]) + len(
                input_cfg["scalars"]
            )
            if (
                self.fixed_position_pipeline
                and self.conditioning_config["input_mode"]
                in {"position", "content_position"}
            ):
                theta = (
                    rope_theta
                    if input_cfg["theta"] is None
                    else float(input_cfg["theta"])
                )
                self.hyper_position_basis = build_position_basis(
                    kind=input_cfg["kind"],
                    extent=extent,
                    basis_dim=int(input_cfg["basis_dim"]),
                    theta=theta,
                    scalars=input_cfg["scalars"],
                    normalization_extent=input_cfg.get(
                        "normalization_extent"
                    ),
                )
                position_dim = self.hyper_position_basis.output_dim
            self.carrier_hypernetwork = CarrierHypernetwork(
                heads=heads,
                pair_dim=head_dim // 2,
                content_dim=content_dim,
                position_dim=position_dim,
                config=self.conditioning_config,
            )
        elif conditioning_kind != "none":
            output_dim = (
                head_dim
                if self.application == "additive"
                and self.geometry in {"free", "pair_normalized"}
                else head_dim // 2
            )

            def make_conditioner(
                conditioned_dim: int = output_dim,
            ) -> GroupedContentConditioner:
                return GroupedContentConditioner(
                    kind=conditioning_kind,
                    groups=readout_groups,
                    content_dim=content_dim,
                    output_dim=conditioned_dim,
                    hidden_dim=self.conditioning_config["hidden_dim"],
                    gate_init=self.conditioning_config["gate_init"],
                    activation=self.conditioning_config["activation"],
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

    def _hyper_position_features(
        self,
        sequence_length: int,
        *,
        dtype: torch.dtype | None,
    ) -> torch.Tensor:
        if self.hyper_position_basis is not None:
            return self.hyper_position_basis(sequence_length, dtype=dtype)
        if self.pipeline is not None:
            return self.pipeline.basis(sequence_length, dtype=dtype)
        if self.q_pipeline is not None:
            return self.q_pipeline.basis(sequence_length, dtype=dtype)
        # A content-only fixed carrier needs only a length-bearing placeholder.
        placeholder = self.base_cos[:sequence_length]
        return placeholder if dtype is None else placeholder.to(dtype=dtype)

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

    def _unit_pair_to_phase(
        self,
        features: torch.Tensor,
        head: GroupedLinearReadout,
        sequence_length: int,
    ) -> torch.Tensor:
        projected = self._apply_phase_head(features, head)
        half = projected.shape[-1] // 2
        delta_x, delta_y = projected[..., :half], projected[..., half:]
        base_sin = self.base_sin[:sequence_length].to(projected.dtype)[None]
        base_cos = self.base_cos[:sequence_length].to(projected.dtype)[None]
        pair_x = base_cos + delta_x
        pair_y = base_sin + delta_y
        inverse_norm = torch.rsqrt(pair_x.square() + pair_y.square() + 1e-6)
        pair_x = pair_x * inverse_norm
        pair_y = pair_y * inverse_norm
        relative_sin = pair_y * base_cos - pair_x * base_sin
        relative_cos = pair_x * base_cos + pair_y * base_sin
        return torch.atan2(relative_sin, relative_cos)

    def _amplitude(
        self,
        raw: torch.Tensor,
    ) -> torch.Tensor:
        amplitude_init = self.output_config["amplitude_init"]
        if self.output_config["amplitude_parameterization"] == "signed":
            return raw + amplitude_init
        if self.output_config["amplitude_parameterization"] == "bounded_sigmoid":
            maximum = self.output_config["amplitude_max"]
            offset = raw.new_tensor(
                math.log(amplitude_init / (maximum - amplitude_init))
            )
            return maximum * torch.sigmoid(raw + offset)
        if amplitude_init == 0:
            offset = raw.new_tensor(-20.0)
        else:
            offset = raw.new_tensor(amplitude_init).expm1().log()
        return torch.nn.functional.softplus(raw + offset)

    def _scale(self, raw: torch.Tensor) -> torch.Tensor:
        scale_init = self.output_config["scale_init"]
        if self.output_config["scale_parameterization"] == "exp":
            return exp_with_identity_grad(raw) * scale_init
        if self.output_config["scale_parameterization"] == "bounded_log":
            log_limit = math.log(self.output_config["scale_max"])
            return (
                exp_with_identity_grad(raw.tanh() * log_limit) * scale_init
            )
        return raw + scale_init

    def _normalize_additive_output(
        self,
        output: torch.Tensor,
        gain_logit: torch.Tensor,
    ) -> torch.Tensor:
        scale = torch.rsqrt(
            output.float().square().mean(dim=-1, keepdim=True) + 1e-6
        ).to(dtype=output.dtype)
        gain = self.output_config["additive_gain_max"] * torch.sigmoid(
            gain_logit
        )
        if output.ndim == 4:
            gain = gain.unsqueeze(0)
        return output * scale * gain.to(dtype=output.dtype)

    def _normalize_additive_pairs(self, output: torch.Tensor) -> torch.Tensor:
        half = output.shape[-1] // 2
        pair_x, pair_y = output[..., :half], output[..., half:]
        pair_norm = (
            pair_x.float().square() + pair_y.float().square()
        ).sqrt().clamp_min(1e-6)
        inverse_norm = pair_norm.reciprocal().to(dtype=output.dtype)
        amplitude = output.new_tensor(self.output_config["amplitude_init"])
        return torch.cat(
            (amplitude * pair_x * inverse_norm, amplitude * pair_y * inverse_norm),
            dim=-1,
        )

    def _rotate_additive_pairs(
        self,
        output: torch.Tensor,
        phase: torch.Tensor,
    ) -> torch.Tensor:
        if output.ndim == 3:
            output = output.unsqueeze(0).expand(phase.shape[0], -1, -1, -1)
        half = output.shape[-1] // 2
        pair_x, pair_y = output[..., :half], output[..., half:]
        phase_sin, phase_cos = phase.sin(), phase.cos()
        rotated_x = pair_x * phase_cos - pair_y * phase_sin
        rotated_y = pair_x * phase_sin + pair_y * phase_cos
        return torch.cat((rotated_x, rotated_y), dim=-1)

    def _apply_content_phase_rotation(
        self,
        q_output: torch.Tensor,
        k_output: torch.Tensor,
        q_content: torch.Tensor | None,
        k_content: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if q_content is None or k_content is None:
            raise ValueError(
                "Phase-rotation conditioning requires Q/K content."
            )
        target = self.conditioning_config["target"]
        if target in {"q", "both"}:
            q_phase = self.phase_rotation_conditioner.phase(q_content, "q")
            q_output = self._rotate_additive_pairs(q_output, q_phase)
        if target in {"k", "both"}:
            k_phase = self.phase_rotation_conditioner.phase(k_content, "k")
            k_output = self._rotate_additive_pairs(k_output, k_phase)
        return q_output, k_output

    def _amplitude_phase_addend(
        self,
        features: torch.Tensor | None,
        amplitude_head: GroupedLinearReadout | None,
        phase_head: GroupedLinearReadout | None,
        sequence_length: int,
        *,
        dtype: torch.dtype | None = None,
        content: torch.Tensor | None = None,
        amplitude_conditioner: GroupedContentConditioner | None = None,
        phase_conditioner: GroupedContentConditioner | None = None,
        log_gain_delta: torch.Tensor | None = None,
        hyper_phase_delta: torch.Tensor | None = None,
        branch: str = "q",
    ) -> torch.Tensor:
        target_dtype = (
            features.dtype
            if features is not None
            else dtype or self.base_cos.dtype
        )
        fixed = self.base_cos[:sequence_length].to(dtype=target_dtype)
        zero = fixed[None].expand(self.heads, -1, -1).new_zeros(
            self.heads,
            sequence_length,
            self.head_dim // 2,
        )
        amplitude_raw = (
            self._apply_phase_head(features, amplitude_head)
            if amplitude_head is not None
            else zero
        )
        phase = (
            self._apply_phase_head(features, phase_head)
            if phase_head is not None
            else zero
        )
        amplitude = self._amplitude(amplitude_raw)
        if self.conditioning_config["kind"] in {
            "local_residual",
            "content_gate",
        }:
            if content is None:
                raise ValueError(
                    "Amplitude/phase content conditioning requires Q/K content."
                )
            amplitude = amplitude_conditioner(amplitude, content)
            phase = phase_conditioner(phase, content)
        elif self.conditioning_config["kind"] == "additive_phase":
            if content is None:
                raise ValueError(
                    "additive_phase conditioning requires dedicated content"
                )
            phase = phase + self.content_actuator(content, branch)
        if log_gain_delta is not None:
            amplitude = amplitude * exp_with_identity_grad(log_gain_delta)
        if hyper_phase_delta is not None:
            phase = phase + hyper_phase_delta
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
        if self.conditioning_config["kind"] in {
            "none",
            "phase_rotation",
            "adaptive_gain",
            "additive_phase",
            "rope_phase",
            "carrier_hypernetwork",
        }:
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
        if self.fixed_position_pipeline:
            q_features = None
            k_features = None
        elif self.qk_coupling == "separate":
            q_features = self.q_pipeline(sequence_length, dtype=dtype)
            k_features = self.k_pipeline(sequence_length, dtype=dtype)
        else:
            shared = self.pipeline(sequence_length, dtype=dtype)
            q_features = shared
            k_features = shared

        q_log_gain_delta = None
        k_log_gain_delta = None
        q_hyper_phase_delta = None
        k_hyper_phase_delta = None
        if self.carrier_hypernetwork is not None:
            position_features = self._hyper_position_features(
                sequence_length,
                dtype=dtype,
            )
            (
                q_log_gain_delta,
                q_hyper_phase_delta,
                k_log_gain_delta,
                k_hyper_phase_delta,
            ) = self.carrier_hypernetwork(
                q_content=q_content,
                k_content=k_content,
                position=position_features,
            )

        q_scale = None
        k_scale = None
        if self.application == "additive" and self.geometry in {
            "free",
            "pair_normalized",
        }:
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
                    dtype=dtype,
                    content=q_content,
                    amplitude_conditioner=self.amplitude_conditioner,
                    phase_conditioner=self.conditioner,
                    log_gain_delta=q_log_gain_delta,
                    hyper_phase_delta=q_hyper_phase_delta,
                    branch="q",
                )
                k_output = self._amplitude_phase_addend(
                    k_features,
                    self.amplitude_head,
                    self.phase_head,
                    sequence_length,
                    dtype=dtype,
                    content=k_content,
                    amplitude_conditioner=self.amplitude_conditioner,
                    phase_conditioner=self.conditioner,
                    log_gain_delta=k_log_gain_delta,
                    hyper_phase_delta=k_hyper_phase_delta,
                    branch="k",
                )
            else:
                q_output = self._amplitude_phase_addend(
                    q_features,
                    self.q_amplitude_head,
                    self.q_phase_head,
                    sequence_length,
                    dtype=dtype,
                    content=q_content,
                    amplitude_conditioner=self.q_amplitude_conditioner,
                    phase_conditioner=self.q_conditioner,
                    log_gain_delta=q_log_gain_delta,
                    hyper_phase_delta=q_hyper_phase_delta,
                    branch="q",
                )
                k_output = self._amplitude_phase_addend(
                    k_features,
                    self.k_amplitude_head,
                    self.k_phase_head,
                    sequence_length,
                    dtype=dtype,
                    content=k_content,
                    amplitude_conditioner=self.k_amplitude_conditioner,
                    phase_conditioner=self.k_conditioner,
                    log_gain_delta=k_log_gain_delta,
                    hyper_phase_delta=k_hyper_phase_delta,
                    branch="k",
                )
        elif self.geometry in {"projected_phase", "unit_pair"}:
            phase_builder = (
                self._unit_pair_to_phase
                if self.geometry == "unit_pair"
                else self._project_to_phase
            )
            if self.qk_coupling == "shared":
                phase = phase_builder(
                    q_features,
                    self.projected_phase_head,
                    sequence_length,
                )
                q_output, k_output = phase, phase
            else:
                q_output = phase_builder(
                    q_features,
                    self.q_projected_phase_head,
                    sequence_length,
                )
                k_output = phase_builder(
                    k_features,
                    self.k_projected_phase_head,
                    sequence_length,
                )
        else:
            if not self.learn_phase:
                zero = self.base_cos[:sequence_length].to(
                    dtype=dtype or self.base_cos.dtype
                )[None].expand(self.heads, -1, -1)
                q_output = zero.new_zeros(zero.shape)
                k_output = q_output
            elif self.qk_coupling == "shared":
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

        if (
            self.application == "rotary"
            and self.carrier_hypernetwork is not None
        ):
            q_output = q_output + q_hyper_phase_delta
            k_output = k_output + k_hyper_phase_delta
            if self.conditioning_config["components"] == "log_gain_phase":
                q_hyper_scale = exp_with_identity_grad(q_log_gain_delta)
                k_hyper_scale = exp_with_identity_grad(k_log_gain_delta)
                q_scale = (
                    q_hyper_scale
                    if q_scale is None
                    else q_scale * q_hyper_scale
                )
                k_scale = (
                    k_hyper_scale
                    if k_scale is None
                    else k_scale * k_hyper_scale
                )

        q_output = q_output * self.output_config["phase_scale"] if (
            self.application == "rotary"
        ) else q_output
        k_output = k_output * self.output_config["phase_scale"] if (
            self.application == "rotary"
        ) else k_output
        if self.conditioning_config["kind"] == "rope_phase":
            if q_content is None or k_content is None:
                raise ValueError("rope_phase conditioning requires dedicated content")
            q_output = q_output + self.content_actuator(q_content, "q")
            k_output = k_output + self.content_actuator(k_content, "k")
        q_output, k_output = self._condition_outputs(
            q_output,
            k_output,
            q_content,
            k_content,
        )
        if self.application == "additive" and self.geometry == "pair_normalized":
            q_output = self._normalize_additive_pairs(q_output)
            k_output = self._normalize_additive_pairs(k_output)
        if self.conditioning_config["kind"] == "phase_rotation":
            q_output, k_output = self._apply_content_phase_rotation(
                q_output,
                k_output,
                q_content,
                k_content,
            )
        if (
            self.application == "additive"
            and self.output_config["additive_normalization"] == "rms"
        ):
            if self.qk_coupling == "shared":
                q_output = self._normalize_additive_output(
                    q_output, self.additive_gain
                )
                k_output = self._normalize_additive_output(
                    k_output, self.additive_gain
                )
            else:
                q_output = self._normalize_additive_output(
                    q_output, self.q_additive_gain
                )
                k_output = self._normalize_additive_output(
                    k_output, self.k_additive_gain
                )
        q_gain = None
        k_gain = None
        if self.conditioning_config["kind"] == "adaptive_gain":
            if q_content is None or k_content is None:
                raise ValueError("adaptive_gain conditioning requires dedicated content")
            q_gain = exp_with_identity_grad(
                self.content_actuator(q_content, "q")
            )
            k_gain = exp_with_identity_grad(
                self.content_actuator(k_content, "k")
            )
        return QKPositionOutput(
            self.application,
            q_output,
            k_output,
            q_scale=q_scale,
            k_scale=k_scale,
            q_gain=q_gain,
            k_gain=k_gain,
            q_log_gain_delta=q_log_gain_delta,
            k_log_gain_delta=k_log_gain_delta,
            q_hyper_phase_delta=q_hyper_phase_delta,
            k_hyper_phase_delta=k_hyper_phase_delta,
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
        if self.phase_rotation_conditioner is not None:
            self.phase_rotation_conditioner.reset_output_parameters()
        if self.content_actuator is not None:
            self.content_actuator.reset_output_parameters()
        if self.carrier_hypernetwork is not None:
            self.carrier_hypernetwork.reset_output_parameters()

    def summarize(
        self,
        sequence_length: int,
        *,
        dtype: torch.dtype | None = None,
        q_ref: torch.Tensor | None = None,
        k_ref: torch.Tensor | None = None,
        q_content: torch.Tensor | None = None,
        k_content: torch.Tensor | None = None,
    ) -> dict[str, float]:
        """Compact finite diagnostics for Q/K channel outputs."""
        if self.conditioning_config["kind"] == "none":
            output = self.forward(sequence_length, dtype=dtype)
        else:
            parameter = next(self.parameters())
            content_dim = self.content_dim
            if q_content is None:
                q_content = torch.zeros(
                    1,
                    self.heads,
                    sequence_length,
                    content_dim,
                    device=parameter.device,
                    dtype=dtype or parameter.dtype,
                )
            if k_content is None:
                k_content = torch.zeros_like(q_content)
            output = self.forward(
                sequence_length,
                dtype=dtype,
                q_content=q_content,
                k_content=k_content,
            )
        metrics: dict[str, float] = {}

        def _stats(prefix: str, tensor: torch.Tensor) -> None:
            values = tensor.detach().float()
            metrics[f"{prefix}/mean"] = values.mean().item()
            metrics[f"{prefix}/std"] = values.std().item()
            metrics[f"{prefix}/rms"] = values.pow(2).mean().sqrt().item()
            metrics[f"{prefix}/abs_max"] = values.abs().max().item()

        def _delta_stats(prefix: str, tensor: torch.Tensor) -> None:
            values = tensor.detach().float()
            absolute = values.abs().flatten()
            metrics[f"{prefix}/rms"] = values.square().mean().sqrt().item()
            metrics[f"{prefix}/p95_abs"] = torch.quantile(
                absolute, 0.95
            ).item()

        if q_content is not None:
            _stats("dedicated_content_q", q_content)
        if k_content is not None:
            _stats("dedicated_content_k", k_content)

        if self.fixed_position_pipeline:
            q_frequency = 1.0 / (
                self.rope_theta
                ** (
                    torch.arange(
                        0,
                        self.head_dim,
                        2,
                        device=self.base_cos.device,
                        dtype=torch.float32,
                    )
                    / self.head_dim
                )
            )
            k_frequency = q_frequency
        else:
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
        if self.conditioning_config["kind"] == "phase_rotation":
            target = self.conditioning_config["target"]
            if target in {"q", "both"}:
                _stats(
                    "content_phase_q",
                    self.phase_rotation_conditioner.phase(q_content, "q"),
                )
            if target in {"k", "both"}:
                _stats(
                    "content_phase_k",
                    self.phase_rotation_conditioner.phase(k_content, "k"),
                )
        if self.conditioning_config["kind"] in {
            "additive_phase",
            "rope_phase",
        }:
            _stats("content_phase_q", self.content_actuator(q_content, "q"))
            _stats("content_phase_k", self.content_actuator(k_content, "k"))
        if output.q_gain is not None:
            _stats("content_gain_q", output.q_gain)
            _stats("content_gain_k", output.k_gain)
        if output.q_hyper_phase_delta is not None:
            _delta_stats(
                "hyper_phase_delta_q", output.q_hyper_phase_delta
            )
            _delta_stats(
                "hyper_phase_delta_k", output.k_hyper_phase_delta
            )
        if (
            output.q_log_gain_delta is not None
            and self.conditioning_config["components"] == "log_gain_phase"
        ):
            _delta_stats(
                "hyper_log_gain_delta_q", output.q_log_gain_delta
            )
            _delta_stats(
                "hyper_log_gain_delta_k", output.k_log_gain_delta
            )
            metrics["hyper_effective_gain_q/max"] = (
                output.q_log_gain_delta.detach().float().exp().max().item()
            )
            metrics["hyper_effective_gain_k/max"] = (
                output.k_log_gain_delta.detach().float().exp().max().item()
            )
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
            if self.output_config["additive_normalization"] == "rms":
                if self.qk_coupling == "shared":
                    q_gain = k_gain = self.additive_gain
                else:
                    q_gain = self.q_additive_gain
                    k_gain = self.k_additive_gain
                gain_max = self.output_config["additive_gain_max"]
                metrics["additive_gain_q_mean"] = (
                    gain_max * torch.sigmoid(q_gain.detach().float())
                ).mean().item()
                metrics["additive_gain_k_mean"] = (
                    gain_max * torch.sigmoid(k_gain.detach().float())
                ).mean().item()
            if q_ref is not None:
                q_values = q_ref.detach().float()
                q_addend = output.q.detach().float()
                if q_addend.ndim == 3:
                    q_addend = q_addend.unsqueeze(0)
                q_rms_per_token = q_values.square().mean(dim=-1).sqrt().clamp_min(
                    1e-12
                )
                q_addend_rms_per_token = q_addend.square().mean(dim=-1).sqrt()
                q_ratio = q_addend_rms_per_token / q_rms_per_token
                q_rms = q_values.pow(2).mean().sqrt().clamp_min(1e-12)
                metrics["addend_q_to_q_rms_ratio"] = (
                    q_addend.pow(2).mean().sqrt() / q_rms
                ).item()
                metrics["addend_q_to_q_ratio_p95"] = torch.quantile(
                    q_ratio, 0.95
                ).item()
                metrics["q_content_combined_cosine_mean"] = (
                    torch.nn.functional.cosine_similarity(
                        q_values,
                        q_values + q_addend,
                        dim=-1,
                    )
                    .mean()
                    .item()
                )
            if k_ref is not None:
                k_values = k_ref.detach().float()
                k_addend = output.k.detach().float()
                if k_addend.ndim == 3:
                    k_addend = k_addend.unsqueeze(0)
                k_rms_per_token = k_values.square().mean(dim=-1).sqrt().clamp_min(
                    1e-12
                )
                k_addend_rms_per_token = k_addend.square().mean(dim=-1).sqrt()
                k_ratio = k_addend_rms_per_token / k_rms_per_token
                k_rms = k_values.pow(2).mean().sqrt().clamp_min(1e-12)
                metrics["addend_k_to_k_rms_ratio"] = (
                    k_addend.pow(2).mean().sqrt() / k_rms
                ).item()
                metrics["addend_k_to_k_ratio_p95"] = torch.quantile(
                    k_ratio, 0.95
                ).item()
                metrics["k_content_combined_cosine_mean"] = (
                    torch.nn.functional.cosine_similarity(
                        k_values,
                        k_values + k_addend,
                        dim=-1,
                    )
                    .mean()
                    .item()
                )
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
        content_dim: int,
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
            head_dim=content_dim,
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
        frequency = exp_with_identity_grad(self.log_frequency.float())
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


class FactorizedPairwiseLogit(torch.nn.Module):
    """Low-rank content/offset interaction with an exact zero-effect gate."""

    def __init__(
        self,
        *,
        groups: int,
        heads: int,
        content_dim: int,
        position_dim: int,
        rank: int,
        position_mode: str,
        gate_init: float,
    ):
        super().__init__()
        self.groups = groups
        self.heads = heads
        self.rank = rank
        self.position_mode = position_mode
        self.query_content = torch.nn.Parameter(
            torch.empty(groups, content_dim, rank)
        )
        self.key_content = torch.nn.Parameter(
            torch.empty(groups, content_dim, rank)
        )
        self.relative_position = torch.nn.Parameter(
            torch.empty(groups, position_dim, rank)
        )
        if position_mode in {"query_absolute", "full_absolute"}:
            self.query_position = torch.nn.Parameter(
                torch.empty(groups, position_dim, rank)
            )
        else:
            self.register_parameter("query_position", None)
        if position_mode == "full_absolute":
            self.key_position = torch.nn.Parameter(
                torch.empty(groups, position_dim, rank)
            )
        else:
            self.register_parameter("key_position", None)
        self.gate = torch.nn.Parameter(
            torch.full((groups,), float(gate_init))
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for parameter in (
            self.query_content,
            self.key_content,
            self.relative_position,
            self.query_position,
            self.key_position,
        ):
            if parameter is not None:
                for weight in parameter:
                    torch.nn.init.xavier_normal_(weight)

    def _project_content(
        self,
        content: torch.Tensor,
        weight: torch.Tensor,
    ) -> torch.Tensor:
        if self.groups == 1:
            return torch.einsum("bhld,dr->bhlr", content, weight[0])
        return torch.einsum("bhld,hdr->bhlr", content, weight)

    def _project_position(
        self,
        position: torch.Tensor,
        weight: torch.Tensor,
    ) -> torch.Tensor:
        if self.groups == 1:
            return torch.einsum("hld,dr->hlr", position, weight[0])
        return torch.einsum("hld,hdr->hlr", position, weight)

    @staticmethod
    def _unit_rms(value: torch.Tensor) -> torch.Tensor:
        return value * torch.rsqrt(
            value.float().square().mean(dim=-1, keepdim=True) + 1e-6
        ).to(dtype=value.dtype)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        position_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        query_length = query.shape[-2]
        key_length = key.shape[-2]
        if max(query_length, key_length) > position_features.shape[-2]:
            raise ValueError(
                "Pairwise logit sequence length exceeds position-feature extent."
            )

        query_factor = self._project_content(query, self.query_content)
        key_factor = self._project_content(key, self.key_content)
        if self.query_position is not None:
            query_factor = query_factor + self._project_position(
                position_features[:, :query_length],
                self.query_position,
            )[None]
        if self.key_position is not None:
            key_factor = key_factor + self._project_position(
                position_features[:, :key_length],
                self.key_position,
            )[None]
        distance_factor = self._project_position(
            position_features,
            self.relative_position,
        )
        gate = (
            self.gate.expand(self.heads)
            if self.groups == 1
            else self.gate
        )
        return (
            self._unit_rms(query_factor),
            self._unit_rms(key_factor),
            self._unit_rms(distance_factor),
            gate,
        )


class LogitBiasChannel(PositionChannel):
    """Relative-distance scalar logit curves with a fixed ``[heads, extent]`` contract."""

    def __init__(
        self,
        config: dict,
        *,
        heads: int,
        head_dim: int,
        model_dim: int,
        content_dim: int,
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
        self.pairwise = None
        self._last_routing_summary: dict[str, torch.Tensor] = {}
        if self.conditioning_kind in {"inkling_table", "inkling_cosnet"}:
            self.inkling = InklingProfileBank(
                kind=self.conditioning_kind,
                groups=_readout_groups(self.head_coupling, heads),
                heads=heads,
                content_dim=content_dim,
                extent=extent,
                config=conditioning,
            )
        elif self.conditioning_kind == "pairwise_low_rank":
            self.pairwise = FactorizedPairwiseLogit(
                groups=_readout_groups(self.head_coupling, heads),
                heads=heads,
                content_dim=content_dim,
                position_dim=head_dim,
                rank=conditioning["pair_rank"],
                position_mode=conditioning["position_mode"],
                gate_init=conditioning["gate_init"],
            )

    def prepare(
        self,
        *,
        dtype: torch.dtype | None = None,
        query: torch.Tensor | None = None,
        key: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None,
    ]:
        features = self.pipeline(dtype=dtype)
        if self.head_coupling == "shared_head":
            base = self.scalar_head(features[:1], self.heads)
        else:
            base = self.scalar_head(features, self.heads)
        if self.inkling is not None:
            if query is None:
                raise ValueError(
                    f"{self.conditioning_kind} logit bias requires normalized "
                    "query content."
                )
            conditional, routing = self.inkling(query)
            with torch.no_grad():
                routing_f = routing.detach().float()
                entropy = -(
                    routing_f.clamp_min(1e-9).log() * routing_f
                ).sum(-1)
                self._last_routing_summary = {
                    "routing_entropy_mean": entropy.mean(),
                    "routing_max_probability": routing_f.max(
                        dim=-1
                    ).values.mean(),
                    "inkling_gate_abs_mean": (
                        self.inkling.gate.detach().float().abs().mean()
                    ),
                }
            return base[None, :, None, :] + conditional, None
        if self.pairwise is not None:
            if query is None or key is None:
                raise ValueError(
                    "pairwise_low_rank logit bias requires normalized query "
                    "and key content."
                )
            return base, self.pairwise(query, key, features)
        return base, None

    def forward(
        self,
        *,
        dtype: torch.dtype | None = None,
        query: torch.Tensor | None = None,
        key: torch.Tensor | None = None,
    ) -> torch.Tensor:
        base, _ = self.prepare(dtype=dtype, query=query, key=key)
        return base

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
        if self.pairwise is not None:
            with torch.no_grad():
                self.pairwise.gate.fill_(
                    float(self.config["conditioning"]["gate_init"])
                )

    def routing_summary(self) -> dict[str, float]:
        summary = {
            key: value.item()
            for key, value in self._last_routing_summary.items()
        }
        if self.pairwise is not None:
            gate = self.pairwise.gate.detach().float()
            summary["pairwise_gate_mean"] = gate.mean().item()
            summary["pairwise_gate_abs_max"] = gate.abs().max().item()
        return summary


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
    """Write attended or query-local position features to the residual stream."""

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
        self.query_projection = None
        if self.mode == "query_position":
            self.register_parameter("gate", None)
            self.query_projection = torch.nn.Linear(
                model_dim,
                model_dim,
                bias=True,
            )
            torch.nn.init.zeros_(self.query_projection.weight)
            torch.nn.init.zeros_(self.query_projection.bias)
        else:
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

    def query_output(
        self,
        length: int,
        *,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if self.mode != "query_position" or self.query_projection is None:
            raise ValueError("query_output is only valid for query_position mode")
        by_head = self.pipeline(length, dtype=dtype)
        merged = by_head.permute(1, 0, 2).reshape(length, self.model_dim)
        return self.query_projection(merged)

    def forward(self, summary: torch.Tensor) -> torch.Tensor:
        if self.mode == "query_position":
            raise ValueError("query_position writes do not consume attention summaries")
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
        if self.query_projection is not None:
            torch.nn.init.zeros_(self.query_projection.weight)
            torch.nn.init.zeros_(self.query_projection.bias)
        else:
            with torch.no_grad():
                self.gate.fill_(float(self.config["gate_init"]))


def build_qk_position_channel(
    config: dict,
    *,
    heads: int,
    head_dim: int,
    model_dim: int,
    content_dim: int | None = None,
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
        content_dim=(
            content_dim
            if content_dim is not None
            else (
                model_dim
                if resolved["conditioning"]["source"] == "residual"
                else head_dim
            )
        ),
        extent=extent,
        rope_theta=rope_theta,
    )


def build_logit_bias_channel(
    config: dict,
    *,
    heads: int,
    head_dim: int,
    model_dim: int,
    content_dim: int | None = None,
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
        content_dim=content_dim or head_dim,
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
    content_total = 0
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
            content_total += _unique_numel(
                getattr(attn, "position_content", None)
            )
            attention_write_total += _unique_numel(
                getattr(attn, "position_write", None)
            )
    position_total = (
        qk_total
        + logit_total
        + content_total
        + attention_write_total
        + residual_stream_total
    )
    return {
        "qk_position_params": qk_total,
        "logit_bias_params": logit_total,
        "position_content_params": content_total,
        "attention_write_params": attention_write_total,
        "residual_stream_params": residual_stream_total,
        "position_params": position_total,
    }
