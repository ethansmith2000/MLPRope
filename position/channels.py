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
from position.precision import PreserveFP32BuffersMixin
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
    q_amplitude_delta: torch.Tensor | None = None
    k_amplitude_delta: torch.Tensor | None = None
    q_hyper_phase_delta: torch.Tensor | None = None
    k_hyper_phase_delta: torch.Tensor | None = None
    q_frequency_delta: torch.Tensor | None = None
    k_frequency_delta: torch.Tensor | None = None
    q_cartesian_real_delta: torch.Tensor | None = None
    k_cartesian_real_delta: torch.Tensor | None = None
    q_cartesian_imag_delta: torch.Tensor | None = None
    k_cartesian_imag_delta: torch.Tensor | None = None


@dataclass
class CarrierDeltas:
    amplitude: torch.Tensor | None = None
    phase: torch.Tensor | None = None
    frequency: torch.Tensor | None = None
    cartesian_real: torch.Tensor | None = None
    cartesian_imag: torch.Tensor | None = None


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


class _MixedReadout(torch.nn.Module):
    """Block-diagonal readout plus a low-rank cross-head residual.

    The grouped path is the existing per-head readout. The residual lets head
    ``h`` read every head's post-nonlinearity features through a rank-``r``
    bottleneck, at a fraction of a dense ``[groups*hidden, groups*out]`` map.

    Init follows LoRA: the down matrix is random with a *rank-independent*
    fan-in and the up matrix is zero, so the residual is exactly zero at step 0
    and the carrier keeps its exact anchor. Output is scaled by ``alpha / rank``
    so the effective update magnitude does not change with rank -- without it a
    rank sweep measures learning rate rather than capacity.
    """

    def __init__(
        self,
        *,
        groups: int,
        hidden_dim: int,
        output_dim: int,
        rank: int,
        alpha: float,
    ):
        super().__init__()
        self.groups = groups
        self.output_dim = output_dim
        self.rank = rank
        self.scale = alpha / rank
        self.grouped = GroupedLinearReadout(
            groups, hidden_dim, output_dim, init="zeros"
        )
        fan_in = groups * hidden_dim
        self.down = torch.nn.Parameter(
            torch.randn(fan_in, rank) * (2.0 / fan_in) ** 0.5
        )
        self.up = torch.nn.Parameter(torch.zeros(rank, groups * output_dim))

    def reset_parameters(self) -> None:
        """Return to the exact carrier anchor: grouped path and up matrix zero."""
        self.grouped.reset_parameters()
        torch.nn.init.zeros_(self.up)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        batch, groups, length, width = hidden.shape
        grouped = torch.einsum("bgld,gdo->bglo", hidden, self.grouped.weight)
        grouped = grouped + self.grouped.bias[None, :, None, :]
        flat = hidden.permute(0, 2, 1, 3).reshape(batch, length, groups * width)
        residual = (flat @ self.down @ self.up) * self.scale
        residual = (
            residual.view(batch, length, groups, self.output_dim)
            .permute(0, 2, 1, 3)
        )
        return grouped + residual


class CarrierHypernetwork(PreserveFP32BuffersMixin, torch.nn.Module):
    """Anchor-relative token-local gain/phase deltas for additive or rotary Q/K."""

    _fp32_buffer_names = ("spectral_tilt", "spectral_omega")

    def __init__(
        self,
        *,
        heads: int,
        pair_dim: int,
        content_dim: int,
        position_dim: int,
        config: dict,
        inverse_frequency: torch.Tensor,
    ):
        super().__init__()
        self.heads = heads
        self.pair_dim = pair_dim
        self.content_dim = content_dim
        self.position_dim = position_dim
        self.input_mode = config["input_mode"]
        self.input_normalization = config["input_normalization"]
        self.learnable_input_gains = config["learnable_input_gains"]
        self.network = config["network"]
        self.components = config["components"]
        self.target = config["target"]
        self.coupling = config["coupling"]
        self.head_coupling = config["head_coupling"]
        self.groups = (
            1 if self.head_coupling == "shared_head" else self.heads
        )
        pairwise_components = {
            "phase": ("phase",),
            "amplitude": ("amplitude",),
            "amplitude_phase": ("amplitude", "phase"),
            "cartesian": ("cartesian_real", "cartesian_imag"),
        }
        # Spectral parameterizations predict one scalar per head per token and
        # expand it over the frequency axis with a fixed profile. Both are
        # translation-invariant: `slope` tilts the amplitude envelope across
        # log-frequency (locality), and `offset` shifts effective position by
        # phase proportional to omega, i.e. cis(omega*((p+m_q)-(p+m_k))).
        spectral_components = {
            "amplitude_slope": ("gain", "slope"),
            "position_offset": ("offset",),
            "slope_offset": ("gain", "slope", "offset"),
        }
        # Mixed modes narrow exactly one branch so the cost of compressing the
        # amplitude readout can be separated from the cost of compressing the
        # angular readout.
        mixed_components = {
            "amplitude_offset": ("amplitude", "offset"),
            "slope_phase": ("gain", "slope", "phase"),
        }
        narrow_names = {"gain", "slope", "offset"}
        if self.components in spectral_components:
            names = spectral_components[self.components]
        elif self.components in mixed_components:
            names = mixed_components[self.components]
        else:
            names = pairwise_components[self.components]
        self.component_names = names
        self.component_widths = tuple(
            1 if name in narrow_names else pair_dim for name in names
        )
        self.spectral = any(name in narrow_names for name in names)
        self.offset_bound = float(config["offset_bound"])
        self.offset_parameterization = config["offset_parameterization"]
        omega = inverse_frequency.detach().to(torch.float32).reshape(-1)
        if omega.numel() != pair_dim:
            raise ValueError(
                "Carrier hypernetwork frequency vector must have pair_dim "
                f"entries, got {omega.numel()} for pair_dim={pair_dim}"
            )
        log_omega = omega.clamp_min(1e-12).log()
        centered = log_omega - log_omega.mean()
        self.register_buffer(
            "spectral_tilt",
            centered / centered.std(unbiased=False).clamp_min(1e-6),
            persistent=False,
        )
        self.register_buffer("spectral_omega", omega, persistent=False)
        input_dim = 0
        if self.input_mode in {"content", "content_position"}:
            input_dim += content_dim
        if self.input_mode in {"position", "content_position"}:
            input_dim += position_dim
        output_dim = sum(self.component_widths)
        learn_content_gain = (
            self.learnable_input_gains
            and self.input_mode in {"content", "content_position"}
        )
        learn_position_gain = (
            self.learnable_input_gains
            and self.input_mode in {"position", "content_position"}
        )
        self.content_input_gain = (
            torch.nn.Parameter(torch.ones(())) if learn_content_gain else None
        )
        self.position_input_gain = (
            torch.nn.Parameter(torch.ones(())) if learn_position_gain else None
        )

        def make_trunk() -> _GroupedHyperTrunk:
            return _GroupedHyperTrunk(
                groups=self.groups,
                input_dim=input_dim,
                hidden_dim=config["hidden_dim"],
                network=self.network,
            )

        self.readout_head_mixing = config["readout_head_mixing"]
        self.readout_mix_rank = int(config["readout_mix_rank"])
        self.readout_output_dim = output_dim

        def make_readout(trunk: _GroupedHyperTrunk):
            if self.readout_head_mixing == "lowrank":
                return _MixedReadout(
                    groups=self.groups,
                    hidden_dim=trunk.output_dim,
                    output_dim=output_dim,
                    rank=self.readout_mix_rank,
                    alpha=float(config["readout_mix_alpha"]),
                )
            if self.readout_head_mixing == "dense":
                # Every head reads every head's post-nonlinearity features,
                # mirroring how one dense W_q projection feeds all heads.
                # Grouping the trunk costs nothing (its input is identical
                # across heads), but grouping the readout confines head h to
                # its own 64 nonlinear features.
                return GroupedLinearReadout(
                    1,
                    self.groups * trunk.output_dim,
                    self.groups * output_dim,
                    init="zeros",
                )
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

    def _prepare_modality(
        self,
        values: torch.Tensor,
        gain: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.input_normalization == "modality_rms":
            inverse_rms = torch.rsqrt(
                values.float().square().mean(dim=-1, keepdim=True) + 1e-6
            ).to(dtype=values.dtype)
            values = values * inverse_rms
        if gain is not None:
            values = values * gain.to(dtype=values.dtype)
        return values

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
            content_values = content[:, :1] if self.groups == 1 else content
            pieces.append(
                self._prepare_modality(content_values, self.content_input_gain)
            )
        if self.input_mode in {"position", "content_position"}:
            position_values = position.to(
                dtype=(
                    content.dtype
                    if content is not None
                    else position.dtype
                )
            )
            position_values = position_values[None, None].expand(
                batch, self.groups, -1, -1
            )
            pieces.append(
                self._prepare_modality(
                    position_values,
                    self.position_input_gain,
                )
            )
        return pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=-1)

    def _read(
        self,
        hidden: torch.Tensor,
        readout: GroupedLinearReadout,
    ) -> torch.Tensor:
        if self.readout_head_mixing == "lowrank":
            return readout(hidden)
        if self.readout_head_mixing == "dense":
            batch, groups, length, width = hidden.shape
            flat = hidden.permute(0, 2, 1, 3).reshape(batch, length, groups * width)
            mixed = flat @ readout.weight[0] + readout.bias[0]
            return (
                mixed.view(batch, length, groups, self.readout_output_dim)
                .permute(0, 2, 1, 3)
                .contiguous()
            )
        output = torch.einsum("bgld,gdo->bglo", hidden, readout.weight)
        return output + readout.bias[None, :, None, :]

    def _branch(
        self,
        values: torch.Tensor,
        branch: str,
    ) -> torch.Tensor:
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

    def _parse(self, raw: torch.Tensor | None) -> CarrierDeltas:
        if raw is None:
            return CarrierDeltas()
        values = dict(
            zip(
                self.component_names,
                raw.split(list(self.component_widths), dim=-1),
                strict=True,
            )
        )
        if not self.spectral:
            return CarrierDeltas(**values)
        # Pairwise components pass through untouched; narrow ones are expanded
        # over the frequency axis below.
        deltas: dict[str, torch.Tensor] = {
            name: tensor
            for name, tensor in values.items()
            if name not in {"gain", "slope", "offset"}
        }
        if "slope" in values:
            amplitude = values["slope"] * self.spectral_tilt.to(
                dtype=raw.dtype
            )
            if "gain" in values:
                amplitude = amplitude + values["gain"]
            deltas["amplitude"] = amplitude
        if "offset" in values:
            # Phase proportional to omega is exactly a shift of effective
            # position, so the logit still depends only on the difference of
            # the two shifted positions.
            shift = self.offset_bound * torch.tanh(values["offset"])
            deltas["phase"] = shift * self.spectral_omega.to(dtype=raw.dtype)
        return CarrierDeltas(**deltas)

    def forward(
        self,
        *,
        q_content: torch.Tensor | None,
        k_content: torch.Tensor | None,
        position: torch.Tensor,
    ) -> tuple[CarrierDeltas, CarrierDeltas]:
        q_values = self._inputs(q_content, position)
        k_values = self._inputs(k_content, position)
        if self.coupling == "shared" and self.target == "both":
            q_raw = self._branch(q_values, "q")
            k_raw = q_raw
        else:
            output_dim = self.readout_output_dim
            q_raw = (
                self._branch(q_values, "q")
                if self.target in {"q", "both"}
                else q_values.new_zeros(*q_values.shape[:-1], output_dim)
            )
            k_raw = (
                self._branch(k_values, "k")
                if self.target in {"k", "both"}
                else k_values.new_zeros(*k_values.shape[:-1], output_dim)
            )
        q_raw = self._expand_heads(q_raw)
        k_raw = self._expand_heads(k_raw)
        return self._parse(q_raw), self._parse(k_raw)

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


class QKPositionChannel(PreserveFP32BuffersMixin, PositionChannel):
    """Q/K absolute-position channel with configurable coupling and geometry."""

    _fp32_buffer_names = ("base_sin", "base_cos", "base_angle")

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
        self.parameter_source = self.output_config["parameter_source"]
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
            self.fixed_amplitude_phase
            or self.fixed_rotary_phase
            or self.parameter_source == "direct"
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
        self.direct_amplitude_raw = None
        self.q_direct_amplitude_raw = None
        self.k_direct_amplitude_raw = None
        self.direct_phase = None
        self.q_direct_phase = None
        self.k_direct_phase = None
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
        frequencies = torch.arange(head_dim // 2, dtype=torch.float32)
        inverse_frequency = 1.0 / (
            rope_theta ** (frequencies / (head_dim // 2))
        )
        base_angle = torch.outer(
            torch.arange(extent, dtype=torch.float32),
            inverse_frequency,
        )
        self.register_buffer("base_angle", base_angle, persistent=False)

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
            if self.parameter_source == "direct":
                direct_shape = (readout_groups, head_dim // 2)
                static_complement = (
                    self.conditioning_config["kind"] == "carrier_hypernetwork"
                    and self.conditioning_config["components"]
                    == "amplitude_phase"
                    and self.conditioning_config["static_complement"]
                )
                dynamic_target = self.conditioning_config["target"]

                def make_direct(enabled: bool) -> torch.nn.Parameter | None:
                    return (
                        torch.nn.Parameter(torch.zeros(direct_shape))
                        if enabled
                        else None
                    )

                if self.qk_coupling == "shared":
                    self.direct_amplitude_raw = make_direct(self.learn_amplitude)
                    self.direct_phase = make_direct(self.learn_phase)
                else:
                    q_static = static_complement and dynamic_target == "k"
                    k_static = static_complement and dynamic_target == "q"
                    self.q_direct_amplitude_raw = make_direct(
                        self.learn_amplitude or q_static
                    )
                    self.k_direct_amplitude_raw = make_direct(
                        self.learn_amplitude or k_static
                    )
                    self.q_direct_phase = make_direct(
                        self.learn_phase or q_static
                    )
                    self.k_direct_phase = make_direct(
                        self.learn_phase or k_static
                    )
            elif self.qk_coupling == "shared":
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
        elif self.application == "rotary" and self.geometry == "phase":
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
                inverse_frequency=inverse_frequency,
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

    def _direct_carrier_value(
        self,
        value: torch.Tensor | None,
        sequence_length: int,
        *,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        if value is None:
            return None
        expanded = _expand_shared_readout(
            value[:, None, :],
            self.head_coupling,
            self.heads,
        )
        return expanded.to(dtype=dtype).expand(-1, sequence_length, -1)

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
        phase_sin = phase.float().sin().to(dtype=output.dtype)
        phase_cos = phase.float().cos().to(dtype=output.dtype)
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
        amplitude_delta: torch.Tensor | None = None,
        hyper_phase_delta: torch.Tensor | None = None,
        frequency_delta: torch.Tensor | None = None,
        cartesian_real_delta: torch.Tensor | None = None,
        cartesian_imag_delta: torch.Tensor | None = None,
        amplitude_raw_override: torch.Tensor | None = None,
        phase_override: torch.Tensor | None = None,
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
        amplitude_raw = amplitude_raw_override
        if amplitude_raw is None:
            amplitude_raw = (
                self._apply_phase_head(features, amplitude_head)
                if amplitude_head is not None
                else zero
            )
        phase = phase_override
        if phase is None:
            phase = (
                self._apply_phase_head(features, phase_head)
                if phase_head is not None
                else zero
            )
        if cartesian_real_delta is not None:
            if cartesian_imag_delta is None:
                raise ValueError("Cartesian carrier requires both residual components")
            base_sin = self.base_sin[:sequence_length].to(target_dtype)[None]
            base_cos = self.base_cos[:sequence_length].to(target_dtype)[None]
            real = 1.0 + cartesian_real_delta
            imag = cartesian_imag_delta
            cos = base_cos * real - base_sin * imag
            sin = base_sin * real + base_cos * imag
            return torch.cat((cos, sin), dim=-1)
        if cartesian_imag_delta is not None:
            raise ValueError("Cartesian carrier requires both residual components")
        if amplitude_delta is not None:
            amplitude_raw = amplitude_delta
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
        if hyper_phase_delta is not None:
            phase = phase + hyper_phase_delta
        phase = phase * self.output_config["phase_scale"]
        if frequency_delta is not None:
            base_angle = self.base_angle[:sequence_length][None]
            total_phase = (1.0 + frequency_delta.float()) * (
                base_angle + phase.float()
            )
            sin = total_phase.sin().to(dtype=phase.dtype)
            cos = total_phase.cos().to(dtype=phase.dtype)
        else:
            base_sin = self.base_sin[:sequence_length].float()[None]
            base_cos = self.base_cos[:sequence_length].float()[None]
            delta_sin = phase.float().sin()
            delta_cos = phase.float().cos()
            sin = (base_sin * delta_cos + base_cos * delta_sin).to(
                dtype=phase.dtype
            )
            cos = (base_cos * delta_cos - base_sin * delta_sin).to(
                dtype=phase.dtype
            )
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

        q_amplitude_delta = None
        k_amplitude_delta = None
        q_hyper_phase_delta = None
        k_hyper_phase_delta = None
        q_frequency_delta = None
        k_frequency_delta = None
        q_cartesian_real_delta = None
        k_cartesian_real_delta = None
        q_cartesian_imag_delta = None
        k_cartesian_imag_delta = None
        if self.carrier_hypernetwork is not None:
            position_features = self._hyper_position_features(
                sequence_length,
                dtype=dtype,
            )
            q_deltas, k_deltas = self.carrier_hypernetwork(
                q_content=q_content,
                k_content=k_content,
                position=position_features,
            )
            q_amplitude_delta = q_deltas.amplitude
            k_amplitude_delta = k_deltas.amplitude
            q_hyper_phase_delta = q_deltas.phase
            k_hyper_phase_delta = k_deltas.phase
            q_frequency_delta = q_deltas.frequency
            k_frequency_delta = k_deltas.frequency
            q_cartesian_real_delta = q_deltas.cartesian_real
            k_cartesian_real_delta = k_deltas.cartesian_real
            q_cartesian_imag_delta = q_deltas.cartesian_imag
            k_cartesian_imag_delta = k_deltas.cartesian_imag
            if self.conditioning_config["components"] == "amplitude_phase":
                dynamic_target = self.conditioning_config["target"]
                if dynamic_target not in {"q", "both"}:
                    q_amplitude_delta = None
                    q_hyper_phase_delta = None
                if dynamic_target not in {"k", "both"}:
                    k_amplitude_delta = None
                    k_hyper_phase_delta = None

        target_dtype = dtype or self.base_cos.dtype
        q_direct_amplitude = None
        k_direct_amplitude = None
        q_direct_phase = None
        k_direct_phase = None
        if self.parameter_source == "direct":
            if self.qk_coupling == "shared":
                direct_amplitude = self._direct_carrier_value(
                    self.direct_amplitude_raw,
                    sequence_length,
                    dtype=target_dtype,
                )
                direct_phase = self._direct_carrier_value(
                    self.direct_phase,
                    sequence_length,
                    dtype=target_dtype,
                )
                q_direct_amplitude = k_direct_amplitude = direct_amplitude
                q_direct_phase = k_direct_phase = direct_phase
            else:
                q_direct_amplitude = self._direct_carrier_value(
                    self.q_direct_amplitude_raw,
                    sequence_length,
                    dtype=target_dtype,
                )
                k_direct_amplitude = self._direct_carrier_value(
                    self.k_direct_amplitude_raw,
                    sequence_length,
                    dtype=target_dtype,
                )
                q_direct_phase = self._direct_carrier_value(
                    self.q_direct_phase,
                    sequence_length,
                    dtype=target_dtype,
                )
                k_direct_phase = self._direct_carrier_value(
                    self.k_direct_phase,
                    sequence_length,
                    dtype=target_dtype,
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
                    amplitude_delta=q_amplitude_delta,
                    hyper_phase_delta=q_hyper_phase_delta,
                    frequency_delta=q_frequency_delta,
                    cartesian_real_delta=q_cartesian_real_delta,
                    cartesian_imag_delta=q_cartesian_imag_delta,
                    amplitude_raw_override=q_direct_amplitude,
                    phase_override=q_direct_phase,
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
                    amplitude_delta=k_amplitude_delta,
                    hyper_phase_delta=k_hyper_phase_delta,
                    frequency_delta=k_frequency_delta,
                    cartesian_real_delta=k_cartesian_real_delta,
                    cartesian_imag_delta=k_cartesian_imag_delta,
                    amplitude_raw_override=k_direct_amplitude,
                    phase_override=k_direct_phase,
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
                    amplitude_delta=q_amplitude_delta,
                    hyper_phase_delta=q_hyper_phase_delta,
                    frequency_delta=q_frequency_delta,
                    cartesian_real_delta=q_cartesian_real_delta,
                    cartesian_imag_delta=q_cartesian_imag_delta,
                    amplitude_raw_override=q_direct_amplitude,
                    phase_override=q_direct_phase,
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
                    amplitude_delta=k_amplitude_delta,
                    hyper_phase_delta=k_hyper_phase_delta,
                    frequency_delta=k_frequency_delta,
                    cartesian_real_delta=k_cartesian_real_delta,
                    cartesian_imag_delta=k_cartesian_imag_delta,
                    amplitude_raw_override=k_direct_amplitude,
                    phase_override=k_direct_phase,
                    branch="k",
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

        if (
            self.application == "rotary"
            and self.carrier_hypernetwork is not None
        ):
            q_output = q_output + q_hyper_phase_delta
            k_output = k_output + k_hyper_phase_delta

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
            q_amplitude_delta=q_amplitude_delta,
            k_amplitude_delta=k_amplitude_delta,
            q_hyper_phase_delta=q_hyper_phase_delta,
            k_hyper_phase_delta=k_hyper_phase_delta,
            q_frequency_delta=q_frequency_delta,
            k_frequency_delta=k_frequency_delta,
            q_cartesian_real_delta=q_cartesian_real_delta,
            k_cartesian_real_delta=k_cartesian_real_delta,
            q_cartesian_imag_delta=q_cartesian_imag_delta,
            k_cartesian_imag_delta=k_cartesian_imag_delta,
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
        ):
            if module is not None:
                module.reset_parameters()
        if self.phase_rotation_conditioner is not None:
            self.phase_rotation_conditioner.reset_output_parameters()
        if self.content_actuator is not None:
            self.content_actuator.reset_output_parameters()
        if self.carrier_hypernetwork is not None:
            self.carrier_hypernetwork.reset_output_parameters()
        for parameter in (
            self.direct_amplitude_raw,
            self.q_direct_amplitude_raw,
            self.k_direct_amplitude_raw,
            self.direct_phase,
            self.q_direct_phase,
            self.k_direct_phase,
        ):
            if parameter is not None:
                torch.nn.init.zeros_(parameter)

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
        for branch, phase_delta in (
            ("q", output.q_hyper_phase_delta),
            ("k", output.k_hyper_phase_delta),
        ):
            if phase_delta is not None:
                _delta_stats(f"hyper_phase_delta_{branch}", phase_delta)
        for branch, amplitude_delta in (
            ("q", output.q_amplitude_delta),
            ("k", output.k_amplitude_delta),
        ):
            if amplitude_delta is not None:
                _delta_stats(
                    f"hyper_amplitude_delta_{branch}",
                    amplitude_delta,
                )
                metrics[f"hyper_effective_amplitude_{branch}/max"] = (
                    self._amplitude(amplitude_delta)
                    .detach()
                    .float()
                    .max()
                    .item()
                )
        for branch, frequency_delta in (
            ("q", output.q_frequency_delta),
            ("k", output.k_frequency_delta),
        ):
            if frequency_delta is not None:
                _delta_stats(
                    f"hyper_frequency_delta_{branch}",
                    frequency_delta,
                )
                multiplier = 1.0 + frequency_delta.detach().float()
                metrics[f"hyper_frequency_multiplier_{branch}/min"] = (
                    multiplier.min().item()
                )
                metrics[f"hyper_frequency_multiplier_{branch}/max"] = (
                    multiplier.max().item()
                )
        for name, q_delta, k_delta in (
            (
                "cartesian_real",
                output.q_cartesian_real_delta,
                output.k_cartesian_real_delta,
            ),
            (
                "cartesian_imag",
                output.q_cartesian_imag_delta,
                output.k_cartesian_imag_delta,
            ),
        ):
            if q_delta is not None:
                _delta_stats(f"hyper_{name}_delta_q", q_delta)
            if k_delta is not None:
                _delta_stats(f"hyper_{name}_delta_k", k_delta)
        if self.carrier_hypernetwork is not None:
            for name in ("content", "position"):
                gain = getattr(
                    self.carrier_hypernetwork,
                    f"{name}_input_gain",
                )
                if gain is not None:
                    metrics[f"hyper_{name}_input_gain"] = gain.detach().item()
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
        content_dim=content_dim if content_dim is not None else head_dim,
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
            (".qk_position.output_weight", ".qk_position.phase_head.weight"),
            (".qk_position.output_bias", ".qk_position.phase_head.bias"),
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
    rope_frequency_total = 0
    rotary_clock_total = 0
    qk_preprojection_total = 0
    position_gain_total = 0
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
            qk_preprojection_total += _unique_numel(
                getattr(attn, "qk_preprojection", None)
            )
            position_gain_total += _unique_numel(
                getattr(attn, "position_gain", None)
            )
            rope_parameter = getattr(attn, "rope_log_frequency_delta", None)
            if rope_parameter is not None:
                rope_frequency_total += rope_parameter.numel()
            rope_frequency_total += _unique_numel(
                getattr(attn, "rope_frequency_controller", None)
            )
            rotary_clock_total += _unique_numel(
                getattr(attn, "rotary_clock", None)
            )
            logit_total += _unique_numel(getattr(attn, "logit_bias", None))
            content_total += _unique_numel(
                getattr(attn, "position_content", None)
            )
            attention_write_total += _unique_numel(
                getattr(attn, "position_write", None)
            )
    position_total = (
        qk_total
        + qk_preprojection_total
        + position_gain_total
        + rope_frequency_total
        + rotary_clock_total
        + logit_total
        + content_total
        + attention_write_total
        + residual_stream_total
    )
    return {
        "qk_position_params": qk_total,
        "qk_preprojection_params": qk_preprojection_total,
        "position_gain_params": position_gain_total,
        "rope_frequency_params": rope_frequency_total,
        "rotary_clock_params": rotary_clock_total,
        "logit_bias_params": logit_total,
        "position_content_params": content_total,
        "attention_write_params": attention_write_total,
        "residual_stream_params": residual_stream_total,
        "position_params": position_total,
    }
