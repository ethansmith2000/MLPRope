"""Small causal controllers shared by dynamic positional mechanisms.

The reference implementations deliberately use ordinary PyTorch operators.
Both the short convolution and the EMA have incremental forms; the full EMA
uses a compile-friendly associative affine scan, so it does not serialize the
sequence dimension in eager Python.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


class CausalEMAState(tuple):
    """Incremental numerator and normalization mass for :class:`CausalEMA`."""

    __slots__ = ()

    def __new__(cls, numerator: torch.Tensor, mass: torch.Tensor):
        return tuple.__new__(cls, (numerator, mass))

    @property
    def numerator(self) -> torch.Tensor:
        return self[0]

    @property
    def mass(self) -> torch.Tensor:
        return self[1]


def _inclusive_affine_scan(
    decay: torch.Tensor,
    update: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compose ``z <- decay*z + update`` transforms along dimension one.

    ``decay`` and ``update`` must have shape ``[B,L,D]``.  The returned pair
    describes the inclusive prefix transform at every token.  This is the
    affine analogue of a Hillis--Steele sum scan and lowers through
    ``torch.compile`` without relying on the currently fragile SplitScan path.
    """
    if decay.ndim != 3 or update.shape != decay.shape:
        raise ValueError("affine scan expects matching [batch, sequence, dim]")
    prefix_decay = decay
    prefix_update = update
    offset = 1
    length = update.shape[1]
    while offset < length:
        earlier_decay = F.pad(
            prefix_decay[:, :-offset],
            (0, 0, offset, 0),
            value=1.0,
        )
        earlier_update = F.pad(
            prefix_update[:, :-offset],
            (0, 0, offset, 0),
        )
        composed_update = prefix_update + prefix_decay * earlier_update
        prefix_decay = prefix_decay * earlier_decay
        prefix_update = composed_update
        offset *= 2
    return prefix_decay, prefix_update


class CausalEMA(torch.nn.Module):
    """Learned bias-corrected causal exponential average.

    A decay may be shared globally, assigned per feature/group, or assigned to
    every group-feature pair. For the applicable coefficient ``beta``:

    ``n_t = beta*n_{t-1} + (1-beta)*x_t``.

    Dividing by the accumulated mass removes the zero-state startup transient.
    In particular, a sequence that is constant from its first token remains
    exactly constant rather than acquiring an unintended absolute-position
    ramp.  The full-sequence implementation is an associative affine scan and
    :meth:`step` is its streaming equivalent.
    """

    def __init__(
        self,
        feature_dim: int,
        *,
        decay_init: float = 0.9,
        groups: int = 1,
        decay_coupling: str = "per_dim",
    ) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.groups = int(groups)
        self.decay_coupling = decay_coupling
        if self.feature_dim <= 0:
            raise ValueError("CausalEMA.feature_dim must be positive")
        if self.groups <= 0:
            raise ValueError("CausalEMA.groups must be positive")
        parameter_shapes = {
            "scalar": (1,),
            "per_dim": (self.feature_dim,),
            "per_group": (self.groups,),
            "per_group_dim": (self.groups, self.feature_dim),
        }
        if decay_coupling not in parameter_shapes:
            raise ValueError(
                "CausalEMA.decay_coupling must be 'scalar', 'per_dim', "
                "'per_group', or 'per_group_dim'"
            )
        if decay_coupling.startswith("per_group") and self.groups == 1:
            raise ValueError("per-group CausalEMA decay requires groups > 1")
        decay_init = float(decay_init)
        if not 0.0 < decay_init < 1.0:
            raise ValueError("CausalEMA.decay_init must lie strictly inside (0, 1)")
        logit = torch.logit(torch.tensor(decay_init, dtype=torch.float32))
        self.decay_logit = torch.nn.Parameter(
            logit.expand(parameter_shapes[decay_coupling]).clone()
        )

    def decay(self) -> torch.Tensor:
        return torch.sigmoid(self.decay_logit.float())

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        grouped = values.ndim == 4
        if grouped:
            if values.shape[1] != self.groups or values.shape[-1] != self.feature_dim:
                raise ValueError(
                    "Grouped CausalEMA expects [batch, groups, sequence, feature_dim]"
                )
        elif (
            values.ndim != 3
            or values.shape[-1] != self.feature_dim
            or self.groups != 1
        ):
            raise ValueError("CausalEMA expects [batch, sequence, feature_dim]")
        values_fp32 = values.float()
        if grouped:
            batch, groups, length, width = values_fp32.shape
            if self.decay_coupling == "scalar":
                decay = self.decay().view(1, 1, 1, 1)
            elif self.decay_coupling == "per_dim":
                decay = self.decay().view(1, 1, 1, width)
            elif self.decay_coupling == "per_group":
                decay = self.decay().view(1, groups, 1, 1)
            else:
                decay = self.decay().view(1, groups, 1, width)
            decay = decay.expand_as(values_fp32).reshape(
                batch * groups, length, width
            )
            scan_values = values_fp32.reshape(batch * groups, length, width)
        else:
            decay_values = self.decay()
            if self.decay_coupling == "scalar":
                decay_values = decay_values.expand(self.feature_dim)
            decay = decay_values[None, None, :].expand_as(values_fp32)
            scan_values = values_fp32
        prefix_decay, numerator = _inclusive_affine_scan(
            decay,
            (1.0 - decay) * scan_values,
        )
        mass = (1.0 - prefix_decay).clamp_min(torch.finfo(torch.float32).eps)
        output = numerator / mass
        if grouped:
            output = output.reshape(batch, groups, length, width)
        return output.to(dtype=values.dtype)

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device | str,
    ) -> CausalEMAState:
        if self.groups != 1:
            raise ValueError(
                "Streaming grouped CausalEMA state is not implemented; grouped "
                "carrier memory uses the full associative scan"
            )
        shape = (batch_size, self.feature_dim)
        return CausalEMAState(
            torch.zeros(shape, device=device, dtype=torch.float32),
            torch.zeros(shape, device=device, dtype=torch.float32),
        )

    def step(
        self,
        value: torch.Tensor,
        state: CausalEMAState | None,
    ) -> tuple[torch.Tensor, CausalEMAState]:
        """Apply one streaming update to ``[B,D]`` or ``[B,1,D]`` input."""
        if self.groups != 1:
            raise ValueError(
                "CausalEMA.step is only supported for a single shared group"
            )
        squeeze = value.ndim == 2
        if squeeze:
            value = value[:, None, :]
        if (
            value.ndim != 3
            or value.shape[1] != 1
            or value.shape[2] != self.feature_dim
        ):
            raise ValueError("CausalEMA.step expects [B,D] or [B,1,D]")
        if state is None:
            state = self.initial_state(value.shape[0], device=value.device)
        expected = (value.shape[0], self.feature_dim)
        if state.numerator.shape != expected or state.mass.shape != expected:
            raise ValueError(
                f"CausalEMA state tensors must have shape {expected}"
            )
        decay = self.decay()
        if self.decay_coupling == "scalar":
            decay = decay.expand(self.feature_dim)
        decay = decay[None, :]
        keep = 1.0 - decay
        numerator = decay * state.numerator + keep * value[:, 0].float()
        mass = decay * state.mass + keep
        output = numerator / mass.clamp_min(torch.finfo(torch.float32).eps)
        if not squeeze:
            output = output[:, None, :]
        return output.to(dtype=value.dtype), CausalEMAState(numerator, mass)

    @torch.no_grad()
    def diagnostics(self) -> dict[str, float]:
        decay = self.decay().detach()
        return {
            "ema_decay_mean": decay.mean().item(),
            "ema_decay_min": decay.min().item(),
            "ema_decay_max": decay.max().item(),
            "ema_effective_window_mean": (1.0 / (1.0 - decay)).mean().item(),
        }


class CausalControlMapper(torch.nn.Module):
    """Map ``[B,L,D]`` inputs to causal controls ``[B,L,O]``.

    ``temporal='pointwise'`` supports a full linear map or a low-rank SiLU map.
    ``temporal='causal_conv'`` applies an identity-initialized depthwise causal
    convolution to the low-rank hidden features. ``temporal='ema'`` instead
    applies a learned bias-corrected causal EMA in that compact latent space.
    The final projection is always zero-initialized so consumers can define an
    exact no-op anchor.
    """

    def __init__(
        self,
        *,
        input_dim: int,
        output_dim: int,
        mapper: str,
        rank: int,
        temporal: str,
        kernel_size: int,
        ema_decay_init: float = 0.9,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.mapper = mapper
        self.rank = int(rank)
        self.temporal = temporal
        self.kernel_size = int(kernel_size)
        if mapper == "linear":
            self.down = None
            self.output = torch.nn.Linear(input_dim, output_dim, bias=True)
        elif mapper == "low_rank_silu":
            self.down = torch.nn.Linear(input_dim, rank, bias=True)
            self.output = torch.nn.Linear(rank, output_dim, bias=True)
        else:
            raise ValueError(f"Unknown causal-control mapper: {mapper!r}")
        if temporal == "pointwise":
            self.temporal_conv = None
            self.temporal_ema = None
        elif temporal == "causal_conv":
            if mapper != "low_rank_silu":
                raise ValueError(
                    "causal_conv requires mapper='low_rank_silu' so temporal "
                    "mixing occurs in the compact latent space"
                )
            self.temporal_conv = torch.nn.Conv1d(
                rank,
                rank,
                kernel_size,
                groups=rank,
                bias=True,
                padding=0,
            )
            self.temporal_ema = None
            self._reset_temporal_parameters()
        elif temporal == "ema":
            if mapper != "low_rank_silu":
                raise ValueError(
                    "ema requires mapper='low_rank_silu' so temporal mixing "
                    "occurs in the compact latent space"
                )
            self.temporal_conv = None
            self.temporal_ema = CausalEMA(rank, decay_init=ema_decay_init)
        else:
            raise ValueError(f"Unknown causal-control temporal mode: {temporal!r}")
        self.reset_output_parameters()

    def _reset_temporal_parameters(self) -> None:
        if self.temporal_conv is None:
            return
        # Conv1d is cross-correlation.  With left padding, the last coefficient
        # multiplies the current token, so this starts as an exact identity.
        with torch.no_grad():
            self.temporal_conv.weight.zero_()
            self.temporal_conv.weight[:, 0, -1] = 1.0
            self.temporal_conv.bias.zero_()

    def reset_output_parameters(self) -> None:
        """Restore the exact zero-control anchor."""
        torch.nn.init.zeros_(self.output.weight)
        if self.output.bias is not None:
            torch.nn.init.zeros_(self.output.bias)

    def _hidden(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 3 or values.shape[-1] != self.input_dim:
            raise ValueError(
                "CausalControlMapper expects [batch, sequence, input_dim]"
            )
        if self.down is None:
            return values
        return F.silu(self.down(values))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        hidden = self._hidden(values)
        if self.temporal_conv is not None:
            hidden_channels = hidden.transpose(1, 2)
            hidden_channels = F.pad(
                hidden_channels,
                (self.kernel_size - 1, 0),
            )
            hidden = self.temporal_conv(hidden_channels).transpose(1, 2)
        elif self.temporal_ema is not None:
            hidden = self.temporal_ema(hidden)
        return self.output(hidden)

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> torch.Tensor | CausalEMAState | None:
        if self.temporal_ema is not None:
            return self.temporal_ema.initial_state(batch_size, device=device)
        if self.temporal_conv is None:
            return None
        return torch.zeros(
            batch_size,
            self.rank,
            self.kernel_size - 1,
            device=device,
            dtype=dtype,
        )

    def step(
        self,
        value: torch.Tensor,
        state: torch.Tensor | CausalEMAState | None,
    ) -> tuple[torch.Tensor, torch.Tensor | CausalEMAState | None]:
        """Incremental equivalent of :meth:`forward` for one token."""
        if value.ndim == 2:
            value = value[:, None, :]
        if value.ndim != 3 or value.shape[1] != 1:
            raise ValueError("CausalControlMapper.step expects [B,D] or [B,1,D]")
        hidden = self._hidden(value)
        if self.temporal_ema is not None:
            if state is not None and not isinstance(state, CausalEMAState):
                raise ValueError("CausalControlMapper EMA state has the wrong type")
            mixed, new_state = self.temporal_ema.step(hidden, state)
            return self.output(mixed).squeeze(1), new_state
        if self.temporal_conv is None:
            return self.output(hidden).squeeze(1), None
        hidden_current = hidden.transpose(1, 2)
        if state is None:
            state = self.initial_state(
                value.shape[0],
                device=value.device,
                dtype=hidden.dtype,
            )
        expected = (value.shape[0], self.rank, self.kernel_size - 1)
        if state is None or tuple(state.shape) != expected:
            raise ValueError(
                f"CausalControlMapper state must have shape {expected}, got "
                f"{None if state is None else tuple(state.shape)}"
            )
        window = torch.cat((state, hidden_current), dim=-1)
        mixed = self.temporal_conv(window).transpose(1, 2)
        new_state = window[:, :, 1:]
        return self.output(mixed).squeeze(1), new_state

    @torch.no_grad()
    def temporal_diagnostics(self) -> dict[str, float]:
        if self.temporal_ema is None:
            return {}
        return self.temporal_ema.diagnostics()
