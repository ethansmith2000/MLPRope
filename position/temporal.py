"""Small causal controllers shared by dynamic positional mechanisms.

The reference implementation deliberately uses ordinary PyTorch operators.
Short causal convolutions compile cleanly and are sufficient to test whether a
local temporal summary helps before introducing an associative-scan or custom
kernel backend for recurrent state.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


class CausalControlMapper(torch.nn.Module):
    """Map ``[B,L,D]`` inputs to causal controls ``[B,L,O]``.

    ``temporal='pointwise'`` supports a full linear map or a low-rank SiLU map.
    ``temporal='causal_conv'`` applies an identity-initialized depthwise causal
    convolution to the low-rank hidden features.  The final projection is
    always zero-initialized so consumers can define an exact no-op anchor.
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
            self._reset_temporal_parameters()
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
        return self.output(hidden)

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
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
        state: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Incremental equivalent of :meth:`forward` for one token."""
        if value.ndim == 2:
            value = value[:, None, :]
        if value.ndim != 3 or value.shape[1] != 1:
            raise ValueError("CausalControlMapper.step expects [B,D] or [B,1,D]")
        hidden = self._hidden(value)
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
