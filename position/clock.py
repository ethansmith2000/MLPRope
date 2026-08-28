"""Spectrally locked causal clocks for dynamic RoPE."""

from __future__ import annotations

import copy
import math
from typing import NamedTuple

import torch

from position.temporal import CausalControlMapper
from position.precision import PreserveFP32BuffersMixin


ROTARY_CLOCK_DEFAULTS = {
    "enabled": False,
    "source": "normalized_residual",
    "head_coupling": "per_head",
    "mapper": "low_rank_silu",
    "rank": 32,
    "temporal": "pointwise",
    "kernel_size": 3,
    "speed_bound": 0.25,
}


def _exclusive_associative_sum(values: torch.Tensor) -> torch.Tensor:
    """Exclusive prefix sum using a compile-friendly parallel scan.

    This is the Hillis--Steele associative scan specialized to addition.  It
    takes ``ceil(log2(length))`` shift-add stages and avoids PyTorch
    Inductor's ``SplitScan`` lowering, which is broken for the length-1024
    CUDA shape used by this project.  The zero-clock anchor remains exact:
    scanning an all-ones fp32 speed produces exactly representable integers.
    """
    if values.ndim != 3:
        raise ValueError("clock scan expects [batch, sequence, groups]")
    inclusive = values
    offset = 1
    length = values.shape[1]
    while offset < length:
        shifted = torch.nn.functional.pad(
            inclusive[:, :-offset],
            (0, 0, offset, 0),
        )
        inclusive = inclusive + shifted
        offset *= 2
    return inclusive - values


def normalize_rotary_clock_config(config: dict | None) -> dict:
    if config is None:
        config = {}
    if not isinstance(config, dict):
        raise TypeError("rotary_clock must be an object")
    unknown = set(config) - set(ROTARY_CLOCK_DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown rotary_clock keys: {sorted(unknown)}")
    normalized = copy.deepcopy(ROTARY_CLOCK_DEFAULTS)
    normalized.update(config)
    if not isinstance(normalized["enabled"], bool):
        raise TypeError("rotary_clock.enabled must be a boolean")
    if normalized["source"] != "normalized_residual":
        raise ValueError("rotary_clock.source must be 'normalized_residual'")
    if normalized["head_coupling"] not in {"shared", "per_head"}:
        raise ValueError("rotary_clock.head_coupling must be 'shared' or 'per_head'")
    if normalized["mapper"] not in {"linear", "low_rank_silu"}:
        raise ValueError("rotary_clock.mapper must be 'linear' or 'low_rank_silu'")
    if normalized["temporal"] not in {"pointwise", "causal_conv"}:
        raise ValueError(
            "rotary_clock.temporal must be 'pointwise' or 'causal_conv'"
        )
    rank = normalized["rank"]
    if isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0:
        raise ValueError("rotary_clock.rank must be a positive integer")
    kernel_size = normalized["kernel_size"]
    if (
        isinstance(kernel_size, bool)
        or not isinstance(kernel_size, int)
        or kernel_size <= 0
    ):
        raise ValueError("rotary_clock.kernel_size must be a positive integer")
    if normalized["temporal"] == "pointwise" and kernel_size != 1:
        # Canonicalize the inert value so config hashes do not distinguish it.
        normalized["kernel_size"] = 1
    if (
        normalized["temporal"] == "causal_conv"
        and normalized["mapper"] != "low_rank_silu"
    ):
        raise ValueError(
            "rotary_clock temporal='causal_conv' requires mapper='low_rank_silu'"
        )
    if normalized["temporal"] == "causal_conv" and kernel_size < 2:
        raise ValueError(
            "rotary_clock temporal='causal_conv' requires kernel_size >= 2"
        )
    speed_bound = normalized["speed_bound"]
    if isinstance(speed_bound, bool) or not isinstance(speed_bound, (int, float)):
        raise TypeError("rotary_clock.speed_bound must be a number")
    normalized["speed_bound"] = float(speed_bound)
    if not math.isfinite(normalized["speed_bound"]) or not (
        0 < normalized["speed_bound"] < 1
    ):
        raise ValueError("rotary_clock.speed_bound must lie strictly inside (0, 1)")
    return normalized


class RotaryClockState(NamedTuple):
    controller: torch.Tensor | None
    clock: torch.Tensor
    reference: torch.Tensor


class RotaryClockController(PreserveFP32BuffersMixin, torch.nn.Module):
    """Produce one positive causal coordinate per head and token.

    The controller emits bounded local speed
    ``s_t = 1 + rho*tanh(raw_t)``.  An exclusive cumulative sum gives
    ``tau_0=0`` and ``tau_t=sum_{j<t}s_j``.  The same ``tau`` rotates Q and K,
    and every frequency plane within a head uses ``omega_i*tau``.
    """

    _fp32_buffer_names = ("inverse_frequency",)

    def __init__(
        self,
        *,
        model_dim: int,
        heads: int,
        pair_dim: int,
        inverse_frequency: torch.Tensor,
        config: dict,
    ) -> None:
        super().__init__()
        self.model_dim = model_dim
        self.heads = heads
        self.pair_dim = pair_dim
        self.config = copy.deepcopy(config)
        self.groups = heads if config["head_coupling"] == "per_head" else 1
        omega = inverse_frequency.detach().float().reshape(-1)
        if omega.numel() != pair_dim:
            raise ValueError("Rotary clock inverse-frequency width mismatch")
        self.register_buffer("inverse_frequency", omega, persistent=False)
        self.controller = CausalControlMapper(
            input_dim=model_dim,
            output_dim=self.groups,
            mapper=config["mapper"],
            rank=int(config["rank"]),
            temporal=config["temporal"],
            kernel_size=int(config["kernel_size"]),
        )

    def reset_output_parameters(self) -> None:
        self.controller.reset_output_parameters()

    def raw_output(self, normalized_residual: torch.Tensor) -> torch.Tensor:
        return self.controller(normalized_residual).float()

    def _expand_heads(self, values: torch.Tensor) -> torch.Tensor:
        if self.groups == 1:
            return values.expand(-1, -1, self.heads)
        return values

    def speed(self, normalized_residual: torch.Tensor) -> torch.Tensor:
        raw = self.raw_output(normalized_residual)
        return 1.0 + float(self.config["speed_bound"]) * raw.tanh()

    def coordinates(self, normalized_residual: torch.Tensor) -> torch.Tensor:
        speed = self.speed(normalized_residual)
        # Exclusive cumsum: the content at token t controls the interval from t
        # to t+1.  This is causal and gives tau_t=t at the zero-output anchor.
        clock = _exclusive_associative_sum(speed)
        return self._expand_heads(clock)

    def phase_delta(self, normalized_residual: torch.Tensor) -> torch.Tensor:
        clock = self.coordinates(normalized_residual)
        length = clock.shape[1]
        reference = torch.arange(
            length,
            device=clock.device,
            dtype=torch.float32,
        )[None, :, None]
        displacement = (clock - reference).permute(0, 2, 1)
        return displacement[..., None] * self.inverse_frequency[None, None, None, :]

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> RotaryClockState:
        controller_state = self.controller.initial_state(
            batch_size,
            device=device,
            dtype=dtype,
        )
        clock = torch.zeros(batch_size, self.groups, device=device, dtype=torch.float32)
        reference = torch.zeros_like(clock)
        return RotaryClockState(controller_state, clock, reference)

    def step_phase_delta(
        self,
        normalized_residual: torch.Tensor,
        state: RotaryClockState,
    ) -> tuple[torch.Tensor, RotaryClockState]:
        raw, controller_state = self.controller.step(
            normalized_residual,
            state.controller,
        )
        raw = raw.float()
        displacement = state.clock - state.reference
        if self.groups == 1:
            displacement = displacement.expand(-1, self.heads)
        phase = displacement[:, :, None, None] * self.inverse_frequency[
            None, None, None, :
        ]
        speed = 1.0 + float(self.config["speed_bound"]) * raw.tanh()
        new_state = RotaryClockState(
            controller_state,
            state.clock + speed,
            state.reference + 1.0,
        )
        return phase, new_state

    @torch.no_grad()
    def diagnostics(self, normalized_residual: torch.Tensor) -> dict[str, float]:
        raw = self.raw_output(normalized_residual).detach().float()
        speed = 1.0 + float(self.config["speed_bound"]) * raw.tanh()
        clock = self._expand_heads(_exclusive_associative_sum(speed))
        positions = torch.arange(
            clock.shape[1], device=clock.device, dtype=torch.float32
        )[None, :, None]
        drift = clock - positions
        phase = drift.permute(0, 2, 1)[..., None] * self.inverse_frequency[
            None, None, None, :
        ]
        return {
            "raw_mean": raw.mean().item(),
            "raw_rms": raw.square().mean().sqrt().item(),
            "raw_abs_max": raw.abs().max().item(),
            "speed_mean": speed.mean().item(),
            "speed_min": speed.min().item(),
            "speed_max": speed.max().item(),
            "clock_drift_rms": drift.square().mean().sqrt().item(),
            "clock_drift_abs_max": drift.abs().max().item(),
            "clock_final_drift_rms": drift[:, -1].square().mean().sqrt().item(),
            "phase_delta_rms": phase.square().mean().sqrt().item(),
            "phase_delta_abs_max": phase.abs().max().item(),
        }
