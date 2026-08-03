"""Split-half multiplicative RoPE helpers.

The cached ``rope_cos`` / ``rope_sin`` tensors have shape ``[seq, head_dim/2]``
for a shared schedule or ``[heads, seq, head_dim/2]`` for learned per-head
frequencies. They pair the first half of each head with the second half. This is
distinct from the interleaved Fourier feature layout used by position mappers.
"""

from __future__ import annotations

import torch


def build_rope_frequencies(
    head_dim: int,
    theta: float,
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return the canonical fp32 inverse-frequency schedule."""
    if head_dim <= 0 or head_dim % 2 != 0:
        raise ValueError("head_dim must be a positive even integer.")
    if theta <= 0:
        raise ValueError("theta must be positive.")
    half = head_dim // 2
    frequency_index = torch.arange(half, device=device, dtype=torch.float32)
    return 1.0 / (theta ** (frequency_index / half))


def build_rope_cache(
    max_seq_len: int,
    head_dim: int,
    theta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if max_seq_len <= 0:
        raise ValueError("max_seq_len must be positive.")
    inv_freq = build_rope_frequencies(head_dim, theta)
    positions = torch.arange(max_seq_len, dtype=torch.float32)
    angles = torch.outer(positions, inv_freq)
    return angles.sin(), angles.cos()


def compose_phase(
    sin: torch.Tensor,
    cos: torch.Tensor,
    phase_delta: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compose ``R(theta + delta)`` from cached ``sin/cos(theta)`` and ``delta``."""
    delta = phase_delta.to(dtype=sin.dtype)
    delta_sin = delta.sin()
    delta_cos = delta.cos()
    return (
        sin * delta_cos + cos * delta_sin,
        cos * delta_cos - sin * delta_sin,
    )


def rotate_half(
    x: torch.Tensor,
    sin: torch.Tensor,
    cos: torch.Tensor,
    scale: torch.Tensor | None = None,
) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos
    if scale is not None:
        out1 = out1 * scale
        out2 = out2 * scale
    return torch.cat([out1, out2], dim=-1)


def apply_rotary(
    q: torch.Tensor,
    k: torch.Tensor,
    rope_sin: torch.Tensor,
    rope_cos: torch.Tensor,
    *,
    q_phase_delta: torch.Tensor | None = None,
    k_phase_delta: torch.Tensor | None = None,
    q_scale: torch.Tensor | None = None,
    k_scale: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply split-half RoPE with optional separate phase and radial scales."""
    seq_len = q.shape[-2]
    if rope_sin.ndim not in {2, 3} or rope_cos.shape != rope_sin.shape:
        raise ValueError(
            "RoPE tables must be matching [L,D/2] or [H,L,D/2] tensors"
        )
    cache_length = rope_sin.shape[-2]
    if seq_len > cache_length:
        raise ValueError(
            f"Sequence length {seq_len} exceeds RoPE cache length {cache_length}"
        )
    # Keep base angles and dynamic phase composition in fp32.  Casting the
    # completed coefficients is safe; constructing or composing angles in
    # bf16/fp16 is not, because large integer positions lose precision.
    if rope_sin.ndim == 2:
        sin = rope_sin[:seq_len].float()[None, None, :, :]
        cos = rope_cos[:seq_len].float()[None, None, :, :]
    else:
        if rope_sin.shape[0] not in {1, q.shape[-3]}:
            raise ValueError(
                "Per-head RoPE tables must have one head or match Q/K heads"
            )
        sin = rope_sin[:, :seq_len].float()[None, :, :, :]
        cos = rope_cos[:, :seq_len].float()[None, :, :, :]

    q_sin, q_cos = sin, cos
    k_sin, k_cos = sin, cos

    def batch_shape(value: torch.Tensor, *, dtype: torch.dtype) -> torch.Tensor:
        value = value.to(dtype=dtype)
        if value.ndim == 3:
            return value[None, :, :, :]
        if value.ndim == 4:
            return value
        raise ValueError(
            "Rotary phase/scale tensors must be [H,L,D/2] or [B,H,L,D/2]"
        )

    if q_phase_delta is not None:
        q_sin, q_cos = compose_phase(
            sin,
            cos,
            batch_shape(q_phase_delta, dtype=torch.float32),
        )
    if k_phase_delta is not None:
        k_sin, k_cos = compose_phase(
            sin,
            cos,
            batch_shape(k_phase_delta, dtype=torch.float32),
        )
    q_sin = q_sin.to(dtype=q.dtype)
    q_cos = q_cos.to(dtype=q.dtype)
    k_sin = k_sin.to(dtype=k.dtype)
    k_cos = k_cos.to(dtype=k.dtype)
    q_scale_batched = (
        None if q_scale is None else batch_shape(q_scale, dtype=q.dtype)
    )
    k_scale_batched = (
        None if k_scale is None else batch_shape(k_scale, dtype=k.dtype)
    )
    return (
        rotate_half(q, q_sin, q_cos, q_scale_batched),
        rotate_half(k, k_sin, k_cos, k_scale_batched),
    )
