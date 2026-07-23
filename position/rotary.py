"""Split-half multiplicative RoPE helpers.

The cached ``rope_cos`` / ``rope_sin`` tensors have shape ``[seq, head_dim/2]`` and
pair the first half of each head with the second half. This is distinct from the
interleaved Fourier feature layout used by position mappers.
"""

from __future__ import annotations

import torch


def build_rope_cache(
    max_seq_len: int,
    head_dim: int,
    theta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if max_seq_len <= 0:
        raise ValueError("max_seq_len must be positive.")
    if head_dim <= 0 or head_dim % 2 != 0:
        raise ValueError("head_dim must be a positive even integer.")
    if theta <= 0:
        raise ValueError("theta must be positive.")
    half = head_dim // 2
    freqs = torch.arange(half, dtype=torch.float32)
    inv_freq = 1.0 / (theta ** (freqs / half))
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
) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


def apply_rotary(
    q: torch.Tensor,
    k: torch.Tensor,
    rope_sin: torch.Tensor,
    rope_cos: torch.Tensor,
    *,
    q_phase_delta: torch.Tensor | None = None,
    k_phase_delta: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply split-half RoPE, optionally with separate Q/K phase residuals."""
    seq_len = q.shape[-2]
    if seq_len > rope_sin.shape[0]:
        raise ValueError(
            f"Sequence length {seq_len} exceeds RoPE cache length {rope_sin.shape[0]}"
        )
    sin = rope_sin[:seq_len].to(dtype=q.dtype)[None, None, :, :]
    cos = rope_cos[:seq_len].to(dtype=q.dtype)[None, None, :, :]

    q_sin, q_cos = sin, cos
    k_sin, k_cos = sin, cos
    if q_phase_delta is not None:
        q_sin, q_cos = compose_phase(
            sin,
            cos,
            q_phase_delta.to(dtype=q.dtype)[None, :, :, :],
        )
    if k_phase_delta is not None:
        k_sin, k_cos = compose_phase(
            sin,
            cos,
            k_phase_delta.to(dtype=k.dtype)[None, :, :, :],
        )
    return rotate_half(q, q_sin, q_cos), rotate_half(k, k_sin, k_cos)
