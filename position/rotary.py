"""Split-half multiplicative RoPE helpers.

The cached ``rope_cos`` / ``rope_sin`` tensors have shape ``[seq, head_dim/2]``.
They pair the first half of each head with the second half. This is distinct from
the interleaved Fourier feature layout used by position mappers.
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


def build_rope_cache_from_frequencies(
    length: int,
    inverse_frequency: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build an fp32 RoPE cache from an explicit shared frequency bank."""
    if length <= 0:
        raise ValueError("length must be positive.")
    if inverse_frequency.ndim != 1 or inverse_frequency.numel() == 0:
        raise ValueError("inverse_frequency must be a non-empty vector")
    frequency = inverse_frequency.float()
    positions = torch.arange(length, device=frequency.device, dtype=torch.float32)
    angles = torch.outer(positions, frequency)
    return angles.sin(), angles.cos()


def rotate_half(
    x: torch.Tensor,
    sin: torch.Tensor,
    cos: torch.Tensor,
) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos
    return torch.cat([out1, out2], dim=-1)


def apply_rotary(
    q: torch.Tensor,
    k: torch.Tensor,
    rope_sin: torch.Tensor,
    rope_cos: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply fixed split-half RoPE to Q and K."""
    seq_len = q.shape[-2]
    if rope_sin.ndim != 2 or rope_cos.shape != rope_sin.shape:
        raise ValueError("RoPE tables must be matching [L,D/2] tensors")
    cache_length = rope_sin.shape[0]
    if seq_len > cache_length:
        raise ValueError(
            f"Sequence length {seq_len} exceeds RoPE cache length {cache_length}"
        )
    # Build angles in fp32, then cast the completed coefficients. Constructing
    # large-position angles in bf16/fp16 loses too much precision.
    sin = rope_sin[:seq_len].float()[None, None, :, :]
    cos = rope_cos[:seq_len].float()[None, None, :, :]
    return (
        rotate_half(q, sin.to(dtype=q.dtype), cos.to(dtype=q.dtype)),
        rotate_half(k, sin.to(dtype=k.dtype), cos.to(dtype=k.dtype)),
    )
