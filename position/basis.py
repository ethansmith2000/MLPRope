"""Frozen Fourier position bases with an explicit pair layout."""

from __future__ import annotations

import torch


def interleaved_fourier_basis(
    extent: int,
    basis_dim: int,
    theta: float,
) -> torch.Tensor:
    """Return interleaved cosine/sine features for indices ``0 .. extent-1``.

    Layout is ``[cos_0, sin_0, cos_1, sin_1, ...]`` with ``basis_dim`` even.
    This is the *feature* layout used by mappers. It is distinct from the
    split-half Q/K rotary layout used by multiplicative RoPE, which pairs the
    first and second halves of each head vector.
    """
    if extent <= 0:
        raise ValueError("extent must be positive.")
    if basis_dim <= 0 or basis_dim % 2 != 0:
        raise ValueError("basis_dim must be a positive even integer.")
    if theta <= 0:
        raise ValueError("theta must be positive.")
    half = basis_dim // 2
    frequencies = torch.arange(half, dtype=torch.float32)
    inverse_frequencies = 1.0 / (theta ** (frequencies / half))
    distances = torch.arange(extent, dtype=torch.float32)
    angles = torch.outer(distances, inverse_frequencies)
    return torch.stack((angles.cos(), angles.sin()), dim=-1).flatten(-2)


class FrozenFourierBasis(torch.nn.Module):
    """Cached interleaved Fourier features for absolute or relative indices."""

    def __init__(self, extent: int, basis_dim: int, theta: float):
        super().__init__()
        self.extent = int(extent)
        self.basis_dim = int(basis_dim)
        self.theta = float(theta)
        self.register_buffer(
            "basis",
            interleaved_fourier_basis(self.extent, self.basis_dim, self.theta),
            persistent=False,
        )

    def forward(
        self,
        length: int,
        *,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        if length <= 0:
            raise ValueError("length must be positive.")
        if length > self.extent:
            raise ValueError(
                f"Requested length {length} exceeds basis extent {self.extent}."
            )
        features = self.basis[:length]
        if dtype is not None:
            features = features.to(dtype=dtype)
        return features
