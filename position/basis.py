"""Position bases with explicit Fourier-pair and scalar-feature layouts."""

from __future__ import annotations

from typing import Literal

import torch


BasisKind = Literal[
    "frozen_fourier",
    "learned_temperature_fourier",
    "learned_frequency_fourier",
]
ScalarFeature = Literal["position", "normalized_position", "log_position"]
SCALAR_FEATURES = {"position", "normalized_position", "log_position"}


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

    kind: BasisKind = "frozen_fourier"

    def __init__(
        self,
        extent: int,
        basis_dim: int,
        theta: float,
        scalars: list[str] | tuple[str, ...] = (),
        normalization_extent: int | None = None,
    ):
        super().__init__()
        _validate_basis_args(extent, basis_dim, theta, scalars)
        self.extent = int(extent)
        self.basis_dim = int(basis_dim)
        self.theta = float(theta)
        self.scalars = tuple(scalars)
        self.normalization_extent = int(normalization_extent or extent)
        self.output_dim = self.basis_dim + len(self.scalars)
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
        features = _append_scalar_features(
            self.basis[:length],
            length=length,
            normalization_extent=self.normalization_extent,
            scalars=self.scalars,
        )
        if dtype is not None:
            features = features.to(dtype=dtype)
        return features

    def frequencies(self, *, dtype: torch.dtype | None = None) -> torch.Tensor:
        half = self.basis_dim // 2
        index = torch.arange(half, device=self.basis.device, dtype=torch.float32)
        frequencies = 1.0 / (self.theta ** (index / half))
        return frequencies if dtype is None else frequencies.to(dtype=dtype)


class LearnedTemperatureFourierBasis(torch.nn.Module):
    """RoPE-frequency basis with one learned positive global temperature.

    ``log_temperature=0`` exactly reproduces the frozen basis. A positive value
    increases every angular frequency by the same multiplicative factor.
    """

    kind: BasisKind = "learned_temperature_fourier"

    def __init__(
        self,
        extent: int,
        basis_dim: int,
        theta: float,
        scalars: list[str] | tuple[str, ...] = (),
        normalization_extent: int | None = None,
    ):
        super().__init__()
        _validate_basis_args(extent, basis_dim, theta, scalars)
        self.extent = int(extent)
        self.basis_dim = int(basis_dim)
        self.theta = float(theta)
        self.scalars = tuple(scalars)
        self.normalization_extent = int(normalization_extent or extent)
        self.output_dim = self.basis_dim + len(self.scalars)
        self.log_temperature = torch.nn.Parameter(torch.zeros(()))

    def frequencies(self, *, dtype: torch.dtype | None = None) -> torch.Tensor:
        half = self.basis_dim // 2
        index = torch.arange(
            half,
            device=self.log_temperature.device,
            dtype=torch.float32,
        )
        base = 1.0 / (self.theta ** (index / half))
        frequencies = base * self.log_temperature.float().exp()
        return frequencies if dtype is None else frequencies.to(dtype=dtype)

    def forward(
        self,
        length: int,
        *,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        _validate_length(length, self.extent)
        positions = torch.arange(
            length,
            device=self.log_temperature.device,
            dtype=torch.float32,
        )
        angles = torch.outer(positions, self.frequencies())
        features = torch.stack((angles.cos(), angles.sin()), dim=-1).flatten(-2)
        features = _append_scalar_features(
            features,
            length=length,
            normalization_extent=self.normalization_extent,
            scalars=self.scalars,
        )
        return features if dtype is None else features.to(dtype=dtype)


class LearnedFrequencyFourierBasis(torch.nn.Module):
    """Independently learned positive frequencies, initialized to RoPE."""

    kind: BasisKind = "learned_frequency_fourier"

    def __init__(
        self,
        extent: int,
        basis_dim: int,
        theta: float,
        scalars: list[str] | tuple[str, ...] = (),
        normalization_extent: int | None = None,
    ):
        super().__init__()
        _validate_basis_args(extent, basis_dim, theta, scalars)
        self.extent = int(extent)
        self.basis_dim = int(basis_dim)
        self.theta = float(theta)
        self.scalars = tuple(scalars)
        self.normalization_extent = int(normalization_extent or extent)
        self.output_dim = self.basis_dim + len(self.scalars)
        self.log_frequency_residual = torch.nn.Parameter(
            torch.zeros(self.basis_dim // 2)
        )

    def frequencies(self, *, dtype: torch.dtype | None = None) -> torch.Tensor:
        half = self.basis_dim // 2
        index = torch.arange(
            half,
            device=self.log_frequency_residual.device,
            dtype=torch.float32,
        )
        base = 1.0 / (self.theta ** (index / half))
        frequencies = base * self.log_frequency_residual.float().exp()
        return frequencies if dtype is None else frequencies.to(dtype=dtype)

    def forward(
        self,
        length: int,
        *,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        _validate_length(length, self.extent)
        positions = torch.arange(
            length,
            device=self.log_frequency_residual.device,
            dtype=torch.float32,
        )
        angles = torch.outer(positions, self.frequencies())
        features = torch.stack((angles.cos(), angles.sin()), dim=-1).flatten(-2)
        features = _append_scalar_features(
            features,
            length=length,
            normalization_extent=self.normalization_extent,
            scalars=self.scalars,
        )
        return features if dtype is None else features.to(dtype=dtype)


def _validate_length(length: int, extent: int) -> None:
    if length <= 0:
        raise ValueError("length must be positive.")
    if length > extent:
        raise ValueError(f"Requested length {length} exceeds basis extent {extent}.")


def _validate_basis_args(
    extent: int,
    basis_dim: int,
    theta: float,
    scalars: list[str] | tuple[str, ...],
) -> None:
    # Reuse the numerical helper's validation.
    interleaved_fourier_basis(int(extent), int(basis_dim), float(theta))
    unknown = set(scalars) - SCALAR_FEATURES
    if unknown:
        raise ValueError(
            f"Unknown scalar position features {sorted(unknown)}; "
            f"expected a subset of {sorted(SCALAR_FEATURES)}."
        )
    if len(set(scalars)) != len(scalars):
        raise ValueError("Scalar position features must not contain duplicates.")


def _append_scalar_features(
    fourier: torch.Tensor,
    *,
    length: int,
    normalization_extent: int,
    scalars: tuple[str, ...],
) -> torch.Tensor:
    if not scalars:
        return fourier
    positions = torch.arange(
        length,
        device=fourier.device,
        dtype=torch.float32,
    )
    denominator = float(max(normalization_extent - 1, 1))
    values: list[torch.Tensor] = []
    for scalar in scalars:
        if scalar == "position":
            values.append(positions)
        elif scalar == "normalized_position":
            values.append(positions / denominator)
        elif scalar == "log_position":
            values.append(
                positions.log1p()
                / torch.tensor(
                    float(max(normalization_extent, 2)),
                    device=fourier.device,
                ).log()
            )
        else:  # validated at construction
            raise AssertionError(f"Unhandled scalar feature: {scalar}")
    scalar_tensor = torch.stack(values, dim=-1)
    return torch.cat((fourier, scalar_tensor), dim=-1)


def build_position_basis(
    *,
    kind: BasisKind,
    extent: int,
    basis_dim: int,
    theta: float,
    scalars: list[str] | tuple[str, ...] = (),
    normalization_extent: int | None = None,
) -> torch.nn.Module:
    basis_types = {
        "frozen_fourier": FrozenFourierBasis,
        "learned_temperature_fourier": LearnedTemperatureFourierBasis,
        "learned_frequency_fourier": LearnedFrequencyFourierBasis,
    }
    try:
        basis_type = basis_types[kind]
    except KeyError as exc:
        raise ValueError(
            f"Unknown position basis kind {kind!r}; expected one of "
            f"{sorted(basis_types)}."
        ) from exc
    if normalization_extent is not None and normalization_extent <= 0:
        raise ValueError("normalization_extent must be positive.")
    return basis_type(
        extent,
        basis_dim,
        theta,
        scalars,
        normalization_extent,
    )
