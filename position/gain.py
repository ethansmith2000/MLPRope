"""Position-only multiplicative gain applied to rotary Q and K."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import torch

from position.basis import SCALAR_FEATURES, FrozenFourierBasis
from position.channels import GroupedLinearReadout, PositionChannel
from position.mappers import FeatureMapper


POSITION_GAIN_DEFAULTS = {
    "enabled": False,
    "target": "both",
    "head_coupling": "per_head",
    "basis_dim": 16,
    "theta": None,
    "scalars": ["normalized_position", "log_position"],
    "normalization_extent": None,
    "mapper": "linear",
    "hidden_dim": None,
    "log_gain_bound": 1.0,
}


def normalize_position_gain_config(
    config: dict | None,
    *,
    heads: int,
    head_dim: int,
    rope_theta: float,
    normalization_extent: int,
) -> dict:
    if config is None:
        config = {}
    if not isinstance(config, dict):
        raise TypeError("position_gain must be an object")
    unknown = set(config) - set(POSITION_GAIN_DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown position_gain keys: {sorted(unknown)}")
    normalized = copy.deepcopy(POSITION_GAIN_DEFAULTS)
    normalized.update(config)
    if not isinstance(normalized["enabled"], bool):
        raise TypeError("position_gain.enabled must be a boolean")
    if normalized["target"] not in {"q", "k", "both"}:
        raise ValueError("position_gain.target must be 'q', 'k', or 'both'")
    if normalized["head_coupling"] not in {"shared", "per_head"}:
        raise ValueError(
            "position_gain.head_coupling must be 'shared' or 'per_head'"
        )
    if isinstance(heads, bool) or int(heads) <= 0:
        raise ValueError("heads must be a positive integer")
    basis_dim = normalized["basis_dim"]
    if isinstance(basis_dim, bool) or not isinstance(basis_dim, int):
        raise TypeError("position_gain.basis_dim must be an integer")
    if basis_dim <= 0 or basis_dim % 2:
        raise ValueError("position_gain.basis_dim must be a positive even integer")
    scalars = normalized["scalars"]
    if not isinstance(scalars, list) or not all(
        isinstance(value, str) for value in scalars
    ):
        raise TypeError("position_gain.scalars must be a list of strings")
    unknown_scalars = set(scalars) - SCALAR_FEATURES
    if unknown_scalars:
        raise ValueError(
            f"Unknown position_gain scalar features: {sorted(unknown_scalars)}"
        )
    if len(set(scalars)) != len(scalars):
        raise ValueError("position_gain scalar features must not contain duplicates")
    theta = rope_theta if normalized["theta"] is None else float(normalized["theta"])
    if not math.isfinite(theta) or theta <= 0:
        raise ValueError("position_gain.theta must be finite and positive")
    normalized["theta"] = theta
    if normalized["mapper"] != "linear":
        raise ValueError(
            "position_gain.mapper currently supports only the controlled "
            "linear trunk"
        )
    hidden_dim = normalized["hidden_dim"]
    hidden_dim = int(head_dim if hidden_dim is None else hidden_dim)
    if hidden_dim <= 0:
        raise ValueError("position_gain.hidden_dim must be positive")
    normalized["hidden_dim"] = hidden_dim
    bound = normalized["log_gain_bound"]
    if isinstance(bound, bool) or not isinstance(bound, (int, float)):
        raise TypeError("position_gain.log_gain_bound must be a number")
    bound = float(bound)
    if not math.isfinite(bound) or bound <= 0:
        raise ValueError("position_gain.log_gain_bound must be finite and positive")
    normalized["log_gain_bound"] = bound
    configured_extent = normalized["normalization_extent"]
    configured_extent = int(
        normalization_extent if configured_extent is None else configured_extent
    )
    if configured_extent <= 0:
        raise ValueError("position_gain.normalization_extent must be positive")
    normalized["normalization_extent"] = configured_extent
    return normalized


@dataclass
class PositionGainOutput:
    q: torch.Tensor
    k: torch.Tensor


class PositionGain(PositionChannel):
    """Map absolute position features to bounded scalar Q/K gains.

    The readouts are zero-initialized, so both gains are exactly one at the
    anchor.  ``exp(b*tanh(raw/b))`` bounds each gain to ``[exp(-b), exp(b)]``
    while retaining unit derivative with respect to ``raw`` at zero.
    """

    def __init__(
        self,
        config: dict,
        *,
        heads: int,
        head_dim: int,
        extent: int,
    ) -> None:
        super().__init__()
        self.config = copy.deepcopy(config)
        self.heads = int(heads)
        self.groups = self.heads if config["head_coupling"] == "per_head" else 1
        self.extent = int(extent)
        self.basis = FrozenFourierBasis(
            extent=extent,
            basis_dim=int(config["basis_dim"]),
            theta=float(config["theta"]),
            scalars=config["scalars"],
            normalization_extent=int(config["normalization_extent"]),
        )
        self.mapper = FeatureMapper(
            kind="linear",
            groups=self.groups,
            input_dim=self.basis.output_dim,
            output_dim=int(config["hidden_dim"]),
            residual=False,
            rank=1,
            hidden_dim=int(config["hidden_dim"]),
        )
        target = config["target"]
        self.q_readout = (
            GroupedLinearReadout(
                self.groups,
                int(config["hidden_dim"]),
                1,
                init="zeros",
            )
            if target in {"q", "both"}
            else None
        )
        self.k_readout = (
            GroupedLinearReadout(
                self.groups,
                int(config["hidden_dim"]),
                1,
                init="zeros",
            )
            if target in {"k", "both"}
            else None
        )

    def reset_output_parameters(self) -> None:
        for readout in (self.q_readout, self.k_readout):
            if readout is not None:
                readout.reset_parameters()

    def _features(self, length: int, dtype: torch.dtype) -> torch.Tensor:
        features = self.basis(length, dtype=dtype)[None]
        features = features.expand(self.groups, -1, -1)
        return self.mapper(features)

    def _gain(
        self,
        features: torch.Tensor,
        readout: GroupedLinearReadout | None,
    ) -> torch.Tensor:
        if readout is None:
            gain = features.new_ones(self.groups, features.shape[1], 1)
        else:
            raw = readout(features).float()
            bound = float(self.config["log_gain_bound"])
            log_gain = bound * torch.tanh(raw / bound)
            gain = log_gain.exp().to(dtype=features.dtype)
        if self.groups == 1:
            gain = gain.expand(self.heads, -1, -1)
        return gain[None]

    def forward(self, length: int, *, dtype: torch.dtype) -> PositionGainOutput:
        if length <= 0 or length > self.extent:
            raise ValueError(
                f"position_gain length must lie in [1, {self.extent}], got {length}"
            )
        features = self._features(length, dtype)
        return PositionGainOutput(
            q=self._gain(features, self.q_readout),
            k=self._gain(features, self.k_readout),
        )

    @torch.no_grad()
    def diagnostics(self, length: int) -> dict[str, float]:
        parameter = next(self.parameters())
        output = self(length, dtype=parameter.dtype)
        metrics: dict[str, float] = {}
        bound = float(self.config["log_gain_bound"])
        for branch, gain in (("q", output.q), ("k", output.k)):
            values = gain.detach().float()
            log_values = values.log()
            metrics[f"{branch}_gain_mean"] = values.mean().item()
            metrics[f"{branch}_gain_min"] = values.min().item()
            metrics[f"{branch}_gain_max"] = values.max().item()
            metrics[f"{branch}_log_gain_rms"] = (
                log_values.square().mean().sqrt().item()
            )
            metrics[f"{branch}_near_bound_fraction"] = (
                (log_values.abs() >= 0.95 * bound).float().mean().item()
            )
        return metrics
