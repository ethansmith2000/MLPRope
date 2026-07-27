"""Position feature mappers with a uniform grouped tensor contract."""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F


MapperKind = Literal[
    "identity",
    "euclidean_affine",
    "linear",
    "low_rank",
    "bottleneck_mlp",
    "mlp",
]


class FeatureMapper(torch.nn.Module):
    """Map ``[groups, length, input_dim]`` features to the same layout."""

    kind: MapperKind

    def __init__(
        self,
        *,
        kind: MapperKind,
        groups: int,
        input_dim: int,
        output_dim: int,
        residual: bool,
        rank: int,
        hidden_dim: int,
    ):
        super().__init__()
        if groups <= 0:
            raise ValueError("groups must be positive.")
        if input_dim <= 0 or output_dim <= 0:
            raise ValueError("mapper dimensions must be positive.")
        if residual and input_dim != output_dim:
            raise ValueError(
                "Residual mappers require matching input and output dimensions."
            )
        if kind == "identity" and input_dim != output_dim:
            raise ValueError(
                "Identity mapper requires matching input and output dimensions."
            )
        if kind == "euclidean_affine" and input_dim != output_dim:
            raise ValueError(
                "Euclidean affine mapper requires matching input/output dimensions."
            )
        if rank <= 0:
            raise ValueError("rank must be positive.")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")

        self.kind = kind
        self.groups = groups
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.residual = residual
        self.rank = rank
        self.hidden_dim = hidden_dim
        self.branch_hidden = rank if kind in {"low_rank", "bottleneck_mlp"} else hidden_dim

        if kind == "euclidean_affine":
            self.scale = torch.nn.Parameter(torch.zeros(groups, input_dim))
            self.offset = torch.nn.Parameter(torch.zeros(groups, input_dim))
        elif kind == "linear":
            self.weight = torch.nn.Parameter(
                torch.empty(groups, input_dim, output_dim)
            )
            self.bias = torch.nn.Parameter(torch.zeros(groups, output_dim))
        elif kind in {"low_rank", "bottleneck_mlp", "mlp"}:
            self.down = torch.nn.Parameter(
                torch.empty(groups, input_dim, self.branch_hidden)
            )
            self.down_bias = torch.nn.Parameter(
                torch.zeros(groups, self.branch_hidden)
            )
            self.up = torch.nn.Parameter(
                torch.zeros(groups, self.branch_hidden, output_dim)
            )
            self.up_bias = torch.nn.Parameter(torch.zeros(groups, output_dim))
        elif kind != "identity":
            raise ValueError(f"Unknown mapper kind: {kind!r}")
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.kind == "euclidean_affine":
            torch.nn.init.zeros_(self.scale)
            torch.nn.init.zeros_(self.offset)
        elif self.kind == "linear":
            for weight in self.weight:
                torch.nn.init.xavier_normal_(weight)
            torch.nn.init.zeros_(self.bias)
        elif self.kind in {"low_rank", "bottleneck_mlp", "mlp"}:
            for weight in self.down:
                torch.nn.init.xavier_normal_(weight)
            torch.nn.init.zeros_(self.down_bias)
            # Residual branches start as exact identity. Direct-output branches
            # need a live signal (especially when followed by a zero gate).
            if self.residual:
                torch.nn.init.zeros_(self.up)
            else:
                for weight in self.up:
                    torch.nn.init.xavier_normal_(weight)
            torch.nn.init.zeros_(self.up_bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3:
            raise ValueError(
                f"Expected features of shape [groups, length, dim], got {tuple(features.shape)}"
            )
        if features.shape[0] != self.groups:
            raise ValueError(
                f"Expected {self.groups} groups, got {features.shape[0]}."
            )
        if features.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected input dim {self.input_dim}, got {features.shape[-1]}."
            )

        if self.kind == "identity":
            mapped = features
        elif self.kind == "euclidean_affine":
            mapped = (
                features * (1.0 + self.scale[:, None, :])
                + self.offset[:, None, :]
            )
        elif self.kind == "linear":
            mapped = torch.einsum("grd,gde->gre", features, self.weight)
            mapped = mapped + self.bias[:, None, :]
            if self.residual:
                mapped = features + mapped
        else:
            hidden = torch.einsum("grd,gdk->grk", features, self.down)
            hidden = hidden + self.down_bias[:, None, :]
            if self.kind in {"bottleneck_mlp", "mlp"}:
                hidden = F.gelu(hidden)
            branch = torch.einsum("grk,gkd->grd", hidden, self.up)
            branch = branch + self.up_bias[:, None, :]
            mapped = features + branch if self.residual else branch
        return mapped


def build_mapper(
    *,
    kind: MapperKind,
    groups: int,
    input_dim: int,
    output_dim: int,
    residual: bool,
    rank: int,
    hidden_dim: int,
) -> FeatureMapper:
    return FeatureMapper(
        kind=kind,
        groups=groups,
        input_dim=input_dim,
        output_dim=output_dim,
        residual=residual,
        rank=rank,
        hidden_dim=hidden_dim,
    )
