"""Straight-through autograd parameterizations used by position modules."""

from __future__ import annotations

import torch


class _ExpWithIdentityGradient(torch.autograd.Function):
    """Use ``exp`` in the forward pass and an identity Jacobian backward."""

    @staticmethod
    def forward(ctx, value: torch.Tensor) -> torch.Tensor:
        del ctx
        return torch.exp(value)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor]:
        del ctx
        return (grad_output,)


def exp_with_identity_grad(value: torch.Tensor) -> torch.Tensor:
    """Exponentiate while passing the upstream gradient through unchanged."""

    return _ExpWithIdentityGradient.apply(value)
