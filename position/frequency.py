"""Position-coherent frequency banks for the pre-Q/K sinusoidal carrier.

These coordinates never modify RoPE.  The model's RoPE path is either the
standard fixed rotation or disabled.  A bank here belongs only to the separate
sinusoidal carrier and is shared by every layer and Q/K branch that consumes
it.  It is static, so it cannot read future tokens or introduce content
leakage.
"""

from __future__ import annotations

import copy
import math
from typing import Literal

import torch

from position.precision import PreserveFP32BuffersMixin
from position.rotary import build_rope_frequencies


FrequencyMode = Literal["fixed", "learned_log", "learned_horizon"]
FREQUENCY_MODES = {"fixed", "learned_log", "learned_horizon"}

SINUSOID_FREQUENCY_DEFAULTS = {
    "mode": "fixed",
    # Only learned_horizon uses this.  It is resolved explicitly in saved
    # configs so changing model_position_extent later cannot change its meaning.
    "reference_length": None,
    # Learned frequency coordinates are clipped as a group, independently of
    # ordinary model parameters.  Null disables their dedicated clipping.
    "max_grad_norm": 1.0,
}


def normalize_sinusoid_frequency_config(
    config: dict | None,
    *,
    default_reference_length: int | None,
) -> dict:
    """Validate and fully resolve a shared-frequency configuration."""
    if config is None:
        config = {}
    if not isinstance(config, dict):
        raise TypeError("frequency configuration must be an object")
    unknown = set(config) - set(SINUSOID_FREQUENCY_DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown frequency configuration keys: {sorted(unknown)}")
    normalized = copy.deepcopy(SINUSOID_FREQUENCY_DEFAULTS)
    normalized.update(config)

    mode = normalized["mode"]
    if mode not in FREQUENCY_MODES:
        raise ValueError(
            f"frequency mode must be one of {sorted(FREQUENCY_MODES)}, got {mode!r}"
        )

    reference_length = normalized["reference_length"]
    if reference_length is None:
        reference_length = default_reference_length
    if reference_length is not None:
        if isinstance(reference_length, bool):
            raise TypeError("frequency reference_length must be an integer")
        reference_length = int(reference_length)
        if reference_length <= 0:
            raise ValueError("frequency reference_length must be positive")
    if mode == "learned_horizon" and reference_length is None:
        raise ValueError("learned_horizon requires a reference_length")
    normalized["reference_length"] = reference_length

    max_grad_norm = normalized["max_grad_norm"]
    if max_grad_norm is not None:
        if isinstance(max_grad_norm, bool) or not isinstance(
            max_grad_norm, (int, float)
        ):
            raise TypeError("frequency max_grad_norm must be a number or null")
        max_grad_norm = float(max_grad_norm)
        if not math.isfinite(max_grad_norm) or max_grad_norm <= 0:
            raise ValueError("frequency max_grad_norm must be finite and positive")
    normalized["max_grad_norm"] = max_grad_norm
    return normalized


class SinusoidFrequencyBank(PreserveFP32BuffersMixin, torch.nn.Module):
    """One carrier-frequency schedule shared globally by all consumers.

    ``learned_log`` uses ``omega = omega_0 * exp(alpha)``.  It is the stable,
    positive parameterization used by LeRoPE.

    ``learned_horizon`` uses
    ``omega = omega_0 + rho / L_ref``.  Consequently the phase is
    ``p*omega_0 + (p/L_ref)*rho`` and its derivative with respect to ``rho`` is
    at most one over the reference context.  This is normalized rather than
    bounded: it contains no tanh or other saturation.
    """

    _fp32_buffer_names = ("base_frequency",)
    _fp32_parameter_names = ("coordinate",)

    def __init__(
        self,
        dimension: int,
        theta: float,
        config: dict,
    ) -> None:
        super().__init__()
        self.dimension = int(dimension)
        self.theta = float(theta)
        self.config = copy.deepcopy(config)
        self.mode: FrequencyMode = config["mode"]
        self.reference_length = config["reference_length"]
        self.max_grad_norm = config["max_grad_norm"]
        self.register_buffer(
            "base_frequency",
            build_rope_frequencies(self.dimension, self.theta),
            persistent=False,
        )
        if self.mode == "fixed":
            self.register_parameter("coordinate", None)
        else:
            self.coordinate = torch.nn.Parameter(
                torch.zeros(self.dimension // 2, dtype=torch.float32)
            )

    def _apply(self, fn, recurse: bool = True):
        # Frequency coordinates retain angular meaning and optimizer state in
        # fp32 even when the surrounding model is explicitly cast to bf16/fp16.
        originals = {
            name: parameter.detach().float().clone()
            for name in self._fp32_parameter_names
            if (parameter := self._parameters.get(name)) is not None
        }
        module = super()._apply(fn, recurse=recurse)
        for name, original in originals.items():
            parameter = self._parameters[name]
            parameter.data = original.to(device=parameter.device, dtype=torch.float32)
            if parameter.grad is not None:
                parameter.grad.data = parameter.grad.detach().to(
                    device=parameter.device,
                    dtype=torch.float32,
                )
        return module

    @property
    def learned(self) -> bool:
        return self.coordinate is not None

    def frequencies(self) -> torch.Tensor:
        base = self.base_frequency.float()
        if self.mode == "fixed":
            return base
        coordinate = self.coordinate.float()
        if self.mode == "learned_log":
            return base * coordinate.exp()
        if self.mode == "learned_horizon":
            return base + coordinate / float(self.reference_length)
        raise AssertionError(f"Unhandled frequency mode: {self.mode}")

    def forward(self) -> torch.Tensor:
        return self.frequencies()

    def endpoint_phase_delta(self) -> torch.Tensor:
        reference_length = float(self.reference_length or 1)
        return (self.frequencies() - self.base_frequency.float()) * reference_length

    def endpoint_phase_coordinate_jacobian(self) -> torch.Tensor:
        """Derivative of endpoint phase with respect to each coordinate.

        Adam may normalize coordinate gradients, but it does not normalize this
        forward Jacobian.  This quantity therefore exposes how a similarly sized
        optimizer step can imply very different functional phase movements.
        """
        if self.mode == "fixed":
            return torch.zeros_like(self.base_frequency, dtype=torch.float32)
        if self.mode == "learned_log":
            return self.frequencies() * float(self.reference_length or 1)
        if self.mode == "learned_horizon":
            return torch.ones_like(self.base_frequency, dtype=torch.float32)
        raise AssertionError(f"Unhandled frequency mode: {self.mode}")

    @torch.no_grad()
    def summarize(self) -> dict[str, float]:
        frequency = self.frequencies().detach().float()
        base = self.base_frequency.detach().float()
        multiplier = frequency / base
        endpoint_delta = self.endpoint_phase_delta().detach().float()
        endpoint_jacobian = (
            self.endpoint_phase_coordinate_jacobian().detach().float()
        )
        if frequency.numel() > 1:
            order_violation = (frequency[1:] >= frequency[:-1]).float().mean()
        else:
            order_violation = frequency.new_zeros(())
        return {
            "frequency_min": frequency.min().item(),
            "frequency_max": frequency.max().item(),
            "frequency_mean": frequency.mean().item(),
            "frequency_nonpositive_fraction": (frequency <= 0).float().mean().item(),
            "frequency_order_violation_fraction": order_violation.item(),
            "multiplier_min": multiplier.min().item(),
            "multiplier_max": multiplier.max().item(),
            "multiplier_mean": multiplier.mean().item(),
            "multiplier_std": multiplier.std(unbiased=False).item(),
            "endpoint_phase_delta_rms": endpoint_delta.square().mean().sqrt().item(),
            "endpoint_phase_delta_abs_max": endpoint_delta.abs().max().item(),
            "endpoint_phase_coordinate_jacobian_rms": (
                endpoint_jacobian.square().mean().sqrt().item()
            ),
            "endpoint_phase_coordinate_jacobian_abs_max": (
                endpoint_jacobian.abs().max().item()
            ),
        }
