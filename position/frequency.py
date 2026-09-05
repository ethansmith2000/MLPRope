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


FrequencyMode = Literal[
    "fixed",
    "learned_log",
    "learned_horizon",
    "learned_global_direct",
    "learned_hybrid_direct",
]
FREQUENCY_MODES = {
    "fixed",
    "learned_log",
    "learned_horizon",
    "learned_global_direct",
    "learned_hybrid_direct",
}

SINUSOID_FREQUENCY_DEFAULTS = {
    "mode": "fixed",
    # Only learned_horizon uses this.  It is resolved explicitly in saved
    # configs so changing model_position_extent later cannot change its meaning.
    "reference_length": None,
    # Learned frequency coordinates are clipped as a group, independently of
    # ordinary model parameters.  Null disables their dedicated clipping.
    "max_grad_norm": 1.0,
    # Direct modes measure their fixed gains in radians of phase at the
    # reference horizon. This is a coordinate scale, not a bound or
    # saturating nonlinearity.
    "endpoint_phase_scale": 1.0,
    # Only learned_hybrid_direct uses this low-order DCT coordinate count.
    "smooth_rank": 4,
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
    if mode in {
        "learned_horizon",
        "learned_global_direct",
        "learned_hybrid_direct",
    } and reference_length is None:
        raise ValueError(f"{mode} requires a reference_length")
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

    endpoint_phase_scale = normalized["endpoint_phase_scale"]
    if isinstance(endpoint_phase_scale, bool) or not isinstance(
        endpoint_phase_scale,
        (int, float),
    ):
        raise TypeError("frequency endpoint_phase_scale must be a number")
    endpoint_phase_scale = float(endpoint_phase_scale)
    if not math.isfinite(endpoint_phase_scale) or endpoint_phase_scale <= 0:
        raise ValueError(
            "frequency endpoint_phase_scale must be finite and positive"
        )
    normalized["endpoint_phase_scale"] = endpoint_phase_scale

    smooth_rank = normalized["smooth_rank"]
    if isinstance(smooth_rank, bool) or not isinstance(smooth_rank, int):
        raise TypeError("frequency smooth_rank must be an integer")
    if smooth_rank <= 0:
        raise ValueError("frequency smooth_rank must be positive")
    normalized["smooth_rank"] = smooth_rank
    return normalized


def _smooth_frequency_basis(size: int, count: int) -> torch.Tensor:
    """Unit-RMS DCT-II coordinates over the ordered frequency index.

    The constant mode is included because the frequency bank has no separate
    learned offset. Nonconstant modes use the usual sqrt(2) normalization.
    """
    if count > size:
        raise ValueError("frequency smooth_rank cannot exceed the pair dimension")
    index = torch.arange(size, dtype=torch.float32)[:, None] + 0.5
    modes = torch.arange(count, dtype=torch.float32)[None, :]
    basis = torch.cos(math.pi * index * modes / size)
    if count > 1:
        basis[:, 1:] *= math.sqrt(2.0)
    return basis


class SinusoidFrequencyBank(PreserveFP32BuffersMixin, torch.nn.Module):
    """One carrier-frequency schedule shared globally by all consumers.

    ``learned_log`` uses ``omega = omega_0 * exp(alpha)``.  It is the stable,
    positive parameterization used by LeRoPE.

    ``learned_horizon`` uses
    ``omega = omega_0 + rho / L_ref``.  Consequently the phase is
    ``p*omega_0 + (p/L_ref)*rho`` and its derivative with respect to ``rho`` is
    at most one over the reference context.  This is normalized rather than
    bounded: it contains no tanh or other saturation.

    ``learned_global_direct`` learns one direct coordinate that dilates the
    whole schedule. Its fixed gain makes one coordinate unit equal
    ``endpoint_phase_scale`` radians for the highest-frequency pair at the
    reference horizon.

    ``learned_hybrid_direct`` learns a few smooth direct coordinates. Pair
    ``i`` uses gain ``min(omega_i, endpoint_phase_scale / L_ref)``: low
    frequencies therefore move relatively, while high-frequency endpoint
    phase sensitivity is capped in scale without a nonlinear parameter map.
    """

    _fp32_buffer_names = (
        "base_frequency",
        "coordinate_basis",
        "direct_gain",
    )
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
        self.endpoint_phase_scale = float(config["endpoint_phase_scale"])
        self.smooth_rank = int(config["smooth_rank"])
        self.register_buffer(
            "base_frequency",
            build_rope_frequencies(self.dimension, self.theta),
            persistent=False,
        )
        pair_dim = self.dimension // 2
        if self.mode == "learned_hybrid_direct":
            coordinate_basis = _smooth_frequency_basis(
                pair_dim,
                self.smooth_rank,
            )
            direct_gain = torch.minimum(
                self.base_frequency,
                self.base_frequency.new_tensor(
                    self.endpoint_phase_scale / float(self.reference_length)
                ),
            )
        elif self.mode == "learned_global_direct":
            coordinate_basis = torch.ones((pair_dim, 1), dtype=torch.float32)
            direct_gain = (
                self.base_frequency
                * self.endpoint_phase_scale
                / (
                    float(self.reference_length)
                    * self.base_frequency.max()
                )
            )
        else:
            coordinate_basis = torch.empty((pair_dim, 0), dtype=torch.float32)
            direct_gain = torch.empty((pair_dim,), dtype=torch.float32)
        self.register_buffer(
            "coordinate_basis",
            coordinate_basis,
            persistent=False,
        )
        self.register_buffer("direct_gain", direct_gain, persistent=False)
        if self.mode == "fixed":
            self.register_parameter("coordinate", None)
        else:
            coordinate_count = (
                1
                if self.mode == "learned_global_direct"
                else self.smooth_rank
                if self.mode == "learned_hybrid_direct"
                else pair_dim
            )
            self.coordinate = torch.nn.Parameter(
                torch.zeros(coordinate_count, dtype=torch.float32)
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
        if self.mode in {"learned_global_direct", "learned_hybrid_direct"}:
            deformation = self.coordinate_basis @ coordinate
            return base + self.direct_gain * deformation
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
        if self.mode in {"learned_global_direct", "learned_hybrid_direct"}:
            return (
                float(self.reference_length)
                * self.direct_gain[:, None]
                * self.coordinate_basis
            )
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
            "coordinate_count": float(
                0 if self.coordinate is None else self.coordinate.numel()
            ),
            "endpoint_phase_delta_rms": endpoint_delta.square().mean().sqrt().item(),
            "endpoint_phase_delta_abs_max": endpoint_delta.abs().max().item(),
            "endpoint_phase_coordinate_jacobian_rms": (
                endpoint_jacobian.square().mean().sqrt().item()
            ),
            "endpoint_phase_coordinate_jacobian_abs_max": (
                endpoint_jacobian.abs().max().item()
            ),
        }
