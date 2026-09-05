"""Optimization diagnostics for learned sinusoidal interventions.

Parameter-gradient magnitude alone is not a functional step-size metric. Adam
rescales coordinates with running moments, so this module logs three levels
without altering optimization:

1. raw and clipped parameter gradients;
2. Adam momentum/second-moment state and the realized parameter update; and
3. exact functional movement of a static sinusoidal carrier.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from position.channels import QKPositionChannel
from position.preprojection import QKPreprojectionPosition


@dataclass
class InterventionParameterGroup:
    name: str
    parameters: list[torch.nn.Parameter]
    static_carrier_modules: list[torch.nn.Module] = field(default_factory=list)


@dataclass
class InterventionOptimizationSample:
    groups: list[InterventionParameterGroup]
    parameter_before: dict[int, torch.Tensor]
    raw_gradients: dict[int, torch.Tensor]
    static_carrier_before: dict[str, list[torch.Tensor]]
    metrics: dict[str, float | None]


def collect_intervention_parameter_groups(
    model: torch.nn.Module,
) -> list[InterventionParameterGroup]:
    """Collect disjoint parameter groups for active sinusoidal interventions."""
    buckets: dict[str, list[torch.nn.Parameter]] = {
        "pre_qk_sinusoid_adapter": [],
        "additive_qk_sinusoid": [],
        "position_content_projection": [],
    }
    seen: set[int] = set()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad or id(parameter) in seen:
            continue
        group_name = None
        if ".qk_preprojection." in name:
            group_name = "pre_qk_sinusoid_adapter"
        elif ".qk_position." in name:
            group_name = "additive_qk_sinusoid"
        elif ".position_content." in name:
            group_name = "position_content_projection"
        if group_name is not None:
            buckets[group_name].append(parameter)
            seen.add(id(parameter))

    preprojection_modules = [
        module
        for module in model.modules()
        if isinstance(module, QKPreprojectionPosition)
    ]
    static_additive_modules = [
        module
        for module in model.modules()
        if isinstance(module, QKPositionChannel)
        and module.conditioning_config["kind"] == "none"
    ]
    return [
        InterventionParameterGroup(
            name=name,
            parameters=parameters,
            static_carrier_modules=(
                preprojection_modules
                if name == "pre_qk_sinusoid_adapter"
                else static_additive_modules
                if name == "additive_qk_sinusoid"
                else []
            ),
        )
        for name, parameters in buckets.items()
        if parameters
    ]


def _optimizer_base(optimizer):
    current = optimizer
    visited = set()
    while hasattr(current, "optimizer") and id(current) not in visited:
        visited.add(id(current))
        nested = current.optimizer
        if nested is current:
            break
        current = nested
    return current


def _vector_stats(values: list[torch.Tensor]) -> dict[str, float]:
    if not values:
        return {
            "l2": 0.0,
            "rms": 0.0,
            "abs_max": 0.0,
            "finite_fraction": 1.0,
        }
    count = sum(value.numel() for value in values)
    square_sum = sum(
        (value.detach().float().square().sum() for value in values),
        start=torch.zeros((), device=values[0].device, dtype=torch.float32),
    )
    finite_count = sum(
        (torch.isfinite(value).sum() for value in values),
        start=torch.zeros((), device=values[0].device, dtype=torch.long),
    )
    abs_max = torch.stack(
        [value.detach().float().abs().max() for value in values]
    ).max()
    return {
        "l2": square_sum.sqrt().item(),
        "rms": (square_sum / count).sqrt().item(),
        "abs_max": abs_max.item(),
        "finite_fraction": (finite_count.float() / count).item(),
    }


def _cosine(left: list[torch.Tensor], right: list[torch.Tensor]) -> float | None:
    if not left or len(left) != len(right):
        return None
    device = left[0].device
    dot = torch.zeros((), device=device, dtype=torch.float32)
    left_square = torch.zeros_like(dot)
    right_square = torch.zeros_like(dot)
    for a, b in zip(left, right, strict=True):
        a_float = a.detach().float()
        b_float = b.detach().float()
        dot += (a_float * b_float).sum()
        left_square += a_float.square().sum()
        right_square += b_float.square().sum()
    denominator = (left_square * right_square).sqrt()
    if denominator.item() == 0:
        return None
    return (dot / denominator).item()


def _prefixed(
    target: dict[str, float | None],
    prefix: str,
    values: dict[str, float],
) -> None:
    for key, value in values.items():
        target[f"{prefix}/{key}"] = value


def _adam_state_metrics(
    optimizer,
    group: InterventionParameterGroup,
    gradients: list[torch.Tensor],
    prefix: str,
) -> dict[str, float | None]:
    optimizer = _optimizer_base(optimizer)
    momenta = []
    second_moments = []
    matched_gradients = []
    for parameter, gradient in zip(group.parameters, gradients, strict=True):
        state = optimizer.state.get(parameter, {})
        momentum = state.get("exp_avg")
        second_moment = state.get("exp_avg_sq")
        if momentum is not None and second_moment is not None:
            momenta.append(momentum)
            second_moments.append(second_moment)
            matched_gradients.append(gradient)
    if not momenta:
        return {
            f"{prefix}/gradient_momentum_cosine": None,
            f"{prefix}/sqrt_second_moment_rms": None,
            f"{prefix}/sqrt_second_moment_abs_max": None,
            f"{prefix}/sqrt_second_moment_max_to_rms": None,
        }
    sqrt_second = [value.detach().float().sqrt() for value in second_moments]
    stats = _vector_stats(sqrt_second)
    return {
        f"{prefix}/gradient_momentum_cosine": _cosine(
            matched_gradients,
            momenta,
        ),
        f"{prefix}/sqrt_second_moment_rms": stats["rms"],
        f"{prefix}/sqrt_second_moment_abs_max": stats["abs_max"],
        f"{prefix}/sqrt_second_moment_max_to_rms": (
            stats["abs_max"] / stats["rms"] if stats["rms"] > 0 else None
        ),
    }


def _sample_static_carriers(
    modules: list[torch.nn.Module],
    *,
    reference_length: int,
    sample_count: int = 64,
) -> list[torch.Tensor]:
    """Evaluate position-only Q/K carriers at points spanning the context."""
    if not modules:
        return []
    parameter = next(modules[0].parameters(), None)
    buffer = next(modules[0].buffers(), None)
    device = (
        parameter.device
        if parameter is not None
        else buffer.device
        if buffer is not None
        else torch.device("cpu")
    )
    positions = torch.linspace(
        0,
        reference_length - 1,
        min(reference_length, sample_count),
        device=device,
        dtype=torch.float32,
    ).round().long().unique()
    values = []
    for module in modules:
        output = module(reference_length, dtype=torch.float32)
        for branch in (output.q, output.k):
            if branch.shape[-2] != reference_length:
                raise ValueError(
                    "Static carrier output must use its penultimate dimension "
                    "for sequence position"
                )
            values.append(
                branch.detach().float().index_select(-2, positions).clone()
            )
    return values


class InterventionOptimizationMonitor:
    """Sparse, read-only monitor for position-intervention optimizer health."""

    def __init__(
        self,
        groups: list[InterventionParameterGroup],
        *,
        reference_length: int,
    ) -> None:
        self.groups = groups
        self.reference_length = int(reference_length)
        if self.reference_length <= 0:
            raise ValueError("reference_length must be positive")

    @property
    def enabled(self) -> bool:
        return bool(self.groups)

    @torch.no_grad()
    def capture_before_clip(self, optimizer) -> InterventionOptimizationSample:
        optimizer_base = _optimizer_base(optimizer)
        learning_rate_by_parameter = {
            id(parameter): float(optimizer_group["lr"])
            for optimizer_group in optimizer_base.param_groups
            for parameter in optimizer_group["params"]
        }
        parameter_before = {}
        raw_gradients = {}
        static_carrier_before = {}
        metrics: dict[str, float | None] = {}
        for group in self.groups:
            parameters = [p for p in group.parameters if p.grad is not None]
            gradients = [p.grad.detach().float().clone() for p in parameters]
            for parameter, gradient in zip(parameters, gradients, strict=True):
                parameter_before[id(parameter)] = parameter.detach().float().clone()
                raw_gradients[id(parameter)] = gradient
            prefix = f"optimization/{group.name}"
            learning_rates = [
                learning_rate_by_parameter[id(parameter)]
                for parameter in parameters
            ]
            if learning_rates:
                metrics[f"{prefix}/learning_rate_min"] = min(learning_rates)
                metrics[f"{prefix}/learning_rate_max"] = max(learning_rates)
            _prefixed(metrics, f"{prefix}/raw_gradient", _vector_stats(gradients))
            parameter_values = [p.detach().float() for p in parameters]
            _prefixed(metrics, f"{prefix}/parameter", _vector_stats(parameter_values))
            metrics.update(
                _adam_state_metrics(
                    optimizer,
                    InterventionParameterGroup(
                        group.name,
                        parameters,
                    ),
                    gradients,
                    f"{prefix}/adam_before",
                )
            )
            if group.static_carrier_modules:
                static_carrier_before[group.name] = _sample_static_carriers(
                    group.static_carrier_modules,
                    reference_length=self.reference_length,
                )
        return InterventionOptimizationSample(
            groups=self.groups,
            parameter_before=parameter_before,
            raw_gradients=raw_gradients,
            static_carrier_before=static_carrier_before,
            metrics=metrics,
        )

    @torch.no_grad()
    def capture_after_clip(
        self,
        sample: InterventionOptimizationSample,
    ) -> None:
        for group in sample.groups:
            parameters = [p for p in group.parameters if id(p) in sample.raw_gradients]
            clipped = [p.grad.detach().float() for p in parameters]
            prefix = f"optimization/{group.name}"
            stats = _vector_stats(clipped)
            _prefixed(sample.metrics, f"{prefix}/clipped_gradient", stats)
            raw_l2 = sample.metrics[f"{prefix}/raw_gradient/l2"]
            sample.metrics[f"{prefix}/gradient_clip_ratio"] = (
                stats["l2"] / raw_l2 if raw_l2 and raw_l2 > 0 else None
            )

    @torch.no_grad()
    def capture_after_step(
        self,
        sample: InterventionOptimizationSample,
        optimizer,
    ) -> dict[str, float | None]:
        for group in sample.groups:
            parameters = [p for p in group.parameters if id(p) in sample.raw_gradients]
            raw_gradients = [sample.raw_gradients[id(p)] for p in parameters]
            updates = [
                p.detach().float() - sample.parameter_before[id(p)]
                for p in parameters
            ]
            prefix = f"optimization/{group.name}"
            update_stats = _vector_stats(updates)
            _prefixed(sample.metrics, f"{prefix}/parameter_update", update_stats)
            sample.metrics[f"{prefix}/descent_update_gradient_cosine"] = _cosine(
                [-update for update in updates],
                raw_gradients,
            )
            raw_l2 = sample.metrics[f"{prefix}/raw_gradient/l2"]
            sample.metrics[f"{prefix}/update_to_raw_gradient_l2_ratio"] = (
                update_stats["l2"] / raw_l2 if raw_l2 and raw_l2 > 0 else None
            )
            sample.metrics.update(
                _adam_state_metrics(
                    optimizer,
                    InterventionParameterGroup(
                        group.name,
                        parameters,
                    ),
                    raw_gradients,
                    f"{prefix}/adam_after",
                )
            )
            if group.static_carrier_modules:
                before = sample.static_carrier_before[group.name]
                after = _sample_static_carriers(
                    group.static_carrier_modules,
                    reference_length=self.reference_length,
                )
                function_steps = [
                    current - previous
                    for previous, current in zip(before, after, strict=True)
                ]
                function_stats = _vector_stats(function_steps)
                _prefixed(
                    sample.metrics,
                    f"{prefix}/carrier_function_step",
                    function_stats,
                )
                sample.metrics[
                    f"{prefix}/carrier_function_to_parameter_update_rms_ratio"
                ] = (
                    function_stats["rms"] / update_stats["rms"]
                    if update_stats["rms"] > 0
                    else None
                )
        return sample.metrics


def intervention_optimization_due(
    step: int,
    *,
    warmup_steps: list[int],
    every: int | None,
) -> bool:
    if step in warmup_steps:
        return True
    return every is not None and step % every == 0
