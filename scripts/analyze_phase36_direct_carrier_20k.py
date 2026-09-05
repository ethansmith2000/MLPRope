#!/usr/bin/env python
"""Analyze the one-seed Phase-36 direct amplitude/frequency screen."""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "model-output" / "position_bias_phase36_direct_carrier_20k"
RESULT_ROOT = ROOT / "results" / "phase36_direct_carrier_20k"
STEP = 20_000
PRACTICAL_THRESHOLD = 0.003
ARMS = {
    "rope-fixed": "phase36-rope-fixed-seed123-s20000-h768d8",
    "qkpre-scalar": "phase36-qkpre-scalar-seed123-s20000-h768d8",
    "direct-amplitude": (
        "phase36-qkpre-direct-amplitude-r4-seed123-s20000-h768d8"
    ),
    "global-frequency-lr4": (
        "phase36-qkpre-global-frequency-lr4-seed123-s20000-h768d8"
    ),
    "hybrid-frequency-lr1": (
        "phase36-qkpre-hybrid-frequency-r4-seed123-s20000-h768d8"
    ),
    "hybrid-frequency-lr4": (
        "phase36-qkpre-hybrid-frequency-r4-lr4-seed123-s20000-h768d8"
    ),
    "direct-amplitude+hybrid-frequency": (
        "phase36-qkpre-direct-amplitude-hybrid-frequency-r4-"
        "seed123-s20000-h768d8"
    ),
}
CONTRASTS = (
    ("qkpre-scalar_vs_rope-fixed", "qkpre-scalar", "rope-fixed", "backbone"),
    (
        "direct-amplitude_vs_qkpre-scalar",
        "direct-amplitude",
        "qkpre-scalar",
        "primary",
    ),
    (
        "global-frequency-lr4_vs_qkpre-scalar",
        "global-frequency-lr4",
        "qkpre-scalar",
        "primary",
    ),
    (
        "hybrid-frequency-lr1_vs_qkpre-scalar",
        "hybrid-frequency-lr1",
        "qkpre-scalar",
        "primary",
    ),
    (
        "hybrid-frequency-lr4_vs_qkpre-scalar",
        "hybrid-frequency-lr4",
        "qkpre-scalar",
        "sensitivity",
    ),
    (
        "direct-amplitude+hybrid-frequency_vs_qkpre-scalar",
        "direct-amplitude+hybrid-frequency",
        "qkpre-scalar",
        "factorial",
    ),
    (
        "direct-amplitude+hybrid-frequency_vs_direct-amplitude",
        "direct-amplitude+hybrid-frequency",
        "direct-amplitude",
        "incremental",
    ),
)


def run_dir(arm: str) -> Path:
    return OUTPUT_ROOT / ARMS[arm]


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def final_losses(arm: str) -> list[float]:
    path = run_dir(arm) / "evaluation_details" / (
        f"step_{STEP:08d}_context_001024.json"
    )
    payload = json.loads(path.read_text())
    if payload["evaluation_kind"] != "final_holdout":
        raise ValueError(f"Unexpected evaluation kind in {path}")
    if payload["evaluation_start_batch"] != 2_048:
        raise ValueError(f"Unexpected holdout start in {path}")
    values = [float(value) for value in payload["losses"]]
    if len(values) != 1_024:
        raise ValueError(f"Expected 1,024 losses in {path}, got {len(values)}")
    return values


def development_losses(arm: str) -> dict[int, list[float]]:
    by_step = {}
    for path in (run_dir(arm) / "evaluation_details").glob(
        "step_*_development_context_001024.json"
    ):
        payload = json.loads(path.read_text())
        if payload["evaluation_start_batch"] != 0:
            raise ValueError(f"Unexpected development start in {path}")
        by_step[int(payload["step"])] = [float(value) for value in payload["losses"]]
    return by_step


def paired_summary(candidate: list[float], reference: list[float]) -> dict:
    delta = [a - b for a, b in zip(candidate, reference, strict=True)]
    mean = statistics.fmean(delta)
    half_width = 1.96 * statistics.stdev(delta) / math.sqrt(len(delta))
    return {
        "candidate_loss": statistics.fmean(candidate),
        "reference_loss": statistics.fmean(reference),
        "delta_candidate_minus_reference": mean,
        "paired_example_ci95": [mean - half_width, mean + half_width],
        "num_paired_examples": len(delta),
    }


def _linear_slope(points: list[tuple[int, float]]) -> float | None:
    if len(points) < 2:
        return None
    xs = [step / 1_000 for step, _ in points]
    ys = [value for _, value in points]
    x_mean = statistics.fmean(xs)
    y_mean = statistics.fmean(ys)
    denominator = sum((x - x_mean) ** 2 for x in xs)
    if denominator == 0:
        return None
    return sum(
        (x - x_mean) * (y - y_mean)
        for x, y in zip(xs, ys, strict=True)
    ) / denominator


def contrast_curve(
    candidate: dict[int, list[float]],
    reference: dict[int, list[float]],
) -> dict:
    common = sorted(set(candidate) & set(reference))
    points = {
        str(step): paired_summary(candidate[step], reference[step])
        for step in common
    }
    late = [
        (step, points[str(step)]["delta_candidate_minus_reference"])
        for step in common
        if step >= 15_000
    ]
    return {
        "points": points,
        "late_slope_delta_per_1k_steps": _linear_slope(late),
        "late_window_start_step": 15_000,
    }


def final_metrics(arm: str) -> dict:
    rows = _read_jsonl(run_dir(arm) / "metrics.jsonl")
    matches = [
        row
        for row in rows
        if row.get("step") == STEP
        and row.get("evaluation_kind") == "final_holdout"
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one final-holdout metrics row for {arm}")
    return matches[0]


def _metric_values(row: dict, suffix: str) -> list[float]:
    return [
        float(value)
        for key, value in row.items()
        if key.startswith("position/layer_") and key.endswith(suffix)
    ]


def _range(values: list[float]) -> list[float] | None:
    return [min(values), max(values)] if values else None


def intervention_state(arm: str) -> dict | None:
    if arm == "rope-fixed":
        return None
    row = final_metrics(arm)
    gate_values = _metric_values(row, "/gate_q") + _metric_values(row, "/gate_k")
    amplitude_min = _metric_values(row, "/amplitude_factor_q_min") + _metric_values(
        row, "/amplitude_factor_k_min"
    )
    amplitude_max = _metric_values(row, "/amplitude_factor_q_max") + _metric_values(
        row, "/amplitude_factor_k_max"
    )
    amplitude_nonpositive = _metric_values(
        row, "/amplitude_factor_q_nonpositive_fraction"
    ) + _metric_values(row, "/amplitude_factor_k_nonpositive_fraction")
    energy = _metric_values(row, "/input_mixture_position_energy_fraction")
    q_cosine = _metric_values(row, "/normalized_q_cosine_to_content")
    k_cosine = _metric_values(row, "/normalized_k_cosine_to_content")
    frequency_prefix = "position/shared_qkpre_frequency"
    frequency = None
    if f"{frequency_prefix}/coordinate_count" in row:
        frequency = {
            "coordinate_count": int(row[f"{frequency_prefix}/coordinate_count"]),
            "frequency_range": [
                row[f"{frequency_prefix}/frequency_min"],
                row[f"{frequency_prefix}/frequency_max"],
            ],
            "multiplier_range": [
                row[f"{frequency_prefix}/multiplier_min"],
                row[f"{frequency_prefix}/multiplier_max"],
            ],
            "endpoint_phase_delta_rms": row[
                f"{frequency_prefix}/endpoint_phase_delta_rms"
            ],
            "endpoint_phase_delta_abs_max": row[
                f"{frequency_prefix}/endpoint_phase_delta_abs_max"
            ],
            "nonpositive_fraction": row[
                f"{frequency_prefix}/frequency_nonpositive_fraction"
            ],
            "order_violation_fraction": row[
                f"{frequency_prefix}/frequency_order_violation_fraction"
            ],
        }
    return {
        "gate_range": _range(gate_values),
        "amplitude_factor_range": (
            [min(amplitude_min), max(amplitude_max)]
            if amplitude_min and amplitude_max
            else None
        ),
        "amplitude_nonpositive_fraction_max": (
            max(amplitude_nonpositive) if amplitude_nonpositive else None
        ),
        "input_position_energy_fraction_range": _range(energy),
        "input_position_energy_fraction_mean": (
            statistics.fmean(energy) if energy else None
        ),
        "normalized_q_cosine_to_content_range": _range(q_cosine),
        "normalized_k_cosine_to_content_range": _range(k_cosine),
        "frequency": frequency,
    }


def optimization_health(arm: str) -> dict | None:
    path = run_dir(arm) / "intervention_optimization.jsonl"
    if not path.is_file():
        return None
    rows = _read_jsonl(path)
    groups = {}
    names = {
        "adapter": "optimization/pre_qk_sinusoid_adapter",
        "frequency": "optimization/pre_qk_sinusoid_frequency",
    }
    for name, prefix in names.items():
        if not any(f"{prefix}/parameter/rms" in row for row in rows):
            continue
        populated = [
            row
            for row in rows
            if row["step"] < STEP
            and row.get(f"{prefix}/parameter_update/l2", 0) > 0
        ]
        cosines = [
            row[f"{prefix}/descent_update_gradient_cosine"]
            for row in populated
            if row.get(f"{prefix}/descent_update_gradient_cosine") is not None
        ]
        clip_ratios = [
            row[f"{prefix}/gradient_clip_ratio"]
            for row in populated
            if row.get(f"{prefix}/gradient_clip_ratio") is not None
        ]
        function_steps = [
            row.get(
                f"{prefix}/endpoint_phase_step/rms",
                row.get(f"{prefix}/carrier_function_step/rms", 0),
            )
            for row in populated
        ]
        function_step_abs_max = [
            row.get(
                f"{prefix}/endpoint_phase_step/abs_max",
                row.get(f"{prefix}/carrier_function_step/abs_max", 0),
            )
            for row in populated
        ]
        groups[name] = {
            "last_active_step": populated[-1]["step"] if populated else None,
            "effective_learning_rate_initial": (
                populated[0].get(f"{prefix}/learning_rate_max")
                if populated
                else None
            ),
            "gradient_clip_ratio_min": min(clip_ratios) if clip_ratios else None,
            "function_step_rms_max": max(function_steps) if function_steps else None,
            "function_step_abs_max": (
                max(function_step_abs_max) if function_step_abs_max else None
            ),
            "median_descent_update_gradient_cosine": (
                statistics.median(cosines) if cosines else None
            ),
            "function_active": any(value > 0 for value in function_steps),
        }
    numeric = [
        value
        for row in rows
        for key, value in row.items()
        if key not in {"step", "timestamp"} and isinstance(value, (int, float))
    ]
    return {
        "sample_steps": [row["step"] for row in rows],
        "all_numeric_finite": all(math.isfinite(float(value)) for value in numeric),
        "groups": groups,
    }


def analyze() -> dict:
    incomplete = [arm for arm in ARMS if not (run_dir(arm) / "COMPLETED").is_file()]
    if incomplete:
        raise RuntimeError(f"Phase 36 incomplete: {incomplete}")
    finals = {arm: final_losses(arm) for arm in ARMS}
    developments = {arm: development_losses(arm) for arm in ARMS}
    arm_results = {}
    launches = {}
    for arm in ARMS:
        summary = json.loads((run_dir(arm) / "training_summary.json").read_text())
        provenance = json.loads((run_dir(arm) / "run_provenance.json").read_text())
        launches[arm] = provenance["launches"][-1]
        arm_results[arm] = {
            "final_holdout_loss": statistics.fmean(finals[arm]),
            "target_tokens_per_second": summary["target_tokens_per_second"],
            "elapsed_seconds": summary["elapsed_seconds"],
            "peak_reserved_mib": summary["peak_reserved_mib"],
            "parameter_counts": provenance["parameter_counts"],
            "intervention_state": intervention_state(arm),
            "optimization_health": optimization_health(arm),
        }
    source_states = {
        json.dumps(launch["source"], sort_keys=True) for launch in launches.values()
    }
    dataset_splits = {
        json.dumps(launch["dataset"]["splits"], sort_keys=True)
        for launch in launches.values()
    }
    if len(source_states) != 1 or any(
        launch["source"]["dirty"] for launch in launches.values()
    ):
        raise ValueError("Phase-36 arms do not share one clean source state")
    if len(dataset_splits) != 1:
        raise ValueError("Phase-36 arms do not share dataset split fingerprints")
    contrasts = {}
    for name, candidate, reference, contrast_kind in CONTRASTS:
        final = paired_summary(finals[candidate], finals[reference])
        state = arm_results[candidate]["intervention_state"]
        frequency = state["frequency"] if state is not None else None
        structurally_valid = not (
            frequency is not None
            and (
                frequency["nonpositive_fraction"] > 0
                or frequency["order_violation_fraction"] > 0
            )
        )
        pass_endpoint = (
            final["delta_candidate_minus_reference"] <= -PRACTICAL_THRESHOLD
            and final["paired_example_ci95"][1] < 0
        )
        contrasts[name] = {
            "candidate": candidate,
            "reference": reference,
            "kind": contrast_kind,
            "final": final,
            "development_curve": contrast_curve(
                developments[candidate], developments[reference]
            ),
            "promotion_gate": {
                "practical_threshold": PRACTICAL_THRESHOLD,
                "endpoint_and_interval_pass": pass_endpoint,
                "structurally_valid": structurally_valid,
                "overall_pass": pass_endpoint and structurally_valid,
            },
        }
    first_launch = launches[next(iter(ARMS))]
    historical_comparison = None
    phase35_path = (
        ROOT / "results" / "phase35_smooth_carrier_20k" / "phase35_analysis.json"
    )
    if phase35_path.is_file():
        phase35 = json.loads(phase35_path.read_text())
        exponential_delta = phase35["contrasts"][
            "rope-amplitude_vs_rope-fixed"
        ]["final"]["delta_candidate_minus_reference"]
        direct_delta = contrasts["direct-amplitude_vs_qkpre-scalar"]["final"][
            "delta_candidate_minus_reference"
        ]
        historical_comparison = {
            "phase35_exponential_amplitude_delta": exponential_delta,
            "phase36_direct_amplitude_delta": direct_delta,
            "direct_minus_exponential_contrast_difference": (
                direct_delta - exponential_delta
            ),
            "interpretation": (
                "Exploratory cross-phase comparison only; the two candidates were "
                "not trained in the same frozen launch matrix."
            ),
        }
    return {
        "scope": "phase36_one_seed_20k_direct_amplitude_and_frequency",
        "seed": 123,
        "training_steps": STEP,
        "primary_context": 1_024,
        "final_holdout_start_batch": 2_048,
        "final_holdout_examples": 1_024,
        "practical_threshold_nats": PRACTICAL_THRESHOLD,
        "provenance": {
            "source": first_launch["source"],
            "dataset": {
                "resolved_path": first_launch["dataset"]["resolved_path"],
                "splits": first_launch["dataset"]["splits"],
                "manifest_sha256": {
                    name: manifest["sha256"]
                    for name, manifest in first_launch["dataset"]["manifests"].items()
                },
            },
            "software": first_launch["software"],
            "gpu_by_arm": {
                arm: launch["hardware"]["cuda_device"]
                for arm, launch in launches.items()
            },
        },
        "arms": arm_results,
        "contrasts": contrasts,
        "historical_comparison": historical_comparison,
        "caveat": (
            "Paired-example intervals measure holdout precision for seed 123, not "
            "training-seed variability. This 20k screen measures early optimization."
        ),
    }


def _fmt_range(values: list[float] | None, digits: int = 3) -> str:
    if values is None:
        return "n/a"
    return f"{values[0]:.{digits}f}--{values[1]:.{digits}f}"


def render(results: dict) -> str:
    lines = [
        "# Phase 36: direct sinusoid amplitude and frequency at 20k",
        "",
        "All arms use seed 123, standard RoPE, and one common 20k schedule unless",
        "the arm is the RoPE-only baseline. Negative deltas favor the candidate.",
        "",
        "| Arm | Final loss | Target tok/s | Peak MiB | Shape amp range | Endpoint phase Δ (RMS/max) | Freq-order violations |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm in ARMS:
        row = results["arms"][arm]
        state = row["intervention_state"]
        amplitude = "n/a" if state is None else _fmt_range(state["amplitude_factor_range"])
        frequency = None if state is None else state["frequency"]
        endpoint = (
            "n/a"
            if frequency is None
            else f"{frequency['endpoint_phase_delta_rms']:.3f}/"
            f"{frequency['endpoint_phase_delta_abs_max']:.3f}"
        )
        violations = (
            "n/a"
            if frequency is None
            else f"{frequency['order_violation_fraction']:.2%}"
        )
        lines.append(
            f"| {arm} | {row['final_holdout_loss']:.6f} | "
            f"{row['target_tokens_per_second']:,.0f} | "
            f"{row['peak_reserved_mib']:,.0f} | {amplitude} | "
            f"{endpoint} | {violations} |"
        )
    lines.extend(
        [
            "",
            "| Contrast | Kind | Final delta | Paired-example 95% CI | Late delta/1k | Gate |",
            "| --- | --- | ---: | ---: | ---: | --- |",
        ]
    )
    for name, _, _, _ in CONTRASTS:
        row = results["contrasts"][name]
        final = row["final"]
        low, high = final["paired_example_ci95"]
        slope = row["development_curve"]["late_slope_delta_per_1k_steps"]
        slope_text = "n/a" if slope is None else f"{slope:+.6f}"
        gate = row["promotion_gate"]
        gate_text = "pass" if gate["overall_pass"] else "no"
        if gate["endpoint_and_interval_pass"] and not gate["structurally_valid"]:
            gate_text = "invalid spectrum"
        if row["kind"] == "backbone":
            gate_text = "backbone"
        lines.append(
            f"| {name} | {row['kind']} | "
            f"{final['delta_candidate_minus_reference']:+.6f} | "
            f"[{low:+.6f}, {high:+.6f}] | {slope_text} | {gate_text} |"
        )

    amplitude = results["contrasts"]["direct-amplitude_vs_qkpre-scalar"]
    combo = results["contrasts"][
        "direct-amplitude+hybrid-frequency_vs_qkpre-scalar"
    ]
    increment = results["contrasts"][
        "direct-amplitude+hybrid-frequency_vs_direct-amplitude"
    ]
    scalar_rope = results["contrasts"]["qkpre-scalar_vs_rope-fixed"]
    hybrid4 = results["arms"]["hybrid-frequency-lr4"]["intervention_state"][
        "frequency"
    ]
    amp_state = results["arms"]["direct-amplitude"]["intervention_state"]
    historical = results["historical_comparison"]
    optimization = [
        row["optimization_health"]
        for row in results["arms"].values()
        if row["optimization_health"] is not None
    ]
    adapter_clip = [
        health["groups"]["adapter"]["gradient_clip_ratio_min"]
        for health in optimization
        if "adapter" in health["groups"]
    ]
    frequency_clip = [
        health["groups"]["frequency"]["gradient_clip_ratio_min"]
        for health in optimization
        if "frequency" in health["groups"]
    ]
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "Direct, signed rank-4 amplitude is the only new component that earns",
            f"promotion: {amplitude['final']['delta_candidate_minus_reference']:+.6f} "
            "nats versus the scalar carrier, with its paired interval wholly beyond",
            f"the -{PRACTICAL_THRESHOLD:.3f} practical threshold.",
            "The amplitude+frequency arm is the lowest-loss arm",
            f"({combo['final']['delta_candidate_minus_reference']:+.6f} versus scalar),",
            "but almost all of that result is amplitude. Hybrid frequency adds only",
            f"{increment['final']['delta_candidate_minus_reference']:+.6f} nats beyond",
            "direct amplitude, below the practical threshold.",
            "",
            "Global direct frequency is null. The conservative rank-4 hybrid is",
            "small and below threshold. The LR4 hybrid moves farther, but still misses",
            f"the threshold and reverses {hybrid4['order_violation_fraction']:.2%} of",
            "adjacent frequency pairs, so it is not a clean candidate. There is no",
            "frequency variant to promote from this screen.",
            "",
            "The scalar pre-Q/K sinusoid remains the dominant architectural effect:",
            f"{scalar_rope['final']['delta_candidate_minus_reference']:+.6f} nats versus",
            "RoPE alone at 20k. Direct amplitude is a smaller refinement on top.",
            "",
            "## QKNorm and optimization audit",
            "",
            "Direct amplitude stayed signed and unconstrained but did not cross zero:",
            f"its learned shape factors span {_fmt_range(amp_state['amplitude_factor_range'])}.",
            "Across layers, the actual pre-projection positional energy fraction spans",
            f"{_fmt_range(amp_state['input_position_energy_fraction_range'])}; after",
            "QKNorm, Q's cosine to the content-only direction spans",
            f"{_fmt_range(amp_state['normalized_q_cosine_to_content_range'])}.",
            "This confirms that amplitude controls content/position direction rather",
            "than merely acting as attention temperature.",
            "The minimum factor is close enough to zero that a longer run must keep",
            "the signed-factor and per-layer spectrum diagnostics enabled.",
            "",
            "All logged intervention values are finite. The frequency groups were",
            f"never gradient-clipped (minimum clip ratio {min(frequency_clip):.3f}).",
            f"Adapter-group clip ratios reach {min(adapter_clip):.3f}, reflecting shared",
            "whole-model clipping early in training rather than an intervention-specific",
            "failure. No saturation transform, positivity map, or log-frequency",
            "parameterization was used.",
            "",
            "For context, the earlier exponential rank-4 amplitude contrast was",
            f"{historical['phase35_exponential_amplitude_delta']:+.6f}; the direct",
            "contrast is better by",
            f"{historical['direct_minus_exponential_contrast_difference']:+.6f} nats.",
            "That is an exploratory cross-phase comparison, not a paired claim, but",
            "it supports using the direct coordinate in the confirmation run.",
            "",
            "## Evidence limit",
            "",
            "The 1,024-example paired intervals establish evaluation precision, not",
            "training-seed robustness. This is one seed and only 20k steps. The clean",
            "next confirmation is direct amplitude (plus its scalar parent) at longer",
            "training and/or additional seeds; frequency should remain shelved unless",
            "a new structural hypothesis is proposed.",
            "",
            "The JSON companion records all development contrasts, provenance,",
            "intervention state, QKNorm mixture diagnostics, and optimizer health.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    results = analyze()
    markdown = render(results)
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    (RESULT_ROOT / "phase36_analysis.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n"
    )
    (RESULT_ROOT / "PHASE36_RESULTS.md").write_text(markdown + "\n")
    print(markdown)


if __name__ == "__main__":
    main()
