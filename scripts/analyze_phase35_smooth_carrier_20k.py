#!/usr/bin/env python
"""Analyze the one-seed Phase-35 smooth-carrier screen."""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "model-output" / "position_bias_phase35_smooth_carrier_20k"
RESULT_ROOT = ROOT / "results" / "phase35_smooth_carrier_20k"
STEP = 20_000
ARMS = {
    "rope-fixed": "phase35-rope-tied-fixed-seed123-s20000-h768d8",
    "rope-amplitude": (
        "phase35-rope-tied-smooth-amplitude-r4-seed123-s20000-h768d8"
    ),
    "rope-polar": "phase35-rope-tied-smooth-polar-r4-seed123-s20000-h768d8",
    "rope-split-polar": (
        "phase35-rope-split-smooth-polar-r4-seed123-s20000-h768d8"
    ),
    "nope-fixed": "phase35-nope-tied-fixed-seed123-s20000-h768d8",
    "nope-amplitude": (
        "phase35-nope-tied-smooth-amplitude-r4-seed123-s20000-h768d8"
    ),
    "nope-polar": "phase35-nope-tied-smooth-polar-r4-seed123-s20000-h768d8",
    "nope-split-polar": (
        "phase35-nope-split-smooth-polar-r4-seed123-s20000-h768d8"
    ),
}
PRIMARY_CONTRASTS = (
    ("rope-amplitude_vs_rope-fixed", "rope-amplitude", "rope-fixed"),
    ("rope-polar_vs_rope-amplitude", "rope-polar", "rope-amplitude"),
    ("rope-split-polar_vs_rope-polar", "rope-split-polar", "rope-polar"),
    ("nope-amplitude_vs_nope-fixed", "nope-amplitude", "nope-fixed"),
    ("nope-polar_vs_nope-amplitude", "nope-polar", "nope-amplitude"),
    ("nope-split-polar_vs_nope-polar", "nope-split-polar", "nope-polar"),
)
BACKBONE_CONTRASTS = tuple(
    (f"rope-{mode}_vs_nope-{mode}", f"rope-{mode}", f"nope-{mode}")
    for mode in ("fixed", "amplitude", "polar", "split-polar")
)
CONTRASTS = PRIMARY_CONTRASTS + BACKBONE_CONTRASTS


def run_dir(arm: str) -> Path:
    return OUTPUT_ROOT / ARMS[arm]


def _detail_payload(path: Path) -> dict:
    payload = json.loads(path.read_text())
    return payload


def final_losses(arm: str) -> list[float]:
    path = run_dir(arm) / "evaluation_details" / (
        f"step_{STEP:08d}_context_001024.json"
    )
    payload = _detail_payload(path)
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
        payload = _detail_payload(path)
        if payload["evaluation_start_batch"] != 0:
            raise ValueError(f"Unexpected development start in {path}")
        by_step[int(payload["step"])] = [float(x) for x in payload["losses"]]
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


def optimization_health(arm: str) -> dict:
    path = run_dir(arm) / "intervention_optimization.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    prefix = "optimization/pre_qk_sinusoid_adapter"
    numeric = [
        value
        for row in rows
        for key, value in row.items()
        if key not in {"step", "timestamp"} and isinstance(value, (int, float))
    ]
    populated = [
        row for row in rows if row.get(f"{prefix}/parameter_update/l2", 0) > 0
    ]
    last = rows[-1]
    descent_cosines = [
        row[f"{prefix}/descent_update_gradient_cosine"]
        for row in populated
        if row.get(f"{prefix}/descent_update_gradient_cosine") is not None
    ]
    return {
        "sample_steps": [row["step"] for row in rows],
        "all_numeric_finite": all(math.isfinite(float(value)) for value in numeric),
        "last_raw_gradient_l2": last[f"{prefix}/raw_gradient/l2"],
        "last_parameter_update_l2": last[f"{prefix}/parameter_update/l2"],
        "last_carrier_function_step_rms": last[
            f"{prefix}/carrier_function_step/rms"
        ],
        "last_gradient_clip_ratio": last[f"{prefix}/gradient_clip_ratio"],
        "last_sqrt_second_moment_max_to_rms": last[
            f"{prefix}/adam_after/sqrt_second_moment_max_to_rms"
        ],
        "last_descent_update_gradient_cosine": last[
            f"{prefix}/descent_update_gradient_cosine"
        ],
        "minimum_descent_update_gradient_cosine": (
            min(descent_cosines) if descent_cosines else None
        ),
        "function_active": any(
            row.get(f"{prefix}/carrier_function_step/rms", 0) > 0
            for row in populated
        ),
    }


def carrier_profiles(arm: str) -> dict:
    path = run_dir(arm) / "position_profiles" / f"step_{STEP:08d}.pt"
    payload = torch.load(path, map_location="cpu", weights_only=True)
    profiles = {
        key: value.detach().float()
        for key, value in payload["profiles"].items()
        if "/qk_preprojection/" in key
    }
    serialized = {key: value.tolist() for key, value in profiles.items()}
    q_amplitude = [
        value
        for key, value in profiles.items()
        if key.endswith("/log_amplitude_q")
    ]
    k_amplitude = [
        profiles[key.replace("/log_amplitude_q", "/log_amplitude_k")]
        for key in profiles
        if key.endswith("/log_amplitude_q")
    ]
    q_phase = [value for key, value in profiles.items() if key.endswith("/phase_q")]
    k_phase = [profiles[key.replace("/phase_q", "/phase_k")] for key in profiles if key.endswith("/phase_q")]

    def flat(values: list[torch.Tensor]) -> torch.Tensor:
        return torch.cat([value.reshape(-1) for value in values])

    qa, ka, qp, kp = map(flat, (q_amplitude, k_amplitude, q_phase, k_phase))
    return {
        "effective_profiles": serialized,
        "log_amplitude_abs_max": max(qa.abs().max().item(), ka.abs().max().item()),
        "phase_abs_max": max(qp.abs().max().item(), kp.abs().max().item()),
        "qk_log_amplitude_diff_rms": (qa - ka).square().mean().sqrt().item(),
        "qk_phase_diff_rms": (qp - kp).square().mean().sqrt().item(),
    }


def analyze() -> dict:
    incomplete = [arm for arm in ARMS if not (run_dir(arm) / "COMPLETED").is_file()]
    if incomplete:
        raise RuntimeError(f"Phase 35 incomplete: {incomplete}")
    finals = {arm: final_losses(arm) for arm in ARMS}
    developments = {arm: development_losses(arm) for arm in ARMS}
    arm_results = {}
    for arm in ARMS:
        summary = json.loads((run_dir(arm) / "training_summary.json").read_text())
        provenance = json.loads((run_dir(arm) / "run_provenance.json").read_text())
        arm_results[arm] = {
            "final_holdout_loss": statistics.fmean(finals[arm]),
            "target_tokens_per_second": summary["target_tokens_per_second"],
            "elapsed_seconds": summary["elapsed_seconds"],
            "peak_reserved_mib": summary["peak_reserved_mib"],
            "parameter_counts": provenance["parameter_counts"],
            "optimization_health": optimization_health(arm),
            "carrier_profiles": carrier_profiles(arm),
        }
    contrast_results = {}
    primary_names = {name for name, _, _ in PRIMARY_CONTRASTS}
    for name, candidate, reference in CONTRASTS:
        final = paired_summary(finals[candidate], finals[reference])
        health = arm_results[candidate]["optimization_health"]
        contrast_results[name] = {
            "candidate": candidate,
            "reference": reference,
            "final": final,
            "development_curve": contrast_curve(
                developments[candidate], developments[reference]
            ),
            "promotion_gate": {
                "is_primary_intervention_contrast": name in primary_names,
                "endpoint_and_interval_pass": (
                    name in primary_names
                    and final["delta_candidate_minus_reference"] <= -0.003
                    and final["paired_example_ci95"][1] < 0
                ),
                "optimization_active_and_finite": (
                    health["all_numeric_finite"] and health["function_active"]
                ),
                "late_curve_requires_review": name in primary_names,
            },
        }
    return {
        "scope": "phase35_one_seed_20k_smooth_pre_qk_carrier",
        "seed": 123,
        "training_steps": STEP,
        "smooth_rank": 4,
        "primary_context": 1_024,
        "final_holdout_start_batch": 2_048,
        "final_holdout_examples": 1_024,
        "arms": arm_results,
        "contrasts": contrast_results,
        "caveat": (
            "Paired-example intervals measure holdout precision for seed 123, not "
            "training-seed variability. This 20k screen measures early optimization."
        ),
    }


def render(results: dict) -> str:
    lines = [
        "# Phase 35: smooth sinusoidal carrier at 20k",
        "",
        "All arms use seed 123 and one common 20k schedule. Negative deltas favor",
        "the candidate.",
        "",
        "| Arm | Final loss | Target tok/s | Peak MiB | Amp max | Phase max |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm in ARMS:
        row = results["arms"][arm]
        profile = row["carrier_profiles"]
        lines.append(
            f"| {arm} | {row['final_holdout_loss']:.6f} | "
            f"{row['target_tokens_per_second']:,.0f} | "
            f"{row['peak_reserved_mib']:,.0f} | "
            f"{profile['log_amplitude_abs_max']:.4f} | "
            f"{profile['phase_abs_max']:.4f} |"
        )
    lines.extend(
        [
            "",
            "| Contrast | Final delta | Paired-example 95% CI | Late delta/1k | Gate |",
            "| --- | ---: | ---: | ---: | --- |",
        ]
    )
    for name, _, _ in CONTRASTS:
        row = results["contrasts"][name]
        final = row["final"]
        low, high = final["paired_example_ci95"]
        slope = row["development_curve"]["late_slope_delta_per_1k_steps"]
        slope_text = "n/a" if slope is None else f"{slope:+.6f}"
        gate = row["promotion_gate"]
        gate_text = "pass" if gate["endpoint_and_interval_pass"] else "no"
        if not gate["is_primary_intervention_contrast"]:
            gate_text = "backbone"
        lines.append(
            f"| {name} | {final['delta_candidate_minus_reference']:+.6f} | "
            f"[{low:+.6f}, {high:+.6f}] | {slope_text} | {gate_text} |"
        )
    lines.extend(
        [
            "",
            "The JSON companion contains every development point, complete effective",
            "amplitude/phase profiles, parameter counts, and optimization-health",
            "summaries. Inspect late curves before promoting any endpoint pass.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    results = analyze()
    markdown = render(results)
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    (RESULT_ROOT / "phase35_analysis.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n"
    )
    (RESULT_ROOT / "PHASE35_RESULTS.md").write_text(markdown + "\n")
    print(markdown)


if __name__ == "__main__":
    main()
