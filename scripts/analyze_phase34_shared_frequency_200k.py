#!/usr/bin/env python
"""Analyze the completed one-seed Phase-34 shared-frequency screen."""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "model-output" / "position_bias_phase34_shared_frequency_200k"
RESULT_ROOT = ROOT / "results" / "phase34_shared_frequency_200k"
SUFFIX = "-seed123-s200000-h768d8"
ARMS = (
    "rope-fixed",
    "rope-global-log",
    "qkpre-fixed",
    "qkpre-frequency-log",
    "qkpre-frequency-horizon",
)
CONTRASTS = (
    ("rope-global-log_vs_rope-fixed", "rope-global-log", "rope-fixed"),
    ("qkpre-frequency-log_vs_qkpre-fixed", "qkpre-frequency-log", "qkpre-fixed"),
    (
        "qkpre-frequency-horizon_vs_qkpre-fixed",
        "qkpre-frequency-horizon",
        "qkpre-fixed",
    ),
    (
        "qkpre-frequency-horizon_vs_qkpre-frequency-log",
        "qkpre-frequency-horizon",
        "qkpre-frequency-log",
    ),
)


def run_dir(arm: str) -> Path:
    return OUTPUT_ROOT / f"phase34-{arm}{SUFFIX}"


def final_losses(arm: str) -> list[float]:
    path = run_dir(arm) / "evaluation_details/step_00200000_context_001024.json"
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
    detail_dir = run_dir(arm) / "evaluation_details"
    by_step = {}
    for path in detail_dir.glob("step_*_development_context_001024.json"):
        payload = json.loads(path.read_text())
        if payload["evaluation_kind"] != "development":
            continue
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
    xs = [step / 10_000 for step, _ in points]
    ys = [value for _, value in points]
    x_mean = statistics.fmean(xs)
    y_mean = statistics.fmean(ys)
    denominator = sum((value - x_mean) ** 2 for value in xs)
    if denominator == 0:
        return None
    return sum(
        (x - x_mean) * (y - y_mean) for x, y in zip(xs, ys, strict=True)
    ) / denominator


def contrast_curve(
    candidate: dict[int, list[float]],
    reference: dict[int, list[float]],
) -> dict:
    common_steps = sorted(set(candidate) & set(reference))
    points = {
        str(step): paired_summary(candidate[step], reference[step])
        for step in common_steps
    }
    late = [
        (step, points[str(step)]["delta_candidate_minus_reference"])
        for step in common_steps
        if step >= 100_000
    ]
    return {
        "points": points,
        "late_slope_delta_per_10k_steps": _linear_slope(late),
        "late_window_start_step": 100_000,
    }


def _last_metrics(arm: str) -> dict:
    rows = [
        json.loads(line)
        for line in (run_dir(arm) / "metrics.jsonl").read_text().splitlines()
        if line.strip()
    ]
    matches = [
        row
        for row in rows
        if row.get("step") == 200_000
        and row.get("evaluation_kind") == "final_holdout"
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one final metrics row for {arm}, got {len(matches)}")
    return matches[0]


def frequency_health(arm: str) -> dict | None:
    metrics = _last_metrics(arm)
    prefixes = (
        "position/shared_rope_frequency/",
        "position/shared_qkpre_frequency/",
    )
    frequency_metrics = {
        key: value
        for key, value in metrics.items()
        if key.startswith(prefixes)
    }
    if not frequency_metrics:
        return None
    profile_path = run_dir(arm) / "position_profiles/step_00200000.pt"
    payload = torch.load(profile_path, map_location="cpu", weights_only=True)
    profiles = {
        key: value.detach().float().tolist()
        for key, value in payload["profiles"].items()
        if key.startswith(("shared_rope_frequency/", "shared_qkpre_frequency/"))
    }
    nonpositive = next(
        value
        for key, value in frequency_metrics.items()
        if key.endswith("frequency_nonpositive_fraction")
    )
    order_violations = next(
        value
        for key, value in frequency_metrics.items()
        if key.endswith("frequency_order_violation_fraction")
    )
    finite = all(math.isfinite(float(value)) for value in frequency_metrics.values())
    return {
        "metrics": frequency_metrics,
        "profiles": profiles,
        "numerically_usable": finite and nonpositive == 0 and order_violations == 0,
    }


def analyze() -> dict:
    incomplete = [arm for arm in ARMS if not (run_dir(arm) / "COMPLETED").is_file()]
    if incomplete:
        raise RuntimeError(f"Phase 34 incomplete: {incomplete}")
    final_by_arm = {arm: final_losses(arm) for arm in ARMS}
    development_by_arm = {arm: development_losses(arm) for arm in ARMS}
    arm_results = {}
    for arm in ARMS:
        summary = json.loads((run_dir(arm) / "training_summary.json").read_text())
        provenance = json.loads((run_dir(arm) / "run_provenance.json").read_text())
        arm_results[arm] = {
            "final_holdout_loss": statistics.fmean(final_by_arm[arm]),
            "target_tokens_per_second": summary["target_tokens_per_second"],
            "elapsed_seconds": summary["elapsed_seconds"],
            "peak_reserved_mib": summary["peak_reserved_mib"],
            "parameter_counts": provenance["parameter_counts"],
            "frequency_health": frequency_health(arm),
        }
    contrast_results = {}
    for name, candidate, reference in CONTRASTS:
        final = paired_summary(final_by_arm[candidate], final_by_arm[reference])
        curve = contrast_curve(
            development_by_arm[candidate],
            development_by_arm[reference],
        )
        endpoint_pass = (
            final["delta_candidate_minus_reference"] <= -0.002
            and final["paired_example_ci95"][1] < 0
        )
        health = arm_results[candidate]["frequency_health"]
        contrast_results[name] = {
            "candidate": candidate,
            "reference": reference,
            "final": final,
            "development_curve": curve,
            "promotion_gate": {
                "endpoint_and_interval_pass": endpoint_pass,
                "spectrum_pass": health is None or health["numerically_usable"],
                "late_curve_requires_review": True,
            },
        }
    return {
        "scope": "phase34_one_seed_200k_globally_shared_frequency",
        "seed": 123,
        "training_steps": 200_000,
        "primary_context": 1_024,
        "final_holdout_start_batch": 2_048,
        "final_holdout_examples": 1_024,
        "arms": arm_results,
        "contrasts": contrast_results,
        "caveat": (
            "Paired-example intervals measure holdout precision for seed 123, not "
            "training-seed variability. The saved late slope is descriptive; inspect "
            "the full paired curve before promoting an arm."
        ),
    }


def render(results: dict) -> str:
    lines = [
        "# Phase 34: globally shared frequency at 200k",
        "",
        "All arms use seed 123 and a common 200k learning-rate horizon. Negative",
        "deltas favor the candidate.",
        "",
        "| Arm | Final loss | Target tok/s | Peak MiB |",
        "| --- | ---: | ---: | ---: |",
    ]
    for arm in ARMS:
        row = results["arms"][arm]
        lines.append(
            f"| {arm} | {row['final_holdout_loss']:.6f} | "
            f"{row['target_tokens_per_second']:,.0f} | {row['peak_reserved_mib']:,.0f} |"
        )
    lines.extend(
        [
            "",
            "| Contrast | Final delta | Paired-example 95% CI | Late delta/10k |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for name, _, _ in CONTRASTS:
        row = results["contrasts"][name]
        final = row["final"]
        low, high = final["paired_example_ci95"]
        slope = row["development_curve"]["late_slope_delta_per_10k_steps"]
        slope_text = "n/a" if slope is None else f"{slope:+.6f}"
        lines.append(
            f"| {name} | {final['delta_candidate_minus_reference']:+.6f} | "
            f"[{low:+.6f}, {high:+.6f}] | {slope_text} |"
        )
    lines.extend(
        [
            "",
            "The JSON companion contains every development-curve point, the complete",
            "learned spectra, spectral-health diagnostics, parameter counts, and the",
            "mechanical endpoint/interval gate. Late-curve promotion remains an explicit",
            "review rather than an arbitrary hidden threshold.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    results = analyze()
    markdown = render(results)
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    (RESULT_ROOT / "phase34_analysis.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n"
    )
    (RESULT_ROOT / "PHASE34_RESULTS.md").write_text(markdown)
    print(markdown)


if __name__ == "__main__":
    main()
