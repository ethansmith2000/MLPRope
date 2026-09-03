#!/usr/bin/env python
"""Analyze the completed one-seed Phase-33 long-run ladder."""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "model-output" / "position_bias_phase33_static_qkpre_200k"
RESULT_ROOT = ROOT / "results" / "phase33_static_qkpre_200k"
SUFFIX = "-seed123-s200000-h768d8"
ARMS = (
    "rope-fixed",
    "qkpre-tied-nope",
    "qkpre-tied-rope",
    "qkpre-split-scalar-rope",
    "qkpre-pair-amplitude-rope",
    "qkpre-pair-polar-rope",
)
CONTRASTS = (
    ("qkpre-tied-nope_vs_rope-fixed", "qkpre-tied-nope", "rope-fixed"),
    ("qkpre-tied-rope_vs_rope-fixed", "qkpre-tied-rope", "rope-fixed"),
    ("rope_contribution", "qkpre-tied-rope", "qkpre-tied-nope"),
    ("split-scalar_vs_tied", "qkpre-split-scalar-rope", "qkpre-tied-rope"),
    (
        "pair-amplitude_vs_split-scalar",
        "qkpre-pair-amplitude-rope",
        "qkpre-split-scalar-rope",
    ),
    (
        "pair-polar_vs_pair-amplitude",
        "qkpre-pair-polar-rope",
        "qkpre-pair-amplitude-rope",
    ),
)


def run_dir(arm: str) -> Path:
    return OUTPUT_ROOT / f"phase33-{arm}{SUFFIX}"


def losses(arm: str) -> list[float]:
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


def analyze() -> dict:
    incomplete = [arm for arm in ARMS if not (run_dir(arm) / "COMPLETED").is_file()]
    if incomplete:
        raise RuntimeError(f"Phase 33 incomplete: {incomplete}")
    by_arm = {arm: losses(arm) for arm in ARMS}
    arm_results = {}
    for arm in ARMS:
        summary = json.loads((run_dir(arm) / "training_summary.json").read_text())
        arm_results[arm] = {
            "final_holdout_loss": statistics.fmean(by_arm[arm]),
            "target_tokens_per_second": summary["target_tokens_per_second"],
            "elapsed_seconds": summary["elapsed_seconds"],
        }
    contrast_results = {
        name: {
            "candidate": candidate,
            "reference": reference,
            **paired_summary(by_arm[candidate], by_arm[reference]),
        }
        for name, candidate, reference in CONTRASTS
    }
    return {
        "scope": "phase33_one_seed_200k_static_qkpre_ladder",
        "seed": 123,
        "training_steps": 200_000,
        "training_tokens": 1_638_236_160,
        "primary_context": 1_024,
        "final_holdout_start_batch": 2_048,
        "final_holdout_examples": 1_024,
        "arms": arm_results,
        "contrasts": contrast_results,
        "interpretation": {
            "tied_carrier_survives_long_training": True,
            "rope_remains_material_with_carrier": True,
            "qk_or_pairwise_static_adapter_earned_followup": False,
        },
        "caveat": (
            "Paired-example intervals measure holdout precision for seed 123, not "
            "training-seed variability. Replicate only candidates that clear the "
            "next one-seed scope screen."
        ),
    }


def render(results: dict) -> str:
    lines = [
        "# Phase 33: static pre-Q/K adapter at 200k",
        "",
        "All arms use seed 123 and a common 200k learning-rate horizon. Final loss",
        "uses the disjoint 1,024-example holdout beginning at validation batch 2,048.",
        "Negative deltas favor the candidate.",
        "",
        "| Arm | Final loss | Target tok/s |",
        "| --- | ---: | ---: |",
    ]
    for arm in ARMS:
        row = results["arms"][arm]
        lines.append(
            f"| {arm} | {row['final_holdout_loss']:.6f} | "
            f"{row['target_tokens_per_second']:,.0f} |"
        )
    lines.extend(
        [
            "",
            "| Contrast | Delta | Paired-example 95% CI |",
            "| --- | ---: | ---: |",
        ]
    )
    for name, _, _ in CONTRASTS:
        row = results["contrasts"][name]
        low, high = row["paired_example_ci95"]
        lines.append(
            f"| {name} | {row['delta_candidate_minus_reference']:+.6f} | "
            f"[{low:+.6f}, {high:+.6f}] |"
        )
    lines.extend(
        [
            "",
            "The tied carrier remains a large improvement and fixed RoPE remains",
            "material on top of it. Separate Q/K gains, pairwise amplitudes, and",
            "pairwise phases are all null at this horizon, so Phase 34 retains only",
            "the tied carrier and tests a globally shared frequency bank.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    results = analyze()
    markdown = render(results)
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    (RESULT_ROOT / "phase33_analysis.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n"
    )
    (RESULT_ROOT / "PHASE33_RESULTS.md").write_text(markdown)
    print(markdown)


if __name__ == "__main__":
    main()

