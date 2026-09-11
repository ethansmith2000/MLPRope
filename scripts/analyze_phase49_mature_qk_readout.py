#!/usr/bin/env python
"""Analyze the frozen 100k mature Q/K readout confirmation cohort."""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "model-output" / "position_bias_phase49_mature_qk_readout"
RESULT_ROOT = ROOT / "results" / "phase49_mature_qk_readout"
STEP = 100_000
FINAL_HOLDOUT_START = 4_096
MATERIALITY = -0.003
RUNS = {
    arm: f"phase49-{arm}-seed123-b32-s100000-h768d8"
    for arm in (
        "rope",
        "scalar-qkpre",
        "scalar-ffnmatch-r32",
        "qk-readout-r32",
        "qk-readout-r128",
    )
}
CONTRASTS = {
    "scalar-qkpre_minus_rope": ("scalar-qkpre", "rope"),
    "scalar-ffnmatch-r32_minus_scalar-qkpre": (
        "scalar-ffnmatch-r32",
        "scalar-qkpre",
    ),
    "qk-readout-r32_minus_rope": ("qk-readout-r32", "rope"),
    "qk-readout-r32_minus_scalar-qkpre": (
        "qk-readout-r32",
        "scalar-qkpre",
    ),
    "qk-readout-r32_minus_scalar-ffnmatch-r32": (
        "qk-readout-r32",
        "scalar-ffnmatch-r32",
    ),
    "qk-readout-r128_minus_scalar-qkpre": (
        "qk-readout-r128",
        "scalar-qkpre",
    ),
    "qk-readout-r128_minus_qk-readout-r32": (
        "qk-readout-r128",
        "qk-readout-r32",
    ),
}
FUNCTION_KEY = "optimization/pre_qk_sinusoid_adapter/carrier_function_step/rms"


def _run_dir(arm: str) -> Path:
    return OUTPUT_ROOT / RUNS[arm]


def _losses(arm: str) -> np.ndarray:
    detail = _run_dir(arm) / "evaluation_details" / (
        f"step_{STEP:08d}_context_001024.json"
    )
    payload = json.loads(detail.read_text())
    values = np.asarray(payload["losses"], dtype=np.float64)
    if (
        payload.get("evaluation_kind") != "final_holdout"
        or payload.get("evaluation_start_batch") != FINAL_HOLDOUT_START
        or values.shape != (1_024,)
    ):
        raise ValueError(f"Unexpected final evaluation in {detail}")
    return values


def _ci(delta: np.ndarray) -> list[float]:
    rng = np.random.default_rng(49)
    samples = []
    for _ in range(10):
        indices = rng.integers(0, delta.size, size=(2_000, delta.size))
        samples.append(delta[indices].mean(axis=1))
    return [
        float(value)
        for value in np.quantile(np.concatenate(samples), (0.025, 0.975))
    ]


def _metrics(arm: str) -> list[dict]:
    path = _run_dir(arm) / "metrics.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _optimization_rows(arm: str) -> list[dict]:
    path = _run_dir(arm) / "intervention_optimization.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _function_steps(arm: str) -> dict[int, float]:
    return {
        int(row["step"]): float(row[FUNCTION_KEY])
        for row in _optimization_rows(arm)
        if isinstance(row.get(FUNCTION_KEY), (int, float))
        and float(row[FUNCTION_KEY]) > 0
    }


def _development_losses(arm: str) -> dict[int, float]:
    return {
        int(row["step"]): float(row["eval_loss"])
        for row in _metrics(arm)
        if row.get("evaluation_kind") == "development"
    }


def _run_summary(arm: str, values: np.ndarray) -> dict:
    run_dir = _run_dir(arm)
    training = json.loads((run_dir / "training_summary.json").read_text())
    provenance = json.loads((run_dir / "run_provenance.json").read_text())
    counts = provenance.get("parameter_counts", {})
    if not counts and provenance.get("launches"):
        counts = provenance["launches"][-1].get("parameter_counts", {})
    final_record = next(
        row for row in reversed(_metrics(arm)) if row.get("evaluation_kind") == "final_holdout"
    )
    diagnostics = {
        key: value
        for key, value in final_record.items()
        if key.startswith("position/")
    }
    numeric = [
        float(value)
        for value in diagnostics.values()
        if isinstance(value, (int, float))
    ]
    return {
        "final_holdout_loss": float(values.mean()),
        "target_tokens_per_second": training["target_tokens_per_second"],
        "elapsed_seconds": training["elapsed_seconds"],
        "peak_reserved_mib": training["peak_reserved_mib"],
        "parameter_counts": counts,
        "position_diagnostics": diagnostics,
        "diagnostics_finite": all(math.isfinite(value) for value in numeric),
    }


def analyze() -> dict:
    missing = [arm for arm in RUNS if not (_run_dir(arm) / "COMPLETED").is_file()]
    if missing:
        raise RuntimeError(f"Phase 49 incomplete: {missing}")
    losses = {arm: _losses(arm) for arm in RUNS}
    runs = {arm: _run_summary(arm, losses[arm]) for arm in RUNS}
    comparisons = {}
    for name, (candidate, reference) in CONTRASTS.items():
        delta = losses[candidate] - losses[reference]
        comparisons[name] = {
            "candidate": candidate,
            "reference": reference,
            "delta": float(delta.mean()),
            "paired_example_bootstrap_ci95": _ci(delta),
        }

    r32_steps = _function_steps("qk-readout-r32")
    r128_steps = _function_steps("qk-readout-r128")
    ratios = {
        step: r128_steps[step] / r32_steps[step]
        for step in r32_steps.keys() & r128_steps.keys()
    }
    early = [ratio for step, ratio in ratios.items() if step <= 64]
    mature = [ratio for step, ratio in ratios.items() if 5_000 <= step <= 95_000]
    mature_median = statistics.median(mature)
    rank_function_match = 0.8 <= mature_median <= 1.25

    development = {arm: _development_losses(arm) for arm in RUNS}
    late_steps = [80_000, 85_000, 90_000, 95_000, 100_000]
    late_r32_deltas = [
        development["qk-readout-r32"][step]
        - development["scalar-qkpre"][step]
        for step in late_steps
    ]
    late_curve_passes = all(delta < 0 for delta in late_r32_deltas)

    r32_parent = comparisons["qk-readout-r32_minus_scalar-qkpre"]
    r32_capacity = comparisons["qk-readout-r32_minus_scalar-ffnmatch-r32"]
    r32_parent_passes = bool(
        r32_parent["delta"] <= MATERIALITY
        and r32_parent["paired_example_bootstrap_ci95"][1] < 0
    )
    r32_capacity_passes = bool(
        r32_capacity["delta"] <= MATERIALITY
        and r32_capacity["paired_example_bootstrap_ci95"][1] < 0
    )
    finite_passes = bool(
        runs["qk-readout-r32"]["diagnostics_finite"]
        and runs["qk-readout-r128"]["diagnostics_finite"]
    )
    replicate_r32 = bool(
        r32_parent_passes
        and r32_capacity_passes
        and late_curve_passes
        and finite_passes
    )

    scalar_total = runs["scalar-qkpre"]["parameter_counts"]["total"]
    r32_increment = runs["qk-readout-r32"]["parameter_counts"]["total"] - scalar_total
    ffn_increment = (
        runs["scalar-ffnmatch-r32"]["parameter_counts"]["total"] - scalar_total
    )
    rank_comparison = comparisons["qk-readout-r128_minus_qk-readout-r32"]
    material_rank128_win = bool(
        rank_function_match
        and rank_comparison["delta"] <= MATERIALITY
        and rank_comparison["paired_example_bootstrap_ci95"][1] < 0
    )

    return {
        "scope": "phase49_mature_qk_readout",
        "context": 1_024,
        "sequence_batch": 32,
        "training_steps": STEP,
        "seed": 123,
        "final_holdout": {
            "start_batch": FINAL_HOLDOUT_START,
            "blocks": 1_024,
            "previously_unused_by_repository_configs": True,
        },
        "runs": runs,
        "comparisons": comparisons,
        "parameter_match": {
            "rank32_increment_over_scalar": r32_increment,
            "ffn_increment_over_scalar": ffn_increment,
            "absolute_error": abs(ffn_increment - r32_increment),
        },
        "rank_function_step_scale_r128_over_r32": {
            "early_through_step_64_median": statistics.median(early),
            "step_5000_to_95000_median": mature_median,
            "mature_min": min(mature),
            "mature_max": max(mature),
            "passes_predeclared_match": rank_function_match,
        },
        "late_curve": {
            "steps": late_steps,
            "rank32_minus_scalar_deltas": late_r32_deltas,
            "all_negative": late_curve_passes,
        },
        "gates": {
            "materiality_delta_at_most": MATERIALITY,
            "paired_interval_upper_below": 0.0,
            "rank32_beats_scalar": r32_parent_passes,
            "rank32_beats_parameter_matched_ffn": r32_capacity_passes,
            "rank32_late_curve_noncollapsing": late_curve_passes,
            "candidate_diagnostics_finite": finite_passes,
            "replicate_rank32_across_training_seeds": replicate_r32,
            "rank128_materially_beats_rank32_at_matched_function_steps": material_rank128_win,
        },
        "caveat": (
            "This mature-horizon cohort uses a newly frozen holdout and exact "
            "paired data order and initialization, but only one training seed. "
            "Paired-example intervals do not estimate training-seed uncertainty."
        ),
    }


def render(results: dict) -> str:
    lines = [
        "# Phase 49: mature Q/K readout confirmation",
        "",
        "| Arm | Final NLL | Total params | Position params | tokens/s |",
        "|---|---:|---:|---:|---:|",
    ]
    for arm, run in results["runs"].items():
        counts = run["parameter_counts"]
        lines.append(
            f"| {arm} | {run['final_holdout_loss']:.6f} | "
            f"{counts['total']} | {counts['position_params']} | "
            f"{run['target_tokens_per_second']:.0f} |"
        )
    lines.extend(
        [
            "",
            "## Paired contrasts on the new holdout",
            "",
            "Negative deltas favor the first named arm.",
            "",
            "| Contrast | Delta | Paired 95% interval |",
            "|---|---:|---:|",
        ]
    )
    for name, comparison in results["comparisons"].items():
        low, high = comparison["paired_example_bootstrap_ci95"]
        lines.append(
            f"| {name} | {comparison['delta']:+.6f} | "
            f"[{low:+.6f}, {high:+.6f}] |"
        )
    match = results["parameter_match"]
    scale = results["rank_function_step_scale_r128_over_r32"]
    gates = results["gates"]
    lines.extend(
        [
            "",
            "## Controls and decisions",
            "",
            f"The FFN control differs from the rank-32 candidate's incremental "
            f"parameter count by {match['absolute_error']} parameters.",
            f"Median rank-128/rank-32 carrier-step ratio from 5k--95k: "
            f"{scale['step_5000_to_95000_median']:.3f} "
            f"(match: {'yes' if scale['passes_predeclared_match'] else 'no'}).",
            f"Rank-32 replication gate: "
            f"{'pass' if gates['replicate_rank32_across_training_seeds'] else 'fail'}.",
            f"Material matched-step rank-128 advantage: "
            f"{'yes' if gates['rank128_materially_beats_rank32_at_matched_function_steps'] else 'no'}.",
            "",
            results["caveat"],
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    results = analyze()
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    (RESULT_ROOT / "summary.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n"
    )
    report = render(results)
    (RESULT_ROOT / "REPORT.md").write_text(report)
    print(report)
