#!/usr/bin/env python
"""Analyze the three-seed mature rank-32 replication."""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PHASE49_ROOT = ROOT / "model-output" / "position_bias_phase49_mature_qk_readout"
PHASE50_ROOT = (
    ROOT / "model-output" / "position_bias_phase50_training_seed_replication"
)
RESULT_ROOT = ROOT / "results" / "phase50_training_seed_replication"
STEP = 100_000
FINAL_HOLDOUT_START = 4_096
SEEDS = (123, 456, 789)
ARMS = ("rope", "scalar-qkpre", "scalar-ffnmatch-r32", "qk-readout-r32")
MATERIALITY = -0.003
T_CRITICAL_95_DF2 = 4.302652729911275
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
}


def _run_dir(seed: int, arm: str) -> Path:
    if seed == 123:
        return PHASE49_ROOT / f"phase49-{arm}-seed123-b32-s100000-h768d8"
    return PHASE50_ROOT / f"phase50-{arm}-seed{seed}-b32-s100000-h768d8"


def _metrics(seed: int, arm: str) -> list[dict]:
    path = _run_dir(seed, arm) / "metrics.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _losses(seed: int, arm: str) -> np.ndarray:
    detail = _run_dir(seed, arm) / "evaluation_details" / (
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
    if not np.isfinite(values).all():
        raise ValueError(f"Non-finite final evaluation in {detail}")
    return values


def _paired_example_ci(delta: np.ndarray, seed: int) -> list[float]:
    rng = np.random.default_rng(50_000 + seed)
    samples = []
    for _ in range(10):
        indices = rng.integers(0, delta.size, size=(2_000, delta.size))
        samples.append(delta[indices].mean(axis=1))
    return [
        float(value)
        for value in np.quantile(np.concatenate(samples), (0.025, 0.975))
    ]


def _contiguous_block_ci(
    delta: np.ndarray,
    seed: int,
    block_length: int = 32,
) -> list[float]:
    if delta.size % block_length:
        raise ValueError("Evaluation window must divide the bootstrap block length")
    block_means = delta.reshape(-1, block_length).mean(axis=1)
    rng = np.random.default_rng(50_500 + seed)
    samples = []
    for _ in range(10):
        indices = rng.integers(
            0,
            block_means.size,
            size=(2_000, block_means.size),
        )
        samples.append(block_means[indices].mean(axis=1))
    return [
        float(value)
        for value in np.quantile(np.concatenate(samples), (0.025, 0.975))
    ]


def _numeric_values(value):
    if isinstance(value, bool):
        return
    if isinstance(value, (int, float)):
        yield float(value)
    elif isinstance(value, dict):
        for nested in value.values():
            yield from _numeric_values(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _numeric_values(nested)


def _diagnostics_status(seed: int, arm: str) -> dict:
    metrics = _metrics(seed, arm)
    final = next(
        row
        for row in reversed(metrics)
        if row.get("evaluation_kind") == "final_holdout"
    )
    position_values = [
        float(value)
        for key, value in final.items()
        if key.startswith("position/") and isinstance(value, (int, float))
    ]
    metrics_values = [value for row in metrics for value in _numeric_values(row)]
    optimization_path = _run_dir(seed, arm) / "intervention_optimization.jsonl"
    optimization_rows = (
        [
            json.loads(line)
            for line in optimization_path.read_text().splitlines()
            if line
        ]
        if optimization_path.is_file()
        else []
    )
    optimization_values = [
        value
        for row in optimization_rows
        for value in _numeric_values(row)
    ]
    expects_position = arm != "rope"
    expects_optimization = arm != "rope"
    return {
        "finite": bool(
            metrics_values
            and all(math.isfinite(value) for value in metrics_values)
            and (not expects_position or position_values)
            and all(math.isfinite(value) for value in position_values)
            and (not expects_optimization or optimization_rows)
            and (not expects_optimization or optimization_values)
            and all(math.isfinite(value) for value in optimization_values)
        ),
        "metric_numeric_values_checked": len(metrics_values),
        "final_position_values_checked": len(position_values),
        "optimization_rows_checked": len(optimization_rows),
        "optimization_numeric_values_checked": len(optimization_values),
    }


def _seed_interval(values: list[float]) -> list[float]:
    mean = statistics.mean(values)
    standard_error = statistics.stdev(values) / math.sqrt(len(values))
    radius = T_CRITICAL_95_DF2 * standard_error
    return [mean - radius, mean + radius]


def analyze() -> dict:
    missing = [
        str(_run_dir(seed, arm))
        for seed in SEEDS
        for arm in ARMS
        if not (_run_dir(seed, arm) / "COMPLETED").is_file()
    ]
    if missing:
        raise RuntimeError(f"Phase 50 incomplete: {missing}")

    losses = {
        seed: {arm: _losses(seed, arm) for arm in ARMS}
        for seed in SEEDS
    }
    runs = {
        str(seed): {
            arm: {
                "final_holdout_loss": float(losses[seed][arm].mean()),
                "diagnostics": _diagnostics_status(seed, arm),
            }
            for arm in ARMS
        }
        for seed in SEEDS
    }

    comparisons = {}
    for name, (candidate, reference) in CONTRASTS.items():
        by_seed = {}
        deltas = []
        for seed in SEEDS:
            delta = losses[seed][candidate] - losses[seed][reference]
            mean_delta = float(delta.mean())
            deltas.append(mean_delta)
            by_seed[str(seed)] = {
                "delta": mean_delta,
                "paired_example_bootstrap_ci95": _paired_example_ci(delta, seed),
                "contiguous_block_32_bootstrap_ci95": _contiguous_block_ci(
                    delta,
                    seed,
                ),
            }
        fresh_deltas = deltas[1:]
        comparisons[name] = {
            "candidate": candidate,
            "reference": reference,
            "by_seed": by_seed,
            "seed_mean_delta": statistics.mean(deltas),
            "seed_sample_std": statistics.stdev(deltas),
            "seed_t_interval95": _seed_interval(deltas),
            "all_three_negative": all(delta < 0 for delta in deltas),
            "fresh_seed_mean_delta": statistics.mean(fresh_deltas),
            "fresh_seeds_both_negative": all(delta < 0 for delta in fresh_deltas),
        }

    parent = comparisons["qk-readout-r32_minus_scalar-qkpre"]
    capacity = comparisons["qk-readout-r32_minus_scalar-ffnmatch-r32"]
    finite = all(
        run["diagnostics"]["finite"]
        for seed_runs in runs.values()
        for run in seed_runs.values()
    )
    parent_gate = bool(
        parent["all_three_negative"] and parent["seed_mean_delta"] <= MATERIALITY
    )
    capacity_gate = bool(
        capacity["all_three_negative"]
        and capacity["seed_mean_delta"] <= MATERIALITY
    )

    return {
        "scope": "phase50_training_seed_replication",
        "context": 1_024,
        "sequence_batch": 32,
        "training_steps": STEP,
        "seeds": list(SEEDS),
        "final_holdout": {"start_batch": FINAL_HOLDOUT_START, "blocks": 1_024},
        "runs": runs,
        "comparisons": comparisons,
        "gates": {
            "materiality_delta_at_most": MATERIALITY,
            "rank32_beats_scalar_in_all_seeds_with_material_mean": parent_gate,
            "rank32_beats_ffn_control_in_all_seeds_with_material_mean": capacity_gate,
            "all_diagnostics_finite": finite,
            "replication_passes": bool(parent_gate and capacity_gate and finite),
        },
        "inference_note": (
            "The primary uncertainty unit is the training seed. The paired-example "
            "and contiguous-block intervals describe evaluation-sample precision "
            "within each seed; the three-seed t interval is descriptive and "
            "necessarily low-powered. Seed 123 selected this candidate, so the "
            "seeds-456/789 mean is also reported separately as fresh confirmation "
            "rather than obscured by the selected seed."
        ),
    }


def render(results: dict) -> str:
    lines = [
        "# Phase 50: mature training-seed replication",
        "",
        "## Final holdout NLL",
        "",
        "| Arm | Seed 123 | Seed 456 | Seed 789 | Mean | Sample SD |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        values = [results["runs"][str(seed)][arm]["final_holdout_loss"] for seed in SEEDS]
        lines.append(
            f"| {arm} | {values[0]:.6f} | {values[1]:.6f} | {values[2]:.6f} | "
            f"{statistics.mean(values):.6f} | {statistics.stdev(values):.6f} |"
        )

    lines.extend(
        [
            "",
            "## Seed-level contrasts",
            "",
            "Negative deltas favor the first named arm.",
            "",
            "| Contrast | Seed 123 | Seed 456 | Seed 789 | Mean | Sample SD | Seed t interval |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name, comparison in results["comparisons"].items():
        deltas = [comparison["by_seed"][str(seed)]["delta"] for seed in SEEDS]
        low, high = comparison["seed_t_interval95"]
        lines.append(
            f"| {name} | {deltas[0]:+.6f} | {deltas[1]:+.6f} | "
            f"{deltas[2]:+.6f} | {comparison['seed_mean_delta']:+.6f} | "
            f"{comparison['seed_sample_std']:.6f} | [{low:+.6f}, {high:+.6f}] |"
        )

    lines.extend(
        [
            "",
            "## Fresh-seed sensitivity",
            "",
            "Seed 123 was used to select the rank-32 candidate. The table below",
            "therefore isolates the two subsequently registered seeds.",
            "",
            "| Contrast | Mean over seeds 456/789 | Both negative |",
            "|---|---:|---:|",
        ]
    )
    for name, comparison in results["comparisons"].items():
        lines.append(
            f"| {name} | {comparison['fresh_seed_mean_delta']:+.6f} | "
            f"{'yes' if comparison['fresh_seeds_both_negative'] else 'no'} |"
        )

    gates = results["gates"]
    lines.extend(
        [
            "",
            "## Predeclared decision",
            "",
            f"Rank 32 beats scalar in every seed with mean delta at most -0.003: "
            f"{'yes' if gates['rank32_beats_scalar_in_all_seeds_with_material_mean'] else 'no'}.",
            f"Rank 32 beats the FFN control in every seed with mean delta at most -0.003: "
            f"{'yes' if gates['rank32_beats_ffn_control_in_all_seeds_with_material_mean'] else 'no'}.",
            f"All diagnostics finite: {'yes' if gates['all_diagnostics_finite'] else 'no'}.",
            f"Overall replication gate: {'pass' if gates['replication_passes'] else 'fail'}.",
            "",
            results["inference_note"],
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
