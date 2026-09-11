#!/usr/bin/env python
"""Analyze dimension-calibrated rank-32/rank-128 Q/K readouts."""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "model-output" / "position_bias_phase47_rank_calibration"
PHASE43_ROOT = ROOT / "model-output" / "position_bias_phase43_lowrank_qk_pathways"
PHASE45_ROOT = ROOT / "model-output" / "position_bias_phase45_novel_static_maps"
PHASE46_ROOT = ROOT / "model-output" / "position_bias_phase46_linear_rank128"
RESULT_ROOT = ROOT / "results" / "phase47_rank_calibration"
STEP = 20_000
RUNS = {
    "calibrated-r32": (
        OUTPUT_ROOT,
        "phase47-calibrated-r32-seed123-b32-s20000-h768d8",
    ),
    "calibrated-r128": (
        OUTPUT_ROOT,
        "phase47-calibrated-r128-seed123-b32-s20000-h768d8",
    ),
    "scalar-parent": (
        PHASE43_ROOT,
        "phase43-qkpre-control-r32-seed123-b32-s20000-h768d8",
    ),
    "uncalibrated-r32": (
        PHASE43_ROOT,
        "phase43-lowrank-qk-residual-r32-seed123-b32-s20000-h768d8",
    ),
    "uncalibrated-r128": (
        PHASE46_ROOT,
        "phase46-linear-r128-separate-seed123-b32-s20000-h768d8",
    ),
    "dense-separate": (
        PHASE45_ROOT,
        "phase45-qk-dense-separate-seed123-b32-s20000-h768d8",
    ),
}
PRIMARY_ARMS = ("calibrated-r32", "calibrated-r128")
CONTRASTS = {
    "calibrated-r32_minus_scalar-parent": ("calibrated-r32", "scalar-parent"),
    "calibrated-r128_minus_scalar-parent": ("calibrated-r128", "scalar-parent"),
    "calibrated-r32_minus_calibrated-r128": (
        "calibrated-r32",
        "calibrated-r128",
    ),
    "calibrated-r32_minus_uncalibrated-r32": (
        "calibrated-r32",
        "uncalibrated-r32",
    ),
    "calibrated-r128_minus_uncalibrated-r128": (
        "calibrated-r128",
        "uncalibrated-r128",
    ),
    "calibrated-r32_minus_dense-separate": (
        "calibrated-r32",
        "dense-separate",
    ),
    "calibrated-r128_minus_dense-separate": (
        "calibrated-r128",
        "dense-separate",
    ),
}
FUNCTION_KEY = "optimization/pre_qk_sinusoid_adapter/carrier_function_step/rms"
CLIP_KEY = "optimization/pre_qk_sinusoid_adapter/gradient_clip_ratio"


def _run_dir(arm: str) -> Path:
    root, name = RUNS[arm]
    return root / name


def _losses(arm: str) -> np.ndarray:
    path = _run_dir(arm) / "evaluation_details" / (
        f"step_{STEP:08d}_context_001024.json"
    )
    payload = json.loads(path.read_text())
    values = np.asarray(payload["losses"], dtype=np.float64)
    if payload.get("evaluation_kind") != "final_holdout" or values.shape != (1_024,):
        raise ValueError(f"Unexpected final evaluation in {path}")
    return values


def _ci(delta: np.ndarray) -> list[float]:
    rng = np.random.default_rng(47)
    means = []
    for _ in range(10):
        indices = rng.integers(0, delta.size, size=(2_000, delta.size))
        means.append(delta[indices].mean(axis=1))
    return [
        float(value)
        for value in np.quantile(np.concatenate(means), (0.025, 0.975))
    ]


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


def _summary(arm: str, losses: np.ndarray) -> dict:
    run_dir = _run_dir(arm)
    training = json.loads((run_dir / "training_summary.json").read_text())
    provenance = json.loads((run_dir / "run_provenance.json").read_text())
    config = json.loads((run_dir / "training_config.json").read_text())
    counts = provenance.get("parameter_counts", {})
    if not counts and provenance.get("launches"):
        counts = provenance["launches"][-1].get("parameter_counts", {})
    clips = [
        float(row[CLIP_KEY])
        for row in _optimization_rows(arm)
        if isinstance(row.get(CLIP_KEY), (int, float))
    ]
    return {
        "rank": int(config["qk_preprojection"]["rank"]),
        "readout_lr_multiplier": float(
            config["qk_preprojection"]["readout_lr_multiplier"]
        ),
        "final_holdout_loss": float(losses.mean()),
        "position_params": int(counts.get("position_params", 0)),
        "target_tokens_per_second": training["target_tokens_per_second"],
        "peak_reserved_mib": training["peak_reserved_mib"],
        "minimum_gradient_clip_ratio": min(clips),
    }


def analyze() -> dict:
    missing = [arm for arm in PRIMARY_ARMS if not (_run_dir(arm) / "COMPLETED").is_file()]
    if missing:
        raise RuntimeError(f"Phase 47 incomplete: {missing}")
    losses = {arm: _losses(arm) for arm in RUNS}
    comparisons = {}
    for name, (candidate, reference) in CONTRASTS.items():
        delta = losses[candidate] - losses[reference]
        comparisons[name] = {
            "delta": float(delta.mean()),
            "paired_example_bootstrap_ci95": _ci(delta),
        }
    r32_steps = _function_steps("calibrated-r32")
    r128_steps = _function_steps("calibrated-r128")
    ratios = {
        step: r128_steps[step] / r32_steps[step]
        for step in r32_steps.keys() & r128_steps.keys()
    }
    early = [ratio for step, ratio in ratios.items() if step <= 64]
    postwarmup = [
        ratio for step, ratio in ratios.items() if 1_000 <= step <= 19_000
    ]
    postwarmup_median = statistics.median(postwarmup)
    passes_match = 0.8 <= postwarmup_median <= 1.25
    return {
        "scope": "phase47_rank_calibration",
        "context": 1_024,
        "sequence_batch": 32,
        "training_steps": STEP,
        "seed": 123,
        "runs": {arm: _summary(arm, losses[arm]) for arm in PRIMARY_ARMS},
        "reference_losses": {
            arm: float(losses[arm].mean()) for arm in RUNS if arm not in PRIMARY_ARMS
        },
        "comparisons": comparisons,
        "function_step_scale_r128_over_r32": {
            "early_through_step_64_median": statistics.median(early),
            "postwarmup_step_1000_to_19000_median": postwarmup_median,
            "postwarmup_min": min(postwarmup),
            "postwarmup_max": max(postwarmup),
            "passes_predeclared_match": passes_match,
        },
        "decision": {
            "observed_best_endpoint": "calibrated-r128",
            "rank_comparison_interpretable_as_capacity": passes_match,
            "reason": (
                "The predeclared carrier-function-step match passed."
                if passes_match
                else (
                    "The predeclared carrier-function-step match failed; the "
                    "rank-128 endpoint advantage cannot be attributed cleanly "
                    "to representational rank."
                )
            ),
        },
        "caveat": (
            "All references use the identical schedule, paired initialization "
            "seed, data order, and holdout. Paired-example intervals do not "
            "estimate training-seed uncertainty."
        ),
    }


def render(results: dict) -> str:
    lines = [
        "# Phase 47: dimension-calibrated low-rank Q/K readouts",
        "",
        "| Arm | Rank | Readout LR multiplier | Final NLL | Position params | tokens/s | Min clip ratio |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for arm, run in results["runs"].items():
        lines.append(
            f"| {arm} | {run['rank']} | {run['readout_lr_multiplier']:.6f} | "
            f"{run['final_holdout_loss']:.6f} | {run['position_params']} | "
            f"{run['target_tokens_per_second']:.0f} | "
            f"{run['minimum_gradient_clip_ratio']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Paired contrasts",
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
    scale = results["function_step_scale_r128_over_r32"]
    match = "yes" if scale["passes_predeclared_match"] else "no"
    lines.extend(
        [
            "",
            "## Function-step calibration",
            "",
            f"Median rank-128/rank-32 carrier-step ratio through step 64: "
            f"{scale['early_through_step_64_median']:.3f}.",
            f"Median ratio from step 1k through 19k: "
            f"{scale['postwarmup_step_1000_to_19000_median']:.3f} "
            f"(predeclared 0.8--1.25 match: {match}).",
            "",
            "## Decision",
            "",
            results["decision"]["reason"],
            "Both calibrated arms remain valid optimization results, but the "
            "rank contrast is not a clean capacity ablation when the match fails.",
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
