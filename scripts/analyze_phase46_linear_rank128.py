#!/usr/bin/env python
"""Analyze linear rank-128 projected-space maps against prior matched runs."""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "model-output" / "position_bias_phase46_linear_rank128"
PHASE43_ROOT = ROOT / "model-output" / "position_bias_phase43_lowrank_qk_pathways"
PHASE45_ROOT = ROOT / "model-output" / "position_bias_phase45_novel_static_maps"
RESULT_ROOT = ROOT / "results" / "phase46_linear_rank128"
STEP = 20_000
RUNS = {
    "linear-r128-shared": (
        OUTPUT_ROOT,
        "phase46-linear-r128-shared-seed123-b32-s20000-h768d8",
    ),
    "linear-r128-separate": (
        OUTPUT_ROOT,
        "phase46-linear-r128-separate-seed123-b32-s20000-h768d8",
    ),
    "scalar-parent": (
        PHASE43_ROOT,
        "phase43-qkpre-control-r32-seed123-b32-s20000-h768d8",
    ),
    "linear-r32-separate": (
        PHASE43_ROOT,
        "phase43-lowrank-qk-residual-r32-seed123-b32-s20000-h768d8",
    ),
    "dense-shared": (
        PHASE45_ROOT,
        "phase45-qk-dense-shared-seed123-b32-s20000-h768d8",
    ),
    "dense-separate": (
        PHASE45_ROOT,
        "phase45-qk-dense-separate-seed123-b32-s20000-h768d8",
    ),
    "nonlinear-r128-separate": (
        PHASE45_ROOT,
        "phase45-qk-mlp-r128-seed123-b32-s20000-h768d8",
    ),
}
PRIMARY_ARMS = ("linear-r128-shared", "linear-r128-separate")
CONTRASTS = {
    "linear-r128-shared_minus_scalar-parent": (
        "linear-r128-shared",
        "scalar-parent",
    ),
    "linear-r128-separate_minus_scalar-parent": (
        "linear-r128-separate",
        "scalar-parent",
    ),
    "linear-r128-shared_minus_linear-r128-separate": (
        "linear-r128-shared",
        "linear-r128-separate",
    ),
    "linear-r128-shared_minus_dense-shared": (
        "linear-r128-shared",
        "dense-shared",
    ),
    "linear-r128-separate_minus_dense-separate": (
        "linear-r128-separate",
        "dense-separate",
    ),
    "linear-r128-separate_minus_nonlinear-r128-separate": (
        "linear-r128-separate",
        "nonlinear-r128-separate",
    ),
    "linear-r128-separate_minus_linear-r32-separate": (
        "linear-r128-separate",
        "linear-r32-separate",
    ),
}


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
    rng = np.random.default_rng(46)
    means = []
    for _ in range(10):
        indices = rng.integers(0, delta.size, size=(2_000, delta.size))
        means.append(delta[indices].mean(axis=1))
    return [
        float(value)
        for value in np.quantile(np.concatenate(means), (0.025, 0.975))
    ]


def _summary(arm: str, losses: np.ndarray) -> dict:
    run_dir = _run_dir(arm)
    training = json.loads((run_dir / "training_summary.json").read_text())
    provenance = json.loads((run_dir / "run_provenance.json").read_text())
    counts = provenance.get("parameter_counts", {})
    if not counts and provenance.get("launches"):
        counts = provenance["launches"][-1].get("parameter_counts", {})
    return {
        "final_holdout_loss": float(losses.mean()),
        "position_params": int(counts.get("position_params", 0)),
        "target_tokens_per_second": training["target_tokens_per_second"],
        "peak_reserved_mib": training["peak_reserved_mib"],
    }


def _function_steps(arm: str) -> dict[int, float]:
    key = "optimization/pre_qk_sinusoid_adapter/carrier_function_step/rms"
    path = _run_dir(arm) / "intervention_optimization.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    return {
        int(row["step"]): float(row[key])
        for row in rows
        if isinstance(row.get(key), (int, float)) and float(row[key]) > 0
    }


def analyze() -> dict:
    missing = [arm for arm in PRIMARY_ARMS if not (_run_dir(arm) / "COMPLETED").is_file()]
    if missing:
        raise RuntimeError(f"Phase 46 incomplete: {missing}")
    losses = {arm: _losses(arm) for arm in RUNS}
    comparisons = {}
    for name, (candidate, reference) in CONTRASTS.items():
        delta = losses[candidate] - losses[reference]
        comparisons[name] = {
            "delta": float(delta.mean()),
            "paired_example_bootstrap_ci95": _ci(delta),
        }
    r32_steps = _function_steps("linear-r32-separate")
    r128_steps = _function_steps("linear-r128-separate")
    ratios = {
        step: r128_steps[step] / r32_steps[step]
        for step in r32_steps.keys() & r128_steps.keys()
    }
    early = [ratio for step, ratio in ratios.items() if step <= 64]
    postwarmup = [
        ratio for step, ratio in ratios.items() if 1_000 <= step <= 19_000
    ]
    return {
        "scope": "phase46_linear_rank128",
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
            "postwarmup_step_1000_to_19000_median": statistics.median(postwarmup),
            "postwarmup_min": min(postwarmup),
            "postwarmup_max": max(postwarmup),
            "interpretation": (
                "Rank 128 took materially larger carrier-function steps under "
                "the same Adam LR, so its loss gain does not isolate capacity."
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
        "# Phase 46: linear rank-128 projected-space maps",
        "",
        "| Arm | Final NLL | Position params | tokens/s |",
        "|---|---:|---:|---:|",
    ]
    for arm, run in results["runs"].items():
        lines.append(
            f"| {arm} | {run['final_holdout_loss']:.6f} | "
            f"{run['position_params']} | {run['target_tokens_per_second']:.0f} |"
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
    lines.extend(
        [
            "",
            "## Optimizer-scale diagnostic",
            "",
            "With the same Adam LR, rank 128's carrier-function step was "
            f"{scale['early_through_step_64_median']:.2f}x rank 32 through step 64 "
            f"and {scale['postwarmup_step_1000_to_19000_median']:.2f}x at the "
            "median sampled post-warmup step. The rank result therefore mixes "
            "capacity with function-space update scale.",
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
