#!/usr/bin/env python
"""Analyze the one-shot empirical rank-32 function-step calibration."""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PHASE48_ROOT = ROOT / "model-output" / "position_bias_phase48_empirical_rank32"
PHASE47_ROOT = ROOT / "model-output" / "position_bias_phase47_rank_calibration"
PHASE45_ROOT = ROOT / "model-output" / "position_bias_phase45_novel_static_maps"
PHASE43_ROOT = ROOT / "model-output" / "position_bias_phase43_lowrank_qk_pathways"
RESULT_ROOT = ROOT / "results" / "phase48_empirical_rank32"
STEP = 20_000
RUNS = {
    "empirical-r32": (
        PHASE48_ROOT,
        "phase48-empirical-r32-seed123-b32-s20000-h768d8",
    ),
    "calibrated-r128": (
        PHASE47_ROOT,
        "phase47-calibrated-r128-seed123-b32-s20000-h768d8",
    ),
    "theoretical-r32": (
        PHASE47_ROOT,
        "phase47-calibrated-r32-seed123-b32-s20000-h768d8",
    ),
    "scalar-parent": (
        PHASE43_ROOT,
        "phase43-qkpre-control-r32-seed123-b32-s20000-h768d8",
    ),
    "dense-separate": (
        PHASE45_ROOT,
        "phase45-qk-dense-separate-seed123-b32-s20000-h768d8",
    ),
}
CONTRASTS = {
    "empirical-r32_minus_calibrated-r128": (
        "empirical-r32",
        "calibrated-r128",
    ),
    "empirical-r32_minus_theoretical-r32": (
        "empirical-r32",
        "theoretical-r32",
    ),
    "empirical-r32_minus_scalar-parent": ("empirical-r32", "scalar-parent"),
    "empirical-r32_minus_dense-separate": ("empirical-r32", "dense-separate"),
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
    rng = np.random.default_rng(48)
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


def _primary_summary(losses: np.ndarray) -> dict:
    run_dir = _run_dir("empirical-r32")
    training = json.loads((run_dir / "training_summary.json").read_text())
    provenance = json.loads((run_dir / "run_provenance.json").read_text())
    config = json.loads((run_dir / "training_config.json").read_text())
    counts = provenance.get("parameter_counts", {})
    if not counts and provenance.get("launches"):
        counts = provenance["launches"][-1].get("parameter_counts", {})
    clips = [
        float(row[CLIP_KEY])
        for row in _optimization_rows("empirical-r32")
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
    if not (_run_dir("empirical-r32") / "COMPLETED").is_file():
        raise RuntimeError("Phase 48 empirical rank-32 run is incomplete")
    losses = {arm: _losses(arm) for arm in RUNS}
    comparisons = {}
    for name, (candidate, reference) in CONTRASTS.items():
        delta = losses[candidate] - losses[reference]
        comparisons[name] = {
            "delta": float(delta.mean()),
            "paired_example_bootstrap_ci95": _ci(delta),
        }

    r32_steps = _function_steps("empirical-r32")
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
    rank_interval = comparisons[
        "empirical-r32_minus_calibrated-r128"
    ]["paired_example_bootstrap_ci95"]
    original_rank_gap = 0.003910406143404543
    matched_rank_gap = comparisons[
        "empirical-r32_minus_calibrated-r128"
    ]["delta"]
    gap_fraction_closed = 1.0 - matched_rank_gap / original_rank_gap
    if not passes_match:
        decision = (
            "The one-shot empirical calibration still failed the predeclared "
            "function-step gate. Close the rank-capacity claim without further "
            "local LR tuning."
        )
    elif rank_interval[0] > 0:
        decision = (
            "The function-step gate passed. Calibrated rank 128 retained a small "
            "paired endpoint advantage, compatible with a modest capacity effect, "
            "but it is below the project's 0.003 scout materiality margin and "
            "does not establish a robust rank claim from one training seed."
        )
    elif rank_interval[1] < 0:
        decision = (
            "The function-step gate passed but empirical rank 32 beat "
            "rank 128; the earlier rank advantage was optimization-mediated."
        )
    else:
        decision = (
            "The function-step gate passed and the rank endpoints were "
            "statistically tied; the earlier rank advantage was not robust to "
            "function-step matching."
        )
    return {
        "scope": "phase48_empirical_rank32",
        "context": 1_024,
        "sequence_batch": 32,
        "training_steps": STEP,
        "seed": 123,
        "calibration_source": {
            "phase47_theoretical_r32_multiplier": 4.898979485566356,
            "phase47_postwarmup_function_ratio": 1.2997578757740733,
            "empirical_r32_multiplier": 6.367487169620489,
            "selected_without_validation_loss": True,
        },
        "run": _primary_summary(losses["empirical-r32"]),
        "reference_losses": {
            arm: float(losses[arm].mean()) for arm in RUNS if arm != "empirical-r32"
        },
        "comparisons": comparisons,
        "function_step_scale_r128_over_r32": {
            "early_through_step_64_median": statistics.median(early),
            "postwarmup_step_1000_to_19000_median": postwarmup_median,
            "postwarmup_min": min(postwarmup),
            "postwarmup_max": max(postwarmup),
            "passes_predeclared_match": passes_match,
        },
        "rank_gap_decomposition": {
            "original_equal_lr_rank128_advantage": original_rank_gap,
            "matched_function_step_rank128_advantage": matched_rank_gap,
            "fraction_of_original_gap_closed_by_calibration": gap_fraction_closed,
        },
        "scout_complete": True,
        "decision": decision,
        "caveat": (
            "The rank-128 comparator was frozen before selecting the empirical "
            "rank-32 multiplier. Runs share the schedule, initialization seed, "
            "data order, and holdout; paired-example intervals do not estimate "
            "training-seed uncertainty."
        ),
    }


def render(results: dict) -> str:
    run = results["run"]
    lines = [
        "# Phase 48: one-shot empirical rank-32 calibration",
        "",
        f"Empirical rank 32 used readout LR multiplier "
        f"`{run['readout_lr_multiplier']:.6f}` and reached final holdout NLL "
        f"`{run['final_holdout_loss']:.6f}`.",
        "",
        "## Paired contrasts",
        "",
        "Negative deltas favor empirical rank 32.",
        "",
        "| Contrast | Delta | Paired 95% interval |",
        "|---|---:|---:|",
    ]
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
            "The matched rank-32 arm closed "
            f"{100 * results['rank_gap_decomposition']['fraction_of_original_gap_closed_by_calibration']:.1f}% "
            "of the original equal-LR rank gap.",
            "",
            "## Decision",
            "",
            results["decision"],
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
