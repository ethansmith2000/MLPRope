#!/usr/bin/env python
"""Analyze the matched Phase-43 low-rank pathway development screen."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "model-output" / "position_bias_phase43_lowrank_qk_pathways"
RESULT_ROOT = ROOT / "results" / "phase43_lowrank_qk_pathways"
STEP = 20_000
ARMS = {
    "lowrank-premap": "phase43-lowrank-premap-r32-seed123-b32-s20000-h768d8",
    "lowrank-qk-replace": (
        "phase43-lowrank-qk-replace-r32-seed123-b32-s20000-h768d8"
    ),
    "lowrank-qk-residual": (
        "phase43-lowrank-qk-residual-r32-seed123-b32-s20000-h768d8"
    ),
    "rope-control": "phase43-rope-control-r32-seed123-b32-s20000-h768d8",
    "qkpre-control": "phase43-qkpre-control-r32-seed123-b32-s20000-h768d8",
}
CONTRASTS = {
    "lowrank-premap_minus_qkpre-control": (
        "lowrank-premap",
        "qkpre-control",
    ),
    "lowrank-qk-replace_minus_rope-control": (
        "lowrank-qk-replace",
        "rope-control",
    ),
    "lowrank-qk-residual_minus_qkpre-control": (
        "lowrank-qk-residual",
        "qkpre-control",
    ),
    "lowrank-qk-residual_minus_lowrank-premap": (
        "lowrank-qk-residual",
        "lowrank-premap",
    ),
}
PRIMARY_CONTRASTS = set(CONTRASTS) - {
    "lowrank-qk-residual_minus_lowrank-premap"
}


def run_dir(arm: str) -> Path:
    return OUTPUT_ROOT / ARMS[arm]


def final_losses(arm: str) -> np.ndarray:
    detail = run_dir(arm) / "evaluation_details" / (
        f"step_{STEP:08d}_context_001024.json"
    )
    payload = json.loads(detail.read_text())
    if payload.get("evaluation_kind") != "final_holdout":
        raise ValueError(f"Unexpected evaluation kind in {detail}")
    if payload.get("evaluation_start_batch") != 2_048:
        raise ValueError(f"Unexpected holdout start in {detail}")
    values = np.asarray(payload["losses"], dtype=np.float64)
    if values.shape != (1_024,):
        raise ValueError(f"Expected 1,024 losses in {detail}, got {values.shape}")
    return values


def bootstrap_ci(delta: np.ndarray, *, seed: int = 43) -> list[float]:
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(10):
        indices = rng.integers(0, delta.size, size=(2_000, delta.size))
        samples.append(delta[indices].mean(axis=1))
    low, high = np.quantile(np.concatenate(samples), (0.025, 0.975))
    return [float(low), float(high)]


def development_curve(arm: str) -> dict[int, float]:
    curve = {}
    with (run_dir(arm) / "metrics.jsonl").open() as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("evaluation_kind") != "development":
                continue
            step = record.get("step")
            loss = record.get("eval_loss/context_1024", record.get("eval_loss"))
            if isinstance(step, int) and isinstance(loss, (int, float)):
                curve[step] = float(loss)
    return curve


def optimization_health(arm: str) -> dict | None:
    path = run_dir(arm) / "intervention_optimization.jsonl"
    if not path.is_file():
        return None
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    prefix = "optimization/pre_qk_sinusoid_adapter"
    numeric = [
        value
        for row in rows
        for key, value in row.items()
        if key not in {"step", "timestamp"} and isinstance(value, (int, float))
    ]
    active = [
        row
        for row in rows
        if row.get(f"{prefix}/carrier_function_step/rms", 0) > 0
    ]
    return {
        "sample_steps": [row["step"] for row in rows],
        "all_numeric_finite": all(math.isfinite(float(value)) for value in numeric),
        "function_active": bool(active),
        "last_active_step": active[-1]["step"] if active else None,
        "last_active_function_step_rms": (
            active[-1][f"{prefix}/carrier_function_step/rms"] if active else None
        ),
        "last_active_gradient_clip_ratio": (
            active[-1].get(f"{prefix}/gradient_clip_ratio") if active else None
        ),
    }


def run_summary(arm: str, losses: np.ndarray) -> dict:
    training = json.loads((run_dir(arm) / "training_summary.json").read_text())
    provenance = json.loads((run_dir(arm) / "run_provenance.json").read_text())
    counts = provenance.get("parameter_counts", {})
    if not counts and provenance.get("launches"):
        counts = provenance["launches"][-1].get("parameter_counts", {})
    return {
        "final_holdout_loss": float(losses.mean()),
        "development_curve": {
            str(step): loss for step, loss in development_curve(arm).items()
        },
        "target_tokens_per_second": training["target_tokens_per_second"],
        "elapsed_seconds": training["elapsed_seconds"],
        "peak_reserved_mib": training["peak_reserved_mib"],
        "parameter_counts": counts,
        "optimization_health": optimization_health(arm),
    }


def analyze() -> dict:
    missing = [
        str(run_dir(arm))
        for arm in ARMS
        if not (run_dir(arm) / "COMPLETED").is_file()
    ]
    if missing:
        raise RuntimeError(f"Phase 43 incomplete: {missing}")
    losses = {arm: final_losses(arm) for arm in ARMS}
    curves = {arm: development_curve(arm) for arm in ARMS}
    runs = {arm: run_summary(arm, losses[arm]) for arm in ARMS}
    contrasts = {}
    for name, (candidate, reference) in CONTRASTS.items():
        delta = losses[candidate] - losses[reference]
        ci = bootstrap_ci(delta)
        common_steps = sorted(set(curves[candidate]) & set(curves[reference]))
        curve = {
            str(step): curves[candidate][step] - curves[reference][step]
            for step in common_steps
        }
        late_change = None
        if 15_000 in common_steps and 20_000 in common_steps:
            late_change = curve["20000"] - curve["15000"]
        health = runs[candidate]["optimization_health"]
        passes = (
            name in PRIMARY_CONTRASTS
            and float(delta.mean()) <= -0.003
            and ci[1] < 0
            and late_change is not None
            and late_change <= 0.002
            and health is not None
            and health["all_numeric_finite"]
            and health["function_active"]
        )
        contrasts[name] = {
            "delta": float(delta.mean()),
            "paired_example_bootstrap_ci95": ci,
            "development_delta_curve": curve,
            "late_delta_change_15k_to_20k": late_change,
            "passes_predeclared_screen": passes,
        }
    return {
        "scope": "phase43_lowrank_qk_pathways",
        "context": 1_024,
        "sequence_batch": 32,
        "training_steps": STEP,
        "nominal_training_tokens": STEP * 32 * 1_024,
        "rank": 32,
        "seed": 123,
        "final_holdout_start_batch": 2_048,
        "final_holdout_examples": 1_024,
        "screen_gate": {
            "delta_at_most": -0.003,
            "paired_interval_upper_below": 0.0,
            "late_delta_change_at_most": 0.002,
            "requires_finite_active_adapter": True,
        },
        "runs": runs,
        "contrasts": contrasts,
        "caveat": (
            "Paired-example intervals measure holdout precision within one "
            "training seed, not training-seed uncertainty."
        ),
    }


def render(results: dict) -> str:
    lines = [
        "# Phase 43: low-rank Q/K positional pathways",
        "",
        "All runs use h768/d8, context 1024, sequence batch 32, seed 123, "
        "20k updates, rank 32 where applicable, and the same disjoint "
        "1,024-example holdout.",
        "",
        "| Arm | Final NLL | tokens/s | Position params |",
        "|---|---:|---:|---:|",
    ]
    for arm in ARMS:
        run = results["runs"][arm]
        lines.append(
            f"| {arm} | {run['final_holdout_loss']:.6f} | "
            f"{run['target_tokens_per_second']:.0f} | "
            f"{run['parameter_counts'].get('position_params', 0)} |"
        )
    lines.extend(
        [
            "",
            "## Paired contrasts",
            "",
            "Negative deltas favor the first named arm.",
            "",
            "| Contrast | Delta | Paired 95% interval | Pass |",
            "|---|---:|---:|:---:|",
        ]
    )
    for name, contrast in results["contrasts"].items():
        low, high = contrast["paired_example_bootstrap_ci95"]
        passed = "yes" if contrast["passes_predeclared_screen"] else "no"
        lines.append(
            f"| {name} | {contrast['delta']:+.6f} | "
            f"[{low:+.6f}, {high:+.6f}] | {passed} |"
        )
    lines.extend(["", results["caveat"], ""])
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
