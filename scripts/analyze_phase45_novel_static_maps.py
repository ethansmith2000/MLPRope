#!/usr/bin/env python
"""Analyze Phase 45 against the existing matched 20k parent runs."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "model-output" / "position_bias_phase45_novel_static_maps"
INPUT_CONTROL_ROOT = ROOT / "model-output" / "position_bias_phase44_input_adapters"
QK_CONTROL_ROOT = ROOT / "model-output" / "position_bias_phase43_lowrank_qk_pathways"
RESULT_ROOT = ROOT / "results" / "phase45_novel_static_maps"
STEP = 20_000
RUNS = {
    "input-dense-linear": "phase45-input-dense-linear-seed123-b32-s20000-h768d8",
    "input-mlp-h768": "phase45-input-mlp-h768-seed123-b32-s20000-h768d8",
    "qk-dense-premap": "phase45-qk-dense-premap-seed123-b32-s20000-h768d8",
    "qk-dense-shared": "phase45-qk-dense-shared-seed123-b32-s20000-h768d8",
    "qk-dense-separate": "phase45-qk-dense-separate-seed123-b32-s20000-h768d8",
    "qk-mlp-r128": "phase45-qk-mlp-r128-seed123-b32-s20000-h768d8",
}
CONTROLS = {
    "input-scalar-control": (
        INPUT_CONTROL_ROOT,
        "phase44-input-scalar-control-seed123-b32-s20000-h768d8",
    ),
    "qkpre-control": (
        QK_CONTROL_ROOT,
        "phase43-qkpre-control-r32-seed123-b32-s20000-h768d8",
    ),
    "input-pair-amplitude": (
        INPUT_CONTROL_ROOT,
        "phase44-per-pair-amplitude-seed123-b32-s20000-h768d8",
    ),
    "qk-r32-premap": (
        QK_CONTROL_ROOT,
        "phase43-lowrank-premap-r32-seed123-b32-s20000-h768d8",
    ),
    "qk-r32-residual": (
        QK_CONTROL_ROOT,
        "phase43-lowrank-qk-residual-r32-seed123-b32-s20000-h768d8",
    ),
}
SECONDARY_CONTRASTS = {
    "qk-dense-shared_minus_qk-dense-separate": (
        "qk-dense-shared",
        "qk-dense-separate",
    ),
    "qk-dense-shared_minus_qk-mlp-r128": (
        "qk-dense-shared",
        "qk-mlp-r128",
    ),
    "qk-dense-shared_minus_qk-r32-residual": (
        "qk-dense-shared",
        "qk-r32-residual",
    ),
    "qk-dense-separate_minus_qk-r32-residual": (
        "qk-dense-separate",
        "qk-r32-residual",
    ),
    "qk-mlp-r128_minus_qk-r32-residual": (
        "qk-mlp-r128",
        "qk-r32-residual",
    ),
    "qk-dense-premap_minus_qk-r32-premap": (
        "qk-dense-premap",
        "qk-r32-premap",
    ),
    "input-mlp-h768_minus_input-pair-amplitude": (
        "input-mlp-h768",
        "input-pair-amplitude",
    ),
}


def _run_dir(arm: str) -> Path:
    if arm in RUNS:
        return OUTPUT_ROOT / RUNS[arm]
    root, name = CONTROLS[arm]
    return root / name


def _losses(arm: str) -> np.ndarray:
    detail = _run_dir(arm) / "evaluation_details" / (
        f"step_{STEP:08d}_context_001024.json"
    )
    payload = json.loads(detail.read_text())
    values = np.asarray(payload["losses"], dtype=np.float64)
    if payload.get("evaluation_kind") != "final_holdout" or values.shape != (1_024,):
        raise ValueError(f"Unexpected final evaluation in {detail}")
    return values


def _ci(delta: np.ndarray) -> list[float]:
    rng = np.random.default_rng(45)
    samples = []
    for _ in range(10):
        indices = rng.integers(0, delta.size, size=(2_000, delta.size))
        samples.append(delta[indices].mean(axis=1))
    return [
        float(value)
        for value in np.quantile(np.concatenate(samples), (0.025, 0.975))
    ]


def _final_record(arm: str) -> dict:
    final = None
    with (_run_dir(arm) / "metrics.jsonl").open() as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("evaluation_kind") == "final_holdout":
                final = record
    if final is None:
        raise RuntimeError(f"Missing final holdout for {arm}")
    return final


def _run_summary(arm: str, values: np.ndarray) -> dict:
    training = json.loads((_run_dir(arm) / "training_summary.json").read_text())
    provenance = json.loads((_run_dir(arm) / "run_provenance.json").read_text())
    counts = provenance.get("parameter_counts", {})
    if not counts and provenance.get("launches"):
        counts = provenance["launches"][-1].get("parameter_counts", {})
    final = _final_record(arm)
    diagnostics = {
        key: value
        for key, value in final.items()
        if key.startswith("position/")
    }
    numeric = [value for value in diagnostics.values() if isinstance(value, (int, float))]
    return {
        "final_holdout_loss": float(values.mean()),
        "target_tokens_per_second": training["target_tokens_per_second"],
        "elapsed_seconds": training["elapsed_seconds"],
        "peak_reserved_mib": training["peak_reserved_mib"],
        "parameter_counts": counts,
        "position_diagnostics": diagnostics,
        "diagnostics_finite": all(math.isfinite(float(value)) for value in numeric),
    }


def analyze() -> dict:
    missing = [arm for arm in RUNS if not (_run_dir(arm) / "COMPLETED").is_file()]
    if missing:
        raise RuntimeError(f"Phase 45 incomplete: {missing}")
    all_arms = (*RUNS, *CONTROLS)
    losses = {arm: _losses(arm) for arm in all_arms}
    runs = {arm: _run_summary(arm, losses[arm]) for arm in RUNS}
    comparisons = {}
    for arm in RUNS:
        parent = "input-scalar-control" if arm.startswith("input-") else "qkpre-control"
        delta = losses[arm] - losses[parent]
        ci = _ci(delta)
        comparisons[f"{arm}_minus_{parent}"] = {
            "candidate": arm,
            "parent": parent,
            "delta": float(delta.mean()),
            "paired_example_bootstrap_ci95": ci,
            "passes_breadth_screen": bool(float(delta.mean()) <= -0.003 and ci[1] < 0),
        }
    secondary_comparisons = {}
    for name, (candidate, reference) in SECONDARY_CONTRASTS.items():
        delta = losses[candidate] - losses[reference]
        secondary_comparisons[name] = {
            "candidate": candidate,
            "reference": reference,
            "delta": float(delta.mean()),
            "paired_example_bootstrap_ci95": _ci(delta),
        }
    return {
        "scope": "phase45_novel_static_maps",
        "context": 1_024,
        "sequence_batch": 32,
        "training_steps": STEP,
        "seed": 123,
        "runs": runs,
        "control_losses": {
            arm: float(losses[arm].mean()) for arm in CONTROLS
        },
        "comparisons": comparisons,
        "secondary_comparisons": secondary_comparisons,
        "screen_gate": {
            "delta_at_most": -0.003,
            "paired_interval_upper_below": 0.0,
        },
        "caveat": (
            "Existing Phase-43/44 controls use the identical 20k paper-base "
            "schedule, paired initialization seed, data order, and final holdout. "
            "Paired-example intervals do not estimate training-seed uncertainty."
        ),
    }


def render(results: dict) -> str:
    lines = [
        "# Phase 45: novel static sinusoid maps",
        "",
        "| Arm | Final NLL | Parent delta | Paired 95% interval | Position params | tokens/s | Pass |",
        "|---|---:|---:|---:|---:|---:|:---:|",
    ]
    comparisons = results["comparisons"]
    for arm, run in results["runs"].items():
        comparison = comparisons[
            next(name for name, value in comparisons.items() if value["candidate"] == arm)
        ]
        low, high = comparison["paired_example_bootstrap_ci95"]
        passed = "yes" if comparison["passes_breadth_screen"] else "no"
        lines.append(
            f"| {arm} | {run['final_holdout_loss']:.6f} | "
            f"{comparison['delta']:+.6f} | [{low:+.6f}, {high:+.6f}] | "
            f"{run['parameter_counts'].get('position_params', 0)} | "
            f"{run['target_tokens_per_second']:.0f} | {passed} |"
        )
    lines.extend(
        [
            "",
            f"Input scalar parent: {results['control_losses']['input-scalar-control']:.6f}.",
            f"Pre-Q/K scalar parent: {results['control_losses']['qkpre-control']:.6f}.",
            "",
            "## Secondary paired contrasts",
            "",
            "Negative deltas favor the first named arm.",
            "",
            "| Contrast | Delta | Paired 95% interval |",
            "|---|---:|---:|",
        ]
    )
    for name, comparison in results["secondary_comparisons"].items():
        low, high = comparison["paired_example_bootstrap_ci95"]
        lines.append(
            f"| {name} | {comparison['delta']:+.6f} | "
            f"[{low:+.6f}, {high:+.6f}] |"
        )
    lines.extend(
        [
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
