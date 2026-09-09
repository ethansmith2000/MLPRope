#!/usr/bin/env python
"""Analyze the matched Phase-44 input-sinusoid adapter screen."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "model-output" / "position_bias_phase44_input_adapters"
RESULT_ROOT = ROOT / "results" / "phase44_input_adapters"
RUNS = {
    "lowrank-linear-r32": (
        "phase44-lowrank-linear-r32-seed123-b32-s20000-h768d8"
    ),
    "per-pair-amplitude": (
        "phase44-per-pair-amplitude-seed123-b32-s20000-h768d8"
    ),
    "input-scalar-control": (
        "phase44-input-scalar-control-seed123-b32-s20000-h768d8"
    ),
}


def _run_dir(arm: str) -> Path:
    return OUTPUT_ROOT / RUNS[arm]


def _losses(arm: str) -> np.ndarray:
    detail = _run_dir(arm) / "evaluation_details/step_00020000_context_001024.json"
    payload = json.loads(detail.read_text())
    values = np.asarray(payload["losses"], dtype=np.float64)
    if payload.get("evaluation_kind") != "final_holdout" or values.shape != (1_024,):
        raise ValueError(f"Unexpected final evaluation in {detail}")
    return values


def _ci(delta: np.ndarray) -> list[float]:
    rng = np.random.default_rng(44)
    samples = []
    for _ in range(10):
        indices = rng.integers(0, delta.size, size=(2_000, delta.size))
        samples.append(delta[indices].mean(axis=1))
    return [float(value) for value in np.quantile(np.concatenate(samples), (0.025, 0.975))]


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


def analyze() -> dict:
    missing = [arm for arm in RUNS if not (_run_dir(arm) / "COMPLETED").is_file()]
    if missing:
        raise RuntimeError(f"Phase 44 incomplete: {missing}")
    losses = {arm: _losses(arm) for arm in RUNS}
    reference = losses["input-scalar-control"]
    comparisons = {}
    for arm in ("lowrank-linear-r32", "per-pair-amplitude"):
        delta = losses[arm] - reference
        comparisons[f"{arm}_minus_input-scalar-control"] = {
            "delta": float(delta.mean()),
            "paired_example_bootstrap_ci95": _ci(delta),
        }
    return {
        "scope": "phase44_input_adapters",
        "context": 1_024,
        "sequence_batch": 32,
        "training_steps": 20_000,
        "seed": 123,
        "runs": {
            arm: {
                "final_holdout_loss": float(values.mean()),
                "input_diagnostics": {
                    key: value
                    for key, value in _final_record(arm).items()
                    if key.startswith("position/input_sinusoid/")
                },
            }
            for arm, values in losses.items()
        },
        "comparisons": comparisons,
        "caveat": (
            "Paired-example intervals measure holdout precision within one seed, "
            "not training-seed uncertainty."
        ),
    }


def render(results: dict) -> str:
    lines = [
        "# Phase 44: input-sinusoid adapters",
        "",
        "| Arm | Final holdout NLL |",
        "|---|---:|",
    ]
    for arm, run in results["runs"].items():
        lines.append(f"| {arm} | {run['final_holdout_loss']:.6f} |")
    lines.extend(["", "| Contrast | Delta | Paired 95% interval |", "|---|---:|---:|"])
    for name, comparison in results["comparisons"].items():
        low, high = comparison["paired_example_bootstrap_ci95"]
        lines.append(
            f"| {name} | {comparison['delta']:+.6f} | [{low:+.6f}, {high:+.6f}] |"
        )
    lines.extend(["", results["caveat"], ""])
    return "\n".join(lines)


if __name__ == "__main__":
    results = analyze()
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    (RESULT_ROOT / "phase44_analysis.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n"
    )
    report = render(results)
    (RESULT_ROOT / "PHASE44_RESULTS.md").write_text(report)
    print(report)
