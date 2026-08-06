#!/usr/bin/env python
"""Analyze paired phase-23 dynamic-frequency results."""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PHASE20_ROOT = ROOT / "model-output" / "position_bias_phase20_rope_frequency"
OUTPUT_ROOT = ROOT / "model-output" / "position_bias_phase23_dynamic_frequency"
SEEDS = (123, 456, 789)
ARMS = (
    "linear-horizon",
    "lowrank-linear-horizon",
    "lowrank-silu-horizon",
)


def fixed_dir(seed: int) -> Path:
    return PHASE20_ROOT / f"phase20-fixed-seed{seed}-s5000-h768d8"


def run_dir(arm: str, seed: int) -> Path:
    return OUTPUT_ROOT / f"phase23-{arm}-seed{seed}-s5000-h768d8"


def load_losses(path: Path) -> list[float]:
    detail = path / "evaluation_details" / "step_00005000_context_001024.json"
    payload = json.loads(detail.read_text())
    if payload["evaluation_kind"] != "final_holdout":
        raise ValueError(f"Unexpected evaluation kind in {detail}")
    if payload["evaluation_start_batch"] != 2_048:
        raise ValueError(f"Unexpected holdout offset in {detail}")
    return [float(value) for value in payload["losses"]]


def paired_summary(candidate: list[float], reference: list[float]) -> dict:
    differences = [
        candidate_loss - reference_loss
        for candidate_loss, reference_loss in zip(
            candidate,
            reference,
            strict=True,
        )
    ]
    mean = statistics.fmean(differences)
    std = statistics.stdev(differences)
    half_width = 1.96 * std / math.sqrt(len(differences))
    return {
        "candidate_loss": statistics.fmean(candidate),
        "reference_loss": statistics.fmean(reference),
        "delta_candidate_minus_reference": mean,
        "paired_example_ci95": [mean - half_width, mean + half_width],
        "num_examples": len(differences),
    }


def analyze() -> dict:
    missing = [
        str(run_dir(arm, seed))
        for arm in ARMS
        for seed in SEEDS
        if not (run_dir(arm, seed) / "COMPLETED").is_file()
    ]
    if missing:
        raise RuntimeError(
            "Phase-23 screen is incomplete; missing markers:\n"
            + "\n".join(missing)
        )
    fixed = {seed: load_losses(fixed_dir(seed)) for seed in SEEDS}
    results: dict = {"primary_context": 1024, "arms": {}, "throughput": {}}
    for arm in ARMS:
        rows = []
        for seed in SEEDS:
            row = paired_summary(load_losses(run_dir(arm, seed)), fixed[seed])
            row["seed"] = seed
            rows.append(row)
        deltas = [row["delta_candidate_minus_reference"] for row in rows]
        results["arms"][arm] = {
            "seed_results": rows,
            "mean_delta": statistics.fmean(deltas),
            "seed_delta_std": statistics.stdev(deltas),
            "wins_all_seeds": all(delta < 0 for delta in deltas),
            "clears_gate": (
                statistics.fmean(deltas) <= -0.01
                and all(delta < 0 for delta in deltas)
            ),
        }
        summaries = [
            json.loads((run_dir(arm, seed) / "training_summary.json").read_text())
            for seed in SEEDS
        ]
        throughput = [row["target_tokens_per_second"] for row in summaries]
        results["throughput"][arm] = {
            "by_seed": throughput,
            "mean": statistics.fmean(throughput),
            "std": statistics.stdev(throughput),
        }
    return results


def render_markdown(results: dict) -> str:
    lines = [
        "# Phase-23 bounded content-frequency results",
        "",
        "Deltas are candidate minus fixed RoPE; negative favors candidate.",
        "Context 1024 is the sole endpoint.",
        "",
        "| Arm | Seed 123 | Seed 456 | Seed 789 | Mean | Wins all? | Clears 0.01? |",
        "| --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for arm in ARMS:
        result = results["arms"][arm]
        deltas = [
            row["delta_candidate_minus_reference"]
            for row in result["seed_results"]
        ]
        lines.append(
            f"| {arm} | {deltas[0]:+.6f} | {deltas[1]:+.6f} | "
            f"{deltas[2]:+.6f} | {result['mean_delta']:+.6f} | "
            f"{result['wins_all_seeds']} | {result['clears_gate']} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    results = analyze()
    rendered = render_markdown(results)
    (OUTPUT_ROOT / "dynamic_frequency_analysis.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n"
    )
    (OUTPUT_ROOT / "DYNAMIC_FREQUENCY_RESULTS.md").write_text(rendered)
    print(rendered)


if __name__ == "__main__":
    main()

