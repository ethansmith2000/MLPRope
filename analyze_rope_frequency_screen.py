#!/usr/bin/env python
"""Analyze paired phase-20 learned-RoPE frequency results."""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT / "model-output" / "position_bias_phase20_rope_frequency"
ARMS = ("fixed", "layer-shared", "layer-head")
SEEDS = (123, 456, 789)
CONTEXTS = (1024, 2048, 4096)
CONTRASTS = (
    ("layer-shared_vs_fixed", "layer-shared", "fixed"),
    ("layer-head_vs_fixed", "layer-head", "fixed"),
    ("layer-head_vs_layer-shared", "layer-head", "layer-shared"),
)


def run_dir(arm: str, seed: int) -> Path:
    return OUTPUT_ROOT / f"phase20-{arm}-seed{seed}-s5000-h768d8"


def load_losses(arm: str, seed: int, context: int) -> list[float]:
    path = (
        run_dir(arm, seed)
        / "evaluation_details"
        / f"step_00005000_context_{context:06d}.json"
    )
    payload = json.loads(path.read_text())
    if payload["evaluation_kind"] != "final_holdout":
        raise ValueError(f"Unexpected evaluation kind in {path}")
    if payload["evaluation_start_batch"] != 2_048:
        raise ValueError(f"Unexpected holdout offset in {path}")
    return [float(value) for value in payload["losses"]]


def paired_summary(candidate: list[float], reference: list[float]) -> dict:
    if len(candidate) != len(reference):
        raise ValueError("Paired loss arrays differ in length")
    differences = [
        candidate_loss - reference_loss
        for candidate_loss, reference_loss in zip(
            candidate,
            reference,
            strict=True,
        )
    ]
    mean = statistics.fmean(differences)
    std = statistics.stdev(differences) if len(differences) > 1 else 0.0
    half_width = 1.96 * std / math.sqrt(len(differences))
    return {
        "candidate_loss": statistics.fmean(candidate),
        "reference_loss": statistics.fmean(reference),
        "delta_candidate_minus_reference": mean,
        "paired_example_std": std,
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
            "Frequency screen is incomplete; missing COMPLETED markers:\n"
            + "\n".join(missing)
        )

    results: dict = {
        "primary_context": 1024,
        "exploratory_contexts_not_used_for_decision": [2048, 4096],
        "contexts": {},
        "throughput": {},
    }
    for context in CONTEXTS:
        context_results = {}
        for name, candidate, reference in CONTRASTS:
            seed_rows = []
            for seed in SEEDS:
                row = paired_summary(
                    load_losses(candidate, seed, context),
                    load_losses(reference, seed, context),
                )
                row["seed"] = seed
                seed_rows.append(row)
            deltas = [row["delta_candidate_minus_reference"] for row in seed_rows]
            context_results[name] = {
                "candidate": candidate,
                "reference": reference,
                "negative_favors_candidate": True,
                "seed_results": seed_rows,
                "mean_delta_across_seeds": statistics.fmean(deltas),
                "seed_delta_std": statistics.stdev(deltas),
                "candidate_wins_all_seeds": all(delta < 0 for delta in deltas),
            }
        results["contexts"][str(context)] = context_results

    for arm in ARMS:
        summaries = [
            json.loads((run_dir(arm, seed) / "training_summary.json").read_text())
            for seed in SEEDS
        ]
        throughput = [row["target_tokens_per_second"] for row in summaries]
        results["throughput"][arm] = {
            "target_tokens_per_second_by_seed": throughput,
            "mean_target_tokens_per_second": statistics.fmean(throughput),
            "std_target_tokens_per_second": statistics.stdev(throughput),
            "measurement": summaries[0]["measurement"],
        }
    return results


def render_markdown(results: dict) -> str:
    lines = [
        "# Phase-20 learned RoPE frequency results",
        "",
        "Deltas are candidate minus reference; negative favors the candidate.",
        "The 1024-token training context is the sole primary endpoint.",
        "",
        "| Contrast | Seed 123 | Seed 456 | Seed 789 | Mean | Wins all? |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for name, result in results["contexts"]["1024"].items():
        deltas = [
            row["delta_candidate_minus_reference"]
            for row in result["seed_results"]
        ]
        lines.append(
            f"| {name} | {deltas[0]:+.6f} | "
            f"{deltas[1]:+.6f} | {deltas[2]:+.6f} | "
            f"{result['mean_delta_across_seeds']:+.6f} | "
            f"{result['candidate_wins_all_seeds']} |"
        )
    lines.extend(
        [
            "",
            "The already-collected 2048/4096 diagnostics are retained in "
            "`frequency_analysis.json` but are not used for the decision.",
            "Full paired intervals are also recorded there.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    results = analyze()
    rendered = render_markdown(results)
    (OUTPUT_ROOT / "frequency_analysis.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n"
    )
    (OUTPUT_ROOT / "FREQUENCY_RESULTS.md").write_text(rendered)
    print(rendered)


if __name__ == "__main__":
    main()
