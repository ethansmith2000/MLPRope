#!/usr/bin/env python
"""Analyze the paired phase-25 additive-carrier 30k promotion."""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT / "model-output" / "position_bias_phase25_rope_embed_basis_30k"
RESULT_ROOT = ROOT / "results" / "phase25_rope_embed_basis_30k"
SEEDS = (123, 456, 789)
ARMS = ("rope-fixed", "basis16-a03", "basis16-a10")
CONTRASTS = (
    ("basis16-a03_vs_rope-fixed", "basis16-a03", "rope-fixed"),
    ("basis16-a10_vs_rope-fixed", "basis16-a10", "rope-fixed"),
    ("basis16-a10_vs_basis16-a03", "basis16-a10", "basis16-a03"),
)


def run_dir(arm: str, seed: int) -> Path:
    return OUTPUT_ROOT / f"phase25-{arm}-seed{seed}-s30000-h768d8"


def load_losses(arm: str, seed: int) -> list[float]:
    path = (
        run_dir(arm, seed)
        / "evaluation_details"
        / "step_00030000_context_001024.json"
    )
    payload = json.loads(path.read_text())
    if payload["evaluation_kind"] != "final_holdout":
        raise ValueError(f"Unexpected evaluation kind in {path}")
    if payload["evaluation_start_batch"] != 2_048:
        raise ValueError(f"Unexpected holdout offset in {path}")
    losses = [float(value) for value in payload["losses"]]
    if len(losses) != 1_024:
        raise ValueError(f"Expected 1,024 final-holdout losses in {path}")
    return losses


def paired_summary(candidate: list[float], reference: list[float]) -> dict:
    differences = [
        candidate_loss - reference_loss
        for candidate_loss, reference_loss in zip(candidate, reference, strict=True)
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
        str(run_dir(arm, seed) / "COMPLETED")
        for arm in ARMS
        for seed in SEEDS
        if not (run_dir(arm, seed) / "COMPLETED").is_file()
    ]
    if missing:
        raise RuntimeError(
            "Phase-25 promotion is incomplete; missing markers:\n"
            + "\n".join(missing)
        )

    losses = {
        arm: {seed: load_losses(arm, seed) for seed in SEEDS}
        for arm in ARMS
    }
    arms = {}
    for arm in ARMS:
        means = {
            str(seed): statistics.fmean(losses[arm][seed]) for seed in SEEDS
        }
        throughput = {
            str(seed): json.loads(
                (run_dir(arm, seed) / "training_summary.json").read_text()
            )["target_tokens_per_second"]
            for seed in SEEDS
        }
        arms[arm] = {
            "loss_by_seed": means,
            "mean_loss_across_seeds": statistics.fmean(means.values()),
            "target_tokens_per_second_by_seed": throughput,
            "median_target_tokens_per_second": statistics.median(
                throughput.values()
            ),
        }

    contrasts = {}
    for name, candidate, reference in CONTRASTS:
        rows = []
        for seed in SEEDS:
            row = paired_summary(losses[candidate][seed], losses[reference][seed])
            row["seed"] = seed
            rows.append(row)
        deltas = [row["delta_candidate_minus_reference"] for row in rows]
        contrasts[name] = {
            "candidate": candidate,
            "reference": reference,
            "negative_favors_candidate": True,
            "seed_results": rows,
            "mean_delta_across_seeds": statistics.fmean(deltas),
            "seed_delta_std": statistics.stdev(deltas),
            "candidate_wins_all_seeds": all(delta < 0 for delta in deltas),
            "clears_materiality_gate": (
                statistics.fmean(deltas) <= -0.01
                and all(delta < 0 for delta in deltas)
            ),
        }
    return {
        "primary_context": 1024,
        "final_holdout_examples": 1_024,
        "final_holdout_start_batch": 2_048,
        "arms": arms,
        "contrasts": contrasts,
    }


def render_markdown(results: dict) -> str:
    lines = [
        "# Phase-25 additive-carrier 30k promotion",
        "",
        "Losses use the disjoint 1,024-example holdout beginning at validation",
        "batch 2,048. Deltas are candidate minus reference; negative is better.",
        "",
        "| Arm | Seed 123 | Seed 456 | Seed 789 | Mean | Median target tok/s |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm in ARMS:
        result = results["arms"][arm]
        by_seed = result["loss_by_seed"]
        lines.append(
            f"| {arm} | {by_seed['123']:.6f} | {by_seed['456']:.6f} | "
            f"{by_seed['789']:.6f} | {result['mean_loss_across_seeds']:.6f} | "
            f"{result['median_target_tokens_per_second']:,.0f} |"
        )
    lines.extend(
        [
            "",
            "| Contrast | Seed 123 | Seed 456 | Seed 789 | Mean | Wins all? | Clears 0.01? |",
            "| --- | ---: | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for name, result in results["contrasts"].items():
        rows = result["seed_results"]
        lines.append(
            f"| {name} | {rows[0]['delta_candidate_minus_reference']:+.6f} | "
            f"{rows[1]['delta_candidate_minus_reference']:+.6f} | "
            f"{rows[2]['delta_candidate_minus_reference']:+.6f} | "
            f"{result['mean_delta_across_seeds']:+.6f} | "
            f"{result['candidate_wins_all_seeds']} | "
            f"{result['clears_materiality_gate']} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    results = analyze()
    rendered = render_markdown(results)
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    (RESULT_ROOT / "phase25_analysis.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n"
    )
    (RESULT_ROOT / "PHASE25_RESULTS.md").write_text(rendered)
    print(rendered)


if __name__ == "__main__":
    main()
