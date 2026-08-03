#!/usr/bin/env python
"""Analyze the locked phase-19 paired confirmation results."""
from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT / "model-output" / "position_bias_phase19_confirmation"
SEEDS = (123, 456, 789)
CONTRASTS = (
    ("position-only_vs_standard-rope", "position-only", "standard-rope"),
    ("position-only_vs_mapped-addrope-a03", "position-only", "mapped-addrope-a03"),
    ("position-only_vs_rope-matched-ffn", "position-only", "rope-matched-ffn"),
    ("content-position_vs_position-only", "content-position", "position-only"),
)
ARMS = (
    "standard-rope",
    "mapped-addrope-a03",
    "position-only",
    "content-position",
    "rope-matched-ffn",
)


def run_dir(arm: str, seed: int) -> Path:
    return OUTPUT_ROOT / f"phase19-{arm}-seed{seed}-s30000-h1024d12"


def load_losses(arm: str, seed: int) -> list[float]:
    path = (
        run_dir(arm, seed)
        / "evaluation_details"
        / "step_00030000_context_001024.json"
    )
    with path.open() as handle:
        payload = json.load(handle)
    if payload["evaluation_kind"] != "final_holdout":
        raise ValueError(f"Unexpected evaluation kind in {path}")
    if payload["evaluation_start_batch"] != 2_048:
        raise ValueError(f"Unexpected holdout offset in {path}")
    return [float(value) for value in payload["losses"]]


def mean_ci95(values: list[float]) -> tuple[float, float, float, float]:
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    se = std / math.sqrt(len(values))
    return mean, std, mean - 1.96 * se, mean + 1.96 * se


def analyze() -> dict:
    missing = []
    for arm in ARMS:
        for seed in SEEDS:
            directory = run_dir(arm, seed)
            if not (directory / "COMPLETED").is_file():
                missing.append(str(directory))
    if missing:
        raise RuntimeError(
            "Confirmation suite is incomplete; missing COMPLETED markers:\n"
            + "\n".join(missing)
        )

    results = {"contrasts": {}, "throughput": {}}
    for name, candidate, reference in CONTRASTS:
        seed_rows = []
        for seed in SEEDS:
            candidate_losses = load_losses(candidate, seed)
            reference_losses = load_losses(reference, seed)
            if len(candidate_losses) != len(reference_losses):
                raise ValueError(f"Paired loss lengths differ for {name}, seed {seed}")
            differences = [
                candidate_loss - reference_loss
                for candidate_loss, reference_loss in zip(
                    candidate_losses,
                    reference_losses,
                    strict=True,
                )
            ]
            mean, std, low, high = mean_ci95(differences)
            seed_rows.append(
                {
                    "seed": seed,
                    "candidate_loss": statistics.fmean(candidate_losses),
                    "reference_loss": statistics.fmean(reference_losses),
                    "delta_candidate_minus_reference": mean,
                    "paired_example_std": std,
                    "paired_example_ci95": [low, high],
                    "num_examples": len(differences),
                }
            )
        seed_deltas = [row["delta_candidate_minus_reference"] for row in seed_rows]
        results["contrasts"][name] = {
            "candidate": candidate,
            "reference": reference,
            "negative_favors_candidate": True,
            "seed_results": seed_rows,
            "mean_delta_across_seeds": statistics.fmean(seed_deltas),
            "seed_delta_std": statistics.stdev(seed_deltas),
            "candidate_wins_all_seeds": all(delta < 0 for delta in seed_deltas),
        }

    for arm in ARMS:
        summaries = []
        for seed in SEEDS:
            path = run_dir(arm, seed) / "training_summary.json"
            with path.open() as handle:
                summaries.append(json.load(handle))
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
        "# Phase-19 confirmation results",
        "",
        "Deltas are candidate minus reference; negative values favor the candidate.",
        "",
        "| Contrast | Seed 123 | Seed 456 | Seed 789 | Mean | Wins all? |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for name, result in results["contrasts"].items():
        deltas = [
            row["delta_candidate_minus_reference"]
            for row in result["seed_results"]
        ]
        lines.append(
            f"| {name} | {deltas[0]:+.6f} | {deltas[1]:+.6f} | "
            f"{deltas[2]:+.6f} | {result['mean_delta_across_seeds']:+.6f} | "
            f"{result['candidate_wins_all_seeds']} |"
        )
    lines.extend(
        [
            "",
            "Per-seed paired-example confidence intervals and structured throughput "
            "measurements are in `confirmation_analysis.json`.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    results = analyze()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "confirmation_analysis.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n"
    )
    (OUTPUT_ROOT / "CONFIRMATION_RESULTS.md").write_text(
        render_markdown(results)
    )
    print(render_markdown(results))


if __name__ == "__main__":
    main()
