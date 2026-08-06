#!/usr/bin/env python
"""Analyze paired phase-21 parameterizations against phase-20 controls."""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PHASE20_ROOT = ROOT / "model-output" / "position_bias_phase20_rope_frequency"
OUTPUT_ROOT = ROOT / "model-output" / "position_bias_phase21_rope_parameterization"
SEEDS = (123, 456, 789)
ARMS = (
    "fixed",
    "exp",
    "exp-full-ste",
    "softplus",
    "additive",
    "bounded-log",
)


def run_dir(arm: str, seed: int) -> Path:
    if arm == "fixed":
        return PHASE20_ROOT / f"phase20-fixed-seed{seed}-s5000-h768d8"
    if arm == "exp":
        return PHASE20_ROOT / f"phase20-layer-shared-seed{seed}-s5000-h768d8"
    return OUTPUT_ROOT / f"phase21-{arm}-seed{seed}-s5000-h768d8"


def load_losses(arm: str, seed: int) -> list[float]:
    path = (
        run_dir(arm, seed)
        / "evaluation_details"
        / "step_00005000_context_001024.json"
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
            "Frequency parameterization screen is incomplete; missing markers:\n"
            + "\n".join(missing)
        )
    results: dict = {
        "primary_context": 1024,
        "seeds": list(SEEDS),
        "arms": {},
        "throughput": {},
    }
    fixed_losses = {seed: load_losses("fixed", seed) for seed in SEEDS}
    exp_losses = {seed: load_losses("exp", seed) for seed in SEEDS}
    for arm in ARMS:
        seed_losses = {seed: load_losses(arm, seed) for seed in SEEDS}
        versus_fixed = [
            paired_summary(seed_losses[seed], fixed_losses[seed])
            for seed in SEEDS
        ]
        versus_exp = [
            paired_summary(seed_losses[seed], exp_losses[seed])
            for seed in SEEDS
        ]
        for seed, row in zip(SEEDS, versus_fixed, strict=True):
            row["seed"] = seed
        for seed, row in zip(SEEDS, versus_exp, strict=True):
            row["seed"] = seed
        fixed_deltas = [row["delta_candidate_minus_reference"] for row in versus_fixed]
        exp_deltas = [row["delta_candidate_minus_reference"] for row in versus_exp]
        results["arms"][arm] = {
            "versus_fixed": versus_fixed,
            "versus_exp": versus_exp,
            "mean_delta_vs_fixed": statistics.fmean(fixed_deltas),
            "mean_delta_vs_exp": statistics.fmean(exp_deltas),
            "wins_fixed_all_seeds": all(delta < 0 for delta in fixed_deltas),
            "wins_exp_all_seeds": all(delta < 0 for delta in exp_deltas),
        }
        summaries = [
            json.loads((run_dir(arm, seed) / "training_summary.json").read_text())
            for seed in SEEDS
        ]
        throughput = [row["target_tokens_per_second"] for row in summaries]
        results["throughput"][arm] = {
            "target_tokens_per_second_by_seed": throughput,
            "mean_target_tokens_per_second": statistics.fmean(throughput),
            "std_target_tokens_per_second": statistics.stdev(throughput),
        }
    return results


def render_markdown(results: dict) -> str:
    lines = [
        "# Phase-21 static RoPE frequency parameterization results",
        "",
        "Deltas are candidate minus reference; negative favors the candidate.",
        "Context 1024 is the sole endpoint.",
        "",
        "| Arm | Seed 123 vs fixed | Seed 456 | Seed 789 | Mean vs fixed | Mean vs exp | Wins fixed all? |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for arm in ARMS:
        result = results["arms"][arm]
        deltas = [
            row["delta_candidate_minus_reference"]
            for row in result["versus_fixed"]
        ]
        lines.append(
            f"| {arm} | {deltas[0]:+.6f} | {deltas[1]:+.6f} | "
            f"{deltas[2]:+.6f} | {result['mean_delta_vs_fixed']:+.6f} | "
            f"{result['mean_delta_vs_exp']:+.6f} | "
            f"{result['wins_fixed_all_seeds']} |"
        )
    lines.extend(
        [
            "",
            "Fixed and ordinary-exp controls are reused from phase 20. No",
            "long-context evaluation is part of this screen.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    results = analyze()
    rendered = render_markdown(results)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "parameterization_analysis.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n"
    )
    (OUTPUT_ROOT / "PARAMETERIZATION_RESULTS.md").write_text(rendered)
    print(rendered)


if __name__ == "__main__":
    main()

