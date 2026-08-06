#!/usr/bin/env python
"""Analyze paired phase-22 fixed-vs-additive 30k results."""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

import torch

from position import build_rope_frequencies


ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT / "model-output" / "position_bias_phase22_rope_additive_30k"
SEEDS = (123, 456, 789)


def run_dir(arm: str, seed: int) -> Path:
    return OUTPUT_ROOT / f"phase22-{arm}-seed{seed}-s30000-h768d8"


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


def spectrum_summary(seed: int) -> dict:
    state = torch.load(
        run_dir("additive", seed) / "pytorch_model.bin",
        map_location="cpu",
        weights_only=True,
    )
    raw = torch.stack(
        [
            state[f"blocks.{layer}.attn.rope_log_frequency_delta"].float()
            for layer in range(8)
        ]
    )
    base = build_rope_frequencies(96, 10_000.0)[None]
    frequency = base + raw
    spacing = frequency.abs().clamp_min(1e-30).log().diff(dim=-1).abs()
    return {
        "raw_rms": float(raw.square().mean().sqrt()),
        "raw_min": float(raw.min()),
        "raw_max": float(raw.max()),
        "frequency_min": float(frequency.min()),
        "frequency_max": float(frequency.max()),
        "nonpositive_fraction": float((frequency <= 0).float().mean()),
        "near_duplicate_spacing_fraction": float((spacing < 0.01).float().mean()),
        "extra_phase_p95_at_1024": float(
            torch.quantile((1024.0 * raw).abs(), 0.95)
        ),
    }


def analyze() -> dict:
    missing = [
        str(run_dir(arm, seed))
        for arm in ("fixed", "additive")
        for seed in SEEDS
        if not (run_dir(arm, seed) / "COMPLETED").is_file()
    ]
    if missing:
        raise RuntimeError(
            "Phase-22 confirmation is incomplete; missing markers:\n"
            + "\n".join(missing)
        )
    seed_results = []
    for seed in SEEDS:
        row = paired_summary(load_losses("additive", seed), load_losses("fixed", seed))
        row["seed"] = seed
        row["spectrum"] = spectrum_summary(seed)
        seed_results.append(row)
    deltas = [row["delta_candidate_minus_reference"] for row in seed_results]
    throughput = {}
    for arm in ("fixed", "additive"):
        values = [
            json.loads((run_dir(arm, seed) / "training_summary.json").read_text())[
                "target_tokens_per_second"
            ]
            for seed in SEEDS
        ]
        throughput[arm] = {
            "by_seed": values,
            "mean": statistics.fmean(values),
            "std": statistics.stdev(values),
        }
    return {
        "primary_context": 1024,
        "seed_results": seed_results,
        "mean_delta": statistics.fmean(deltas),
        "seed_delta_std": statistics.stdev(deltas),
        "wins_all_seeds": all(delta < 0 for delta in deltas),
        "clears_materiality_gate": (
            statistics.fmean(deltas) <= -0.01
            and all(delta < 0 for delta in deltas)
        ),
        "throughput": throughput,
    }


def render_markdown(results: dict) -> str:
    rows = results["seed_results"]
    lines = [
        "# Phase-22 additive-frequency 30k results",
        "",
        "Deltas are additive minus fixed; negative favors additive.",
        "Context 1024 is the sole endpoint.",
        "",
        "| Seed | Additive loss | Fixed loss | Delta | Paired 95% CI |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        ci = row["paired_example_ci95"]
        lines.append(
            f"| {row['seed']} | {row['candidate_loss']:.6f} | "
            f"{row['reference_loss']:.6f} | "
            f"{row['delta_candidate_minus_reference']:+.6f} | "
            f"[{ci[0]:+.6f}, {ci[1]:+.6f}] |"
        )
    lines.extend(
        [
            "",
            f"Mean delta: `{results['mean_delta']:+.6f}`. "
            f"Wins all seeds: `{results['wins_all_seeds']}`. "
            f"Clears 0.01 gate: `{results['clears_materiality_gate']}`.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    results = analyze()
    rendered = render_markdown(results)
    (OUTPUT_ROOT / "additive_30k_analysis.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n"
    )
    (OUTPUT_ROOT / "ADDITIVE_30K_RESULTS.md").write_text(rendered)
    print(rendered)


if __name__ == "__main__":
    main()

