#!/usr/bin/env python
"""Analyze the paired three-seed Phase-28 qk-preprojection promotion."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CONFIG_ROOT = ROOT / "sweep_configs" / "phase28_qkpre_rope_30k"
OUTPUT_ROOT = ROOT / "model-output" / "position_bias_phase28_qkpre_rope_30k"
RESULT_ROOT = ROOT / "results" / "phase28_qkpre_rope_30k"
SEEDS = (123, 456, 789)
ARMS = ("rope-fixed", "qkpre-rope")
STEPS = (5_000, 10_000, 15_000, 20_000, 25_000, 30_000)


def run_dir(arm: str, seed: int) -> Path:
    return OUTPUT_ROOT / f"phase28-{arm}-seed{seed}-s30000-h768d8"


def config_path(arm: str, seed: int) -> Path:
    return CONFIG_ROOT / f"phase28-{arm}-seed{seed}-s30000-h768d8.json"


def load_metrics(arm: str, seed: int) -> list[dict]:
    path = run_dir(arm, seed) / "metrics.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


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


def final_metrics(arm: str, seed: int) -> dict:
    rows = [
        row
        for row in load_metrics(arm, seed)
        if row.get("evaluation_kind") == "final_holdout"
    ]
    if len(rows) != 1:
        raise ValueError(f"Expected one final-holdout row for {arm}, seed {seed}")
    return rows[0]


def development_curve(arm: str, seed: int) -> dict[str, float]:
    rows = {
        int(row["step"]): float(row["eval_loss/context_1024"])
        for row in load_metrics(arm, seed)
        if row.get("evaluation_kind") == "development"
    }
    missing = [step for step in STEPS if step not in rows]
    if missing:
        raise ValueError(f"Missing development steps for {arm}, seed {seed}: {missing}")
    return {str(step): rows[step] for step in STEPS}


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
        "num_paired_examples": len(differences),
    }


def protocol_fingerprint(arm: str, seed: int) -> str:
    config = json.loads(config_path(arm, seed).read_text())
    for key in (
        "run_name",
        "base_output_dir",
        "output_dir",
        "seed",
        "paired_initialization_seed",
    ):
        config.pop(key, None)
    payload = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def mechanism_health(metrics: dict) -> dict:
    names = ("gate", "input_rms", "projected_q_rms", "projected_k_rms")
    summary = {}
    for name in names:
        values = [
            float(value)
            for key, value in metrics.items()
            if "/qk_preprojection/" in key and key.endswith(f"/{name}")
        ]
        if values:
            summary[name] = {
                "layer_min": min(values),
                "layer_mean": statistics.fmean(values),
                "layer_max": max(values),
            }
    return summary


def verify_arm_isolation() -> None:
    for seed in SEEDS:
        fixed = json.loads(config_path("rope-fixed", seed).read_text())
        qkpre = json.loads(config_path("qkpre-rope", seed).read_text())
        for cfg in (fixed, qkpre):
            cfg.pop("run_name", None)
        qkpre_block = qkpre.pop("qk_preprojection")
        fixed_block = fixed.pop("qk_preprojection")
        if fixed != qkpre:
            raise RuntimeError(f"Non-mechanism arm drift detected for seed {seed}")
        if fixed_block != {"enabled": False}:
            raise RuntimeError(f"Unexpected fixed arm qk block for seed {seed}")
        expected = {
            "basis_dim": 768,
            "enabled": True,
            "gate_init": 1.0,
            "learnable_gate": True,
            "theta": None,
        }
        if qkpre_block != expected:
            raise RuntimeError(f"Unexpected qkpre arm block for seed {seed}")


def analyze() -> dict:
    missing = [
        str(run_dir(arm, seed) / "COMPLETED")
        for arm in ARMS
        for seed in SEEDS
        if not (run_dir(arm, seed) / "COMPLETED").is_file()
    ]
    if missing:
        raise RuntimeError("Phase-28 is incomplete; missing markers:\n" + "\n".join(missing))

    verify_arm_isolation()
    protocol = {}
    for arm in ARMS:
        fingerprints = {str(seed): protocol_fingerprint(arm, seed) for seed in SEEDS}
        if len(set(fingerprints.values())) != 1:
            raise RuntimeError(f"Protocol drift detected for {arm}: {fingerprints}")
        protocol[arm] = {
            "fingerprint": next(iter(fingerprints.values())),
            "matches_across_seeds": True,
        }

    losses = {
        arm: {seed: load_losses(arm, seed) for seed in SEEDS}
        for arm in ARMS
    }
    arms = {}
    for arm in ARMS:
        means = {str(seed): statistics.fmean(losses[arm][seed]) for seed in SEEDS}
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
            "median_target_tokens_per_second": statistics.median(throughput.values()),
            "development_curve_by_seed": {
                str(seed): development_curve(arm, seed) for seed in SEEDS
            },
            "mechanism_health_by_seed": {
                str(seed): mechanism_health(final_metrics(arm, seed))
                for seed in SEEDS
            },
        }

    seed_results = []
    for seed in SEEDS:
        row = paired_summary(losses["qkpre-rope"][seed], losses["rope-fixed"][seed])
        row["seed"] = seed
        seed_results.append(row)
    deltas = [row["delta_candidate_minus_reference"] for row in seed_results]
    mean_delta = statistics.fmean(deltas)

    curve_deltas = {}
    for step in STEPS:
        per_seed = []
        for seed in SEEDS:
            qk = arms["qkpre-rope"]["development_curve_by_seed"][str(seed)][str(step)]
            fixed = arms["rope-fixed"]["development_curve_by_seed"][str(seed)][str(step)]
            per_seed.append(qk - fixed)
        curve_deltas[str(step)] = {
            "delta_by_seed": {str(seed): delta for seed, delta in zip(SEEDS, per_seed)},
            "mean_delta_across_seeds": statistics.fmean(per_seed),
            "candidate_wins_all_seeds": all(delta < 0 for delta in per_seed),
        }

    return {
        "scope": "three_seed_qkpre_rope_30k_promotion",
        "seeds": list(SEEDS),
        "primary_context": 1_024,
        "final_holdout_examples_per_seed": 1_024,
        "final_holdout_start_batch": 2_048,
        "protocol": protocol,
        "arm_isolation_verified": True,
        "arms": arms,
        "contrast": {
            "candidate": "qkpre-rope",
            "reference": "rope-fixed",
            "negative_favors_candidate": True,
            "seed_results": seed_results,
            "mean_delta_across_seeds": mean_delta,
            "seed_delta_std": statistics.stdev(deltas),
            "candidate_wins_all_seeds": all(delta < 0 for delta in deltas),
            "clears_30k_materiality_gate": (
                mean_delta <= -0.01 and all(delta < 0 for delta in deltas)
            ),
        },
        "development_curve_delta": curve_deltas,
        "caveat": (
            "Paired-example confidence intervals quantify holdout precision within "
            "a seed; dispersion and sign agreement across seeds are the replication evidence."
        ),
    }


def render_markdown(results: dict) -> str:
    lines = [
        "# Phase-28 qk-preprojection 30k promotion",
        "",
        "Losses use the disjoint 1,024-example holdout beginning at validation",
        "batch 2,048. Deltas are qkpre-rope minus fixed RoPE; negative is better.",
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
    contrast = results["contrast"]
    rows = contrast["seed_results"]
    lines.extend(
        [
            "",
            "| Contrast | Seed 123 | Seed 456 | Seed 789 | Mean | Wins all? | Clears 0.01? |",
            "| --- | ---: | ---: | ---: | ---: | --- | --- |",
            (
                f"| qkpre-rope vs rope-fixed | "
                f"{rows[0]['delta_candidate_minus_reference']:+.6f} | "
                f"{rows[1]['delta_candidate_minus_reference']:+.6f} | "
                f"{rows[2]['delta_candidate_minus_reference']:+.6f} | "
                f"{contrast['mean_delta_across_seeds']:+.6f} | "
                f"{contrast['candidate_wins_all_seeds']} | "
                f"{contrast['clears_30k_materiality_gate']} |"
            ),
            "",
            "| Step | Mean development-loss delta | Wins all seeds? |",
            "| ---: | ---: | --- |",
        ]
    )
    for step in STEPS:
        result = results["development_curve_delta"][str(step)]
        lines.append(
            f"| {step:,} | {result['mean_delta_across_seeds']:+.6f} | "
            f"{result['candidate_wins_all_seeds']} |"
        )
    lines.extend(
        [
            "",
            "The JSON companion contains per-seed paired-example intervals, seed",
            "dispersion, protocol fingerprints, throughput, complete development",
            "curves, and qk-preprojection mechanism-health diagnostics.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    results = analyze()
    rendered = render_markdown(results)
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    (RESULT_ROOT / "phase28_analysis.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n"
    )
    (RESULT_ROOT / "PHASE28_RESULTS.md").write_text(rendered)
    print(rendered)


if __name__ == "__main__":
    main()
