#!/usr/bin/env python
"""Analyze the three-seed Phase-26 survivor replication."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PHASE26_CONFIG_ROOT = ROOT / "sweep_configs" / "phase26_position_breadth"
PHASE27_CONFIG_ROOT = ROOT / "sweep_configs" / "phase27_position_survivor_replication"
PHASE26_OUTPUT_ROOT = ROOT / "model-output" / "position_bias_phase26_breadth"
PHASE27_OUTPUT_ROOT = ROOT / "model-output" / "position_bias_phase27_survivor_replication"
RESULT_ROOT = ROOT / "results" / "phase27_position_survivor_replication"
SEEDS = (123, 456, 789)
ARMS = ("rope-fixed", "qkpre-rope", "posgain-qk")
STEPS = 5_000
CONTRASTS = (
    ("qkpre-rope_vs_rope-fixed", "qkpre-rope", "rope-fixed"),
    ("posgain-qk_vs_rope-fixed", "posgain-qk", "rope-fixed"),
    ("qkpre-rope_vs_posgain-qk", "qkpre-rope", "posgain-qk"),
)


def run_dir(arm: str, seed: int) -> Path:
    if seed == 123:
        return PHASE26_OUTPUT_ROOT / f"phase26-{arm}-seed123-s{STEPS}-h768d8"
    return PHASE27_OUTPUT_ROOT / f"phase27-{arm}-seed{seed}-s{STEPS}-h768d8"


def config_path(arm: str, seed: int) -> Path:
    if seed == 123:
        return PHASE26_CONFIG_ROOT / f"phase26-{arm}-seed123-s{STEPS}-h768d8.json"
    return PHASE27_CONFIG_ROOT / f"phase27-{arm}-seed{seed}-s{STEPS}-h768d8.json"


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
    losses = [float(value) for value in payload["losses"]]
    if len(losses) != 256:
        raise ValueError(f"Expected 256 final-holdout losses in {path}")
    return losses


def load_final_metrics(arm: str, seed: int) -> dict:
    path = run_dir(arm, seed) / "metrics.jsonl"
    final = None
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if row.get("evaluation_kind") == "final_holdout":
            final = row
    if final is None:
        raise ValueError(f"No final-holdout metrics in {path}")
    return final


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
    selected = {}
    suffixes = {
        "position_gain": (
            "q_gain_min",
            "q_gain_max",
            "q_near_bound_fraction",
            "k_gain_min",
            "k_gain_max",
            "k_near_bound_fraction",
        ),
        "qk_preprojection": (
            "gate",
            "input_rms",
            "projected_q_rms",
            "projected_k_rms",
        ),
    }
    for mechanism, names in suffixes.items():
        if not any(f"/{mechanism}/" in key for key in metrics):
            continue
        summary = {}
        for name in names:
            values = [
                float(value)
                for key, value in metrics.items()
                if f"/{mechanism}/" in key and key.endswith(f"/{name}")
            ]
            if values:
                summary[name] = {
                    "layer_min": min(values),
                    "layer_mean": statistics.fmean(values),
                    "layer_max": max(values),
                }
        selected[mechanism] = summary
    return selected


def analyze() -> dict:
    missing = [
        str(run_dir(arm, seed) / "COMPLETED")
        for arm in ARMS
        for seed in SEEDS
        if not (run_dir(arm, seed) / "COMPLETED").is_file()
    ]
    if missing:
        raise RuntimeError(
            "Survivor replication is incomplete; missing markers:\n"
            + "\n".join(missing)
        )

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
            "median_target_tokens_per_second": statistics.median(
                throughput.values()
            ),
            "mechanism_health_by_seed": {
                str(seed): mechanism_health(load_final_metrics(arm, seed))
                for seed in SEEDS
            },
        }

    contrasts = {}
    for name, candidate, reference in CONTRASTS:
        rows = []
        for seed in SEEDS:
            row = paired_summary(losses[candidate][seed], losses[reference][seed])
            row["seed"] = seed
            rows.append(row)
        deltas = [row["delta_candidate_minus_reference"] for row in rows]
        mean_delta = statistics.fmean(deltas)
        contrasts[name] = {
            "candidate": candidate,
            "reference": reference,
            "negative_favors_candidate": True,
            "seed_results": rows,
            "mean_delta_across_seeds": mean_delta,
            "seed_delta_std": statistics.stdev(deltas),
            "candidate_wins_all_seeds": all(delta < 0 for delta in deltas),
            "clears_replication_gate": (
                mean_delta <= -0.02 and all(delta < 0 for delta in deltas)
            ),
        }
    return {
        "scope": "three_seed_survivor_replication",
        "seeds": list(SEEDS),
        "primary_context": 1_024,
        "final_holdout_examples_per_seed": 256,
        "final_holdout_start_batch": 2_048,
        "protocol": protocol,
        "arms": arms,
        "contrasts": contrasts,
        "caveat": (
            "Paired-example confidence intervals quantify holdout precision "
            "within a seed; seed delta dispersion is the replication evidence."
        ),
    }


def render_markdown(results: dict) -> str:
    lines = [
        "# Phase-27 position-survivor replication",
        "",
        "Seed 123 comes from the frozen Phase-26 r3 screen; seeds 456 and 789",
        "are fresh Phase-27 runs with an identical core source and protocol.",
        "Deltas are candidate minus reference; negative is better.",
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
            "| Contrast | Seed 123 | Seed 456 | Seed 789 | Mean | All wins? | Gate? |",
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
            f"{result['clears_replication_gate']} |"
        )
    lines.extend(
        [
            "",
            "The JSON companion contains per-seed paired-example confidence",
            "intervals, seed dispersion, protocol fingerprints, throughput, and",
            "mechanism-health diagnostics.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    results = analyze()
    rendered = render_markdown(results)
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    (RESULT_ROOT / "phase27_analysis.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n"
    )
    (RESULT_ROOT / "PHASE27_RESULTS.md").write_text(rendered)
    print(rendered)


if __name__ == "__main__":
    main()
