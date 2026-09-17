#!/usr/bin/env python
"""Aggregate Phase-53 mechanism evidence across methods and training seeds."""

from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_phase53_paper_mechanism import (
    ARMS,
    ATTENTION_BLOCKS,
    ATTENTION_METRIC_NAMES,
    LOSS_BLOCKS,
    POSITION_BIN_EDGES,
    RESULT_ROOT,
    SEEDS,
    _run_dir,
)


T_CRITICAL_95_DF2 = 4.302652729911275
CONTRASTS = {
    "scalar_minus_rope": ("scalar-qkpre", "rope"),
    "rank32_minus_rope": ("qk-readout-r32", "rope"),
    "rank32_minus_scalar": ("qk-readout-r32", "scalar-qkpre"),
}


def _load(seed: int, arm: str) -> tuple[dict, dict[str, np.ndarray]]:
    stem = RESULT_ROOT / f"seed{seed}_{arm}"
    payload = json.loads(stem.with_suffix(".json").read_text())
    with np.load(stem.with_suffix(".npz")) as archive:
        arrays = {name: archive[name] for name in archive.files}
    expected_shape = (ATTENTION_BLOCKS, 8, 8, len(ATTENTION_METRIC_NAMES))
    if arrays["attention_metrics"].shape != expected_shape:
        raise ValueError(
            f"Unexpected attention shape for seed {seed} {arm}: "
            f"{arrays['attention_metrics'].shape}"
        )
    if arrays["per_block_mean_loss"].shape != (LOSS_BLOCKS,):
        raise ValueError(f"Unexpected loss shape for seed {seed} {arm}")
    if arrays["per_block_position_bin_loss"].shape != (
        LOSS_BLOCKS,
        len(POSITION_BIN_EDGES) - 1,
    ):
        raise ValueError(f"Unexpected position-bin shape for seed {seed} {arm}")
    if payload.get("attention_metric_names") != list(ATTENTION_METRIC_NAMES):
        raise ValueError(f"Metric schema mismatch for seed {seed} {arm}")
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise ValueError(f"Non-finite arrays for seed {seed} {arm}")
    return payload, arrays


def _seed_stats(values: list[float]) -> dict:
    mean = statistics.mean(values)
    sd = statistics.stdev(values)
    radius = T_CRITICAL_95_DF2 * sd / math.sqrt(len(values))
    return {
        "by_seed": {str(seed): value for seed, value in zip(SEEDS, values, strict=True)},
        "seed_mean": mean,
        "seed_sample_std": sd,
        "seed_t_interval95": [mean - radius, mean + radius],
    }


def _block_bootstrap(delta: np.ndarray, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(10):
        indices = rng.integers(0, delta.size, size=(2_000, delta.size))
        samples.append(delta[indices].mean(axis=1))
    return [
        float(x)
        for x in np.quantile(np.concatenate(samples), (0.025, 0.975))
    ]


def _saved_endpoint(seed: int, arm: str) -> float:
    path = _run_dir(seed, arm) / "evaluation_details" / (
        "step_00100000_context_001024.json"
    )
    payload = json.loads(path.read_text())
    return float(np.asarray(payload["losses"], dtype=np.float64).mean())


def analyze() -> dict:
    loaded = {
        seed: {arm: _load(seed, arm) for arm in ARMS}
        for seed in SEEDS
    }
    endpoints = {
        arm: {
            str(seed): float(loaded[seed][arm][1]["per_block_mean_loss"].mean())
            for seed in SEEDS
        }
        for arm in ARMS
    }
    reproduction = {}
    for seed in SEEDS:
        for arm in ARMS:
            measured = endpoints[arm][str(seed)]
            saved = _saved_endpoint(seed, arm)
            difference = measured - saved
            if abs(difference) > 5e-4:
                raise ValueError(
                    f"Endpoint reproduction failed for seed {seed} {arm}: {difference}"
                )
            reproduction[f"seed{seed}_{arm}"] = {
                "measured": measured,
                "saved": saved,
                "difference": difference,
            }

    position_bins = {}
    for arm in ARMS:
        per_seed = np.stack(
            [
                loaded[seed][arm][1]["per_block_position_bin_loss"].mean(axis=0)
                for seed in SEEDS
            ]
        )
        position_bins[arm] = {
            "by_seed": {
                str(seed): [float(x) for x in per_seed[index]]
                for index, seed in enumerate(SEEDS)
            },
            "seed_mean": [float(x) for x in per_seed.mean(axis=0)],
            "seed_sample_std": [float(x) for x in per_seed.std(axis=0, ddof=1)],
        }

    endpoint_contrasts = {}
    position_contrasts = {}
    for name, (candidate, reference) in CONTRASTS.items():
        endpoint_deltas = []
        bin_deltas = []
        by_seed_bins = {}
        for seed in SEEDS:
            candidate_arrays = loaded[seed][candidate][1]
            reference_arrays = loaded[seed][reference][1]
            block_delta = (
                candidate_arrays["per_block_mean_loss"]
                - reference_arrays["per_block_mean_loss"]
            )
            endpoint_deltas.append(float(block_delta.mean()))
            candidate_bins = candidate_arrays["per_block_position_bin_loss"]
            reference_bins = reference_arrays["per_block_position_bin_loss"]
            current_bins = (candidate_bins - reference_bins).mean(axis=0)
            bin_deltas.append(current_bins)
            by_seed_bins[str(seed)] = [float(x) for x in current_bins]
        endpoint_contrasts[name] = _seed_stats(endpoint_deltas)
        bin_deltas_array = np.stack(bin_deltas)
        position_contrasts[name] = {
            "by_seed": by_seed_bins,
            "seed_mean": [float(x) for x in bin_deltas_array.mean(axis=0)],
            "seed_sample_std": [
                float(x) for x in bin_deltas_array.std(axis=0, ddof=1)
            ],
        }

    metric_index = {name: index for index, name in enumerate(ATTENTION_METRIC_NAMES)}
    attention = {}
    for arm in ARMS:
        seed_global = []
        seed_layer = []
        for seed in SEEDS:
            values = loaded[seed][arm][1]["attention_metrics"]
            seed_global.append(values.mean(axis=(0, 1, 2)))
            seed_layer.append(values.mean(axis=(0, 2)))
        global_array = np.stack(seed_global)
        layer_array = np.stack(seed_layer)
        attention[arm] = {
            "global_by_seed": {
                str(seed): {
                    metric: float(global_array[seed_index, index])
                    for index, metric in enumerate(ATTENTION_METRIC_NAMES)
                }
                for seed_index, seed in enumerate(SEEDS)
            },
            "global_seed_mean": {
                metric: float(global_array[:, index].mean())
                for index, metric in enumerate(ATTENTION_METRIC_NAMES)
            },
            "global_seed_sample_std": {
                metric: float(global_array[:, index].std(ddof=1))
                for index, metric in enumerate(ATTENTION_METRIC_NAMES)
            },
            "layer_seed_mean": {
                metric: [float(x) for x in layer_array[:, :, index].mean(axis=0)]
                for index, metric in enumerate(ATTENTION_METRIC_NAMES)
            },
        }

    attention_contrasts = {}
    for contrast_index, (name, (candidate, reference)) in enumerate(CONTRASTS.items()):
        metrics = {}
        for metric, index in metric_index.items():
            seed_means = []
            by_seed = {}
            for seed_index, seed in enumerate(SEEDS):
                candidate_values = loaded[seed][candidate][1]["attention_metrics"][:, :, :, index]
                reference_values = loaded[seed][reference][1]["attention_metrics"][:, :, :, index]
                per_block = (candidate_values - reference_values).mean(axis=(1, 2))
                mean = float(per_block.mean())
                seed_means.append(mean)
                by_seed[str(seed)] = {
                    "mean": mean,
                    "paired_block_bootstrap_ci95": _block_bootstrap(
                        per_block,
                        53_000 + contrast_index * 1_000 + index * 10 + seed_index,
                    ),
                }
            metrics[metric] = {**_seed_stats(seed_means), "within_seed": by_seed}
        attention_contrasts[name] = metrics

    max_reconstruction_error = max(
        loaded[seed][arm][0]["maximum_logit_reconstruction_error"]
        for seed in SEEDS
        for arm in ARMS
    )
    return {
        "scope": "phase53_paper_mechanism",
        "methods": list(ARMS),
        "seeds": list(SEEDS),
        "loss_holdout": {"start_batch": 4_096, "blocks": LOSS_BLOCKS},
        "attention_sample": {
            "blocks_per_model": ATTENTION_BLOCKS,
            "selection": "offset 8 then every 16th block within the holdout",
        },
        "position_bins": [
            {"target_start": start, "target_stop_exclusive": stop}
            for start, stop in zip(POSITION_BIN_EDGES[:-1], POSITION_BIN_EDGES[1:])
        ],
        "endpoints": endpoints,
        "endpoint_reproduction": reproduction,
        "endpoint_contrasts": endpoint_contrasts,
        "position_loss": position_bins,
        "position_loss_contrasts": position_contrasts,
        "attention": attention,
        "attention_contrasts": attention_contrasts,
        "maximum_logit_reconstruction_error": max_reconstruction_error,
        "interpretation_limits": [
            "The training seed is the replication unit.",
            "Attention block intervals quantify sampling precision within a seed; layers and heads are not treated as independent replicates.",
            "Logit components use the trained hidden state and the common Q/K RMS denominator. Their sum is algebraically exact and is checked numerically against logits recomputed from full Q/K; the components remain descriptive local decompositions rather than independently runnable models.",
            "The attention sample is a predeclared evenly spaced subset of the same frozen holdout and is not a new corpus.",
        ],
    }


def _write_csvs(payload: dict) -> None:
    with (RESULT_ROOT / "position_loss.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("arm", "seed", "target_start", "target_stop_exclusive", "nll"))
        for arm in ARMS:
            for seed in SEEDS:
                values = payload["position_loss"][arm]["by_seed"][str(seed)]
                for bin_spec, value in zip(payload["position_bins"], values, strict=True):
                    writer.writerow((
                        arm, seed, bin_spec["target_start"],
                        bin_spec["target_stop_exclusive"], value,
                    ))
    with (RESULT_ROOT / "attention_global.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("arm", "seed", "metric", "value"))
        for arm in ARMS:
            for seed in SEEDS:
                for metric, value in payload["attention"][arm]["global_by_seed"][str(seed)].items():
                    writer.writerow((arm, seed, metric, value))


def render(payload: dict) -> str:
    lines = [
        "# Phase 53: paper mechanism measurements",
        "",
        "All results use the same retained checkpoints and frozen holdout. "
        "Negative NLL deltas favor the first named method.",
        "",
        "## Endpoint reproduction",
        "",
        "| Arm | Seed 123 | Seed 456 | Seed 789 |",
        "|---|---:|---:|---:|",
    ]
    for arm in ARMS:
        values = payload["endpoints"][arm]
        lines.append(
            f"| {arm} | {values['123']:.6f} | {values['456']:.6f} | {values['789']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## NLL by target-position bin",
            "",
            "| Target positions | Scalar - RoPE | Rank 32 - RoPE | Rank 32 - scalar |",
            "|---|---:|---:|---:|",
        ]
    )
    for index, bin_spec in enumerate(payload["position_bins"]):
        label = f"{bin_spec['target_start']}–{bin_spec['target_stop_exclusive'] - 1}"
        lines.append(
            f"| {label} | "
            f"{payload['position_loss_contrasts']['scalar_minus_rope']['seed_mean'][index]:+.6f} | "
            f"{payload['position_loss_contrasts']['rank32_minus_rope']['seed_mean'][index]:+.6f} | "
            f"{payload['position_loss_contrasts']['rank32_minus_scalar']['seed_mean'][index]:+.6f} |"
        )
    selected_metrics = (
        "entropy_normalized",
        "attended_distance_fraction",
        "first_token_mass",
        "distance_mass_0",
        "distance_mass_1_3",
        "distance_mass_256_plus",
    )
    lines.extend(
        [
            "",
            "## Global attention geometry",
            "",
            "Means first average sampled blocks, layers, and heads within each seed, then average seeds.",
            "",
            "| Metric | RoPE | Scalar | Rank 32 |",
            "|---|---:|---:|---:|",
        ]
    )
    for metric in selected_metrics:
        lines.append(
            f"| {metric} | "
            f"{payload['attention']['rope']['global_seed_mean'][metric]:.6f} | "
            f"{payload['attention']['scalar-qkpre']['global_seed_mean'][metric]:.6f} | "
            f"{payload['attention']['qk-readout-r32']['global_seed_mean'][metric]:.6f} |"
        )
    component_metrics = (
        "logit_cc_centered_rms",
        "logit_cp_centered_rms",
        "logit_pc_centered_rms",
        "logit_pp_centered_rms",
        "logit_position_total_centered_rms",
        "logit_position_total_cosine",
        "attention_kl_full_vs_content_only",
    )
    lines.extend(
        [
            "",
            "## Local positional-logit decomposition",
            "",
            "| Metric | Scalar | Rank 32 |",
            "|---|---:|---:|",
        ]
    )
    for metric in component_metrics:
        lines.append(
            f"| {metric} | "
            f"{payload['attention']['scalar-qkpre']['global_seed_mean'][metric]:.6f} | "
            f"{payload['attention']['qk-readout-r32']['global_seed_mean'][metric]:.6f} |"
        )
    lines.extend(
        [
            "",
            f"Maximum four-term logit reconstruction error: "
            f"`{payload['maximum_logit_reconstruction_error']:.3e}`.",
            "",
            "## Interpretation limits",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in payload["interpretation_limits"])
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    payload = analyze()
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    (RESULT_ROOT / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    (RESULT_ROOT / "REPORT.md").write_text(render(payload))
    _write_csvs(payload)
    print(render(payload))


if __name__ == "__main__":
    main()
