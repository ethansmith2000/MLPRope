#!/usr/bin/env python
"""Aggregate Phase-54 carrier-origin sensitivity across training seeds."""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_phase54_position_origin import ARMS, OFFSETS, RESULT_ROOT, SEEDS


PHASE53_ROOT = ROOT / "results" / "phase53_paper_mechanism"
T_CRITICAL_95_DF2 = 4.302652729911275


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


def _load(seed: int, arm: str) -> tuple[dict, np.ndarray]:
    stem = RESULT_ROOT / f"seed{seed}_{arm}"
    payload = json.loads(stem.with_suffix(".json").read_text())
    with np.load(stem.with_suffix(".npz")) as archive:
        offsets = archive["offsets"]
        losses = archive["per_block_mean_loss"]
    if tuple(int(x) for x in offsets) != OFFSETS:
        raise ValueError(f"Offset mismatch for seed {seed} {arm}")
    if losses.shape != (len(OFFSETS), 1_024) or not np.isfinite(losses).all():
        raise ValueError(f"Invalid loss matrix for seed {seed} {arm}: {losses.shape}")
    return payload, losses


def _rope_loss(seed: int) -> np.ndarray:
    path = PHASE53_ROOT / f"seed{seed}_rope.npz"
    with np.load(path) as archive:
        losses = archive["per_block_mean_loss"].astype(np.float64)
    if losses.shape != (1_024,) or not np.isfinite(losses).all():
        raise ValueError(f"Invalid Phase-53 RoPE loss for seed {seed}")
    return losses


def _block_interval(values: np.ndarray, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(10):
        index = rng.integers(0, values.size, size=(2_000, values.size))
        samples.append(values[index].mean(axis=1))
    return [float(x) for x in np.quantile(np.concatenate(samples), (0.025, 0.975))]


def analyze() -> dict:
    loaded = {seed: {arm: _load(seed, arm) for arm in ARMS} for seed in SEEDS}
    carrier_cosines = loaded[SEEDS[0]][ARMS[0]][0]["carrier_cosines"]
    for seed in SEEDS:
        for arm in ARMS:
            payload, losses = loaded[seed][arm]
            if not np.allclose(payload["carrier_cosines"], carrier_cosines, atol=1e-7):
                raise ValueError("Carrier cosine mismatch")
            phase53_path = PHASE53_ROOT / f"seed{seed}_{arm}.npz"
            with np.load(phase53_path) as archive:
                expected = archive["per_block_mean_loss"].astype(np.float64)
            difference = float((losses[0] - expected).mean())
            if abs(difference) > 5e-4:
                raise ValueError(
                    f"Offset-zero reproduction failed for seed {seed} {arm}: {difference}"
                )

    arms = {}
    for arm_index, arm in enumerate(ARMS):
        rows = []
        for offset_index, offset in enumerate(OFFSETS):
            nll_values = []
            penalty_values = []
            rope_delta_values = []
            within_seed = {}
            for seed_index, seed in enumerate(SEEDS):
                losses = loaded[seed][arm][1]
                current = losses[offset_index]
                penalty = current - losses[0]
                rope_delta = current - _rope_loss(seed)
                nll_values.append(float(current.mean()))
                penalty_values.append(float(penalty.mean()))
                rope_delta_values.append(float(rope_delta.mean()))
                within_seed[str(seed)] = {
                    "penalty_mean": float(penalty.mean()),
                    "penalty_paired_block_ci95": _block_interval(
                        penalty, 54_000 + arm_index * 1_000 + offset_index * 10 + seed_index
                    ),
                    "minus_rope_mean": float(rope_delta.mean()),
                }
            rows.append({
                "offset": offset,
                "carrier_cosine": float(carrier_cosines[offset_index]),
                "nll": _seed_stats(nll_values),
                "penalty_vs_offset0": _seed_stats(penalty_values),
                "minus_rope": _seed_stats(rope_delta_values),
                "within_seed": within_seed,
            })
        arms[arm] = rows

    small_offset_material = {
        arm: any(
            row["penalty_vs_offset0"]["seed_mean"] >= 0.01
            for row in arms[arm]
            if row["offset"] in {1, 4}
        )
        for arm in ARMS
    }
    return {
        "scope": "phase54_position_origin",
        "seeds": list(SEEDS),
        "offsets": list(OFFSETS),
        "holdout": {"start_batch": 4_096, "blocks": 1_024},
        "arms": arms,
        "small_offset_material_penalty": small_offset_material,
        "interpretation_limits": [
            "This is an inference-time distribution shift, not randomized-origin training.",
            "RoPE is left at its original indices because a common RoPE-origin shift cancels in Q/K inner products.",
            "Training seed is the replication unit; block intervals describe within-seed evaluation precision.",
        ],
    }


def render(payload: dict) -> str:
    lines = [
        "# Phase 54: position-origin sensitivity",
        "",
        "NLL penalty is relative to the same checkpoint at carrier offset zero. "
        "Negative candidate-minus-RoPE values favor the carrier model.",
    ]
    for arm in ARMS:
        lines.extend([
            "",
            f"## {arm}",
            "",
            "| Offset | Carrier cosine | Mean NLL | Penalty vs 0 | Minus RoPE | Seed t interval for penalty |",
            "|---:|---:|---:|---:|---:|---:|",
        ])
        for row in payload["arms"][arm]:
            interval = row["penalty_vs_offset0"]["seed_t_interval95"]
            lines.append(
                f"| {row['offset']} | {row['carrier_cosine']:.6f} | "
                f"{row['nll']['seed_mean']:.6f} | "
                f"{row['penalty_vs_offset0']['seed_mean']:+.6f} | "
                f"{row['minus_rope']['seed_mean']:+.6f} | "
                f"[{interval[0]:+.6f}, {interval[1]:+.6f}] |"
            )
    lines.extend([
        "",
        "## Frozen decision",
        "",
        *[
            f"- {arm}: offset 1 or 4 penalty at least 0.01: "
            f"{'yes' if payload['small_offset_material_penalty'][arm] else 'no'}"
            for arm in ARMS
        ],
        "",
        "This audit is checkpoint sensitivity only. Large-offset degradation does not by itself "
        "imply that randomized-origin training would fail.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    payload = analyze()
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    (RESULT_ROOT / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    report = render(payload)
    (RESULT_ROOT / "REPORT.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
