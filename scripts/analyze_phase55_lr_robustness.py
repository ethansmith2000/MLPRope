#!/usr/bin/env python
"""Analyze the frozen three-point RoPE/scalar learning-rate comparison."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "sweep_configs" / "phase55_lr_robustness"
RUN_ROOT = ROOT / "model-output" / "position_bias_phase55_lr_robustness"
RESULT_ROOT = ROOT / "results" / "phase55_lr_robustness"
STEP = 100_000
HOLDOUT_START = 6_144
BLOCKS = 1_024
LRS = (1.5e-4, 3.0e-4, 6.0e-4)
ARMS = ("rope", "scalar-qkpre")


def _lr_key(value: float) -> str:
    return f"{value:.1e}"


def _new_configs() -> dict[tuple[float, str], dict]:
    result = {}
    for path in sorted(CONFIG_ROOT.glob("*.json")):
        config = json.loads(path.read_text())
        result[(float(config["learning_rate"]), "scalar-qkpre" if config["qk_preprojection"]["enabled"] else "rope")] = config
    if set(result) != {(lr, arm) for lr in (1.5e-4, 6e-4) for arm in ARMS}:
        raise ValueError("Phase-55 config matrix is incomplete")
    return result


def _new_losses(config: dict) -> np.ndarray:
    run_dir = Path(config["output_dir"])
    marker = json.loads((run_dir / "COMPLETED").read_text())
    if int(marker.get("completed_steps", -1)) != STEP:
        raise ValueError(f"Incomplete run: {run_dir}")
    path = run_dir / "evaluation_details" / f"step_{STEP:08d}_context_001024.json"
    payload = json.loads(path.read_text())
    values = np.asarray(payload["losses"], dtype=np.float64)
    if (
        payload.get("evaluation_kind") != "final_holdout"
        or payload.get("evaluation_start_batch") != HOLDOUT_START
        or values.shape != (BLOCKS,)
        or not np.isfinite(values).all()
    ):
        raise ValueError(f"Invalid final evaluation: {path}")
    return values


def _reference_losses(arm: str) -> np.ndarray:
    path = RESULT_ROOT / f"reference_lr3e4_{arm}.json"
    payload = json.loads(path.read_text())
    values = np.asarray(payload["losses"], dtype=np.float64)
    if (
        float(payload.get("learning_rate")) != 3e-4
        or payload.get("evaluation_start_batch") != HOLDOUT_START
        or payload.get("evaluation_blocks") != BLOCKS
        or values.shape != (BLOCKS,)
        or not np.isfinite(values).all()
    ):
        raise ValueError(f"Invalid reference: {path}")
    return values


def _interval(values: np.ndarray, seed: int, block_size: int | None = None) -> list[float]:
    if block_size is not None:
        if values.size % block_size:
            raise ValueError("Block size must divide evaluation examples")
        values = values.reshape(-1, block_size).mean(axis=1)
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(10):
        index = rng.integers(0, values.size, size=(2_000, values.size))
        samples.append(values[index].mean(axis=1))
    return [float(x) for x in np.quantile(np.concatenate(samples), (0.025, 0.975))]


def _run_health(config: dict) -> dict:
    run_dir = Path(config["output_dir"])
    summary = json.loads((run_dir / "training_summary.json").read_text())
    rows = [
        json.loads(line)
        for line in (run_dir / "metrics.jsonl").read_text().splitlines()
        if line
    ]
    numeric = [
        float(value)
        for row in rows
        for value in row.values()
        if isinstance(value, (int, float))
    ]
    return {
        "target_tokens_per_second": float(summary["target_tokens_per_second"]),
        "elapsed_seconds": float(summary["elapsed_seconds"]),
        "peak_reserved_mib": float(summary["peak_reserved_mib"]),
        "metrics_finite": all(math.isfinite(value) for value in numeric),
    }


def analyze() -> dict:
    configs = _new_configs()
    losses = {(3e-4, arm): _reference_losses(arm) for arm in ARMS}
    health = {}
    for key, config in configs.items():
        losses[key] = _new_losses(config)
        health[f"{_lr_key(key[0])}_{key[1]}"] = _run_health(config)

    grid = {}
    contrasts = {}
    for lr_index, lr in enumerate(LRS):
        key = _lr_key(lr)
        grid[key] = {
            arm: float(losses[(lr, arm)].mean())
            for arm in ARMS
        }
        delta = losses[(lr, "scalar-qkpre")] - losses[(lr, "rope")]
        contrasts[key] = {
            "mean_delta": float(delta.mean()),
            "iid_bootstrap_ci95": _interval(delta, 55_000 + lr_index),
            "contiguous_block_32_bootstrap_ci95": _interval(
                delta, 55_100 + lr_index, block_size=32
            ),
        }

    best = {}
    for arm in ARMS:
        lr = min(LRS, key=lambda value: losses[(value, arm)].mean())
        best[arm] = {"learning_rate": lr, "mean_loss": float(losses[(lr, arm)].mean())}
    best_to_best = best["scalar-qkpre"]["mean_loss"] - best["rope"]["mean_loss"]
    outer_negative = all(
        contrasts[_lr_key(lr)]["mean_delta"] < 0
        for lr in (1.5e-4, 6e-4)
    )
    return {
        "scope": "phase55_lr_robustness",
        "training_seed": 123,
        "paired_initialization_seed": 123,
        "training_steps": STEP,
        "sequence_batch": 32,
        "context": 1_024,
        "holdout": {"start_batch": HOLDOUT_START, "blocks": BLOCKS},
        "grid": grid,
        "matched_contrasts": contrasts,
        "best_by_arm": best,
        "best_scalar_minus_best_rope": best_to_best,
        "run_health": health,
        "gates": {
            "outer_lr_deltas_both_negative": outer_negative,
            "best_scalar_beats_best_rope": best_to_best < 0,
            "all_new_run_metrics_finite": all(item["metrics_finite"] for item in health.values()),
            "strong_robustness": bool(outer_negative and best_to_best < 0),
        },
        "inference_limit": (
            "Seed 123 audits optimizer sensitivity around an already replicated central LR; "
            "it is not an additional training-seed replication or exhaustive tuning."
        ),
    }


def render(payload: dict) -> str:
    lines = [
        "# Phase 55: symmetric learning-rate robustness",
        "",
        "| Peak LR | RoPE NLL | Scalar NLL | Scalar - RoPE | IID 95% interval | Block-32 95% interval |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for lr in LRS:
        key = _lr_key(lr)
        grid = payload["grid"][key]
        contrast = payload["matched_contrasts"][key]
        iid = contrast["iid_bootstrap_ci95"]
        block = contrast["contiguous_block_32_bootstrap_ci95"]
        lines.append(
            f"| `{key}` | {grid['rope']:.6f} | {grid['scalar-qkpre']:.6f} | "
            f"{contrast['mean_delta']:+.6f} | [{iid[0]:+.6f}, {iid[1]:+.6f}] | "
            f"[{block[0]:+.6f}, {block[1]:+.6f}] |"
        )
    rope_best = payload["best_by_arm"]["rope"]
    scalar_best = payload["best_by_arm"]["scalar-qkpre"]
    lines.extend([
        "",
        "## Frozen decisions",
        "",
        f"- Best RoPE: `{rope_best['learning_rate']:.1e}`, NLL `{rope_best['mean_loss']:.6f}`.",
        f"- Best scalar: `{scalar_best['learning_rate']:.1e}`, NLL `{scalar_best['mean_loss']:.6f}`.",
        f"- Best scalar minus best RoPE: `{payload['best_scalar_minus_best_rope']:+.6f}`.",
        f"- Both outer-LR matched deltas negative: **{payload['gates']['outer_lr_deltas_both_negative']}**.",
        f"- Strong robustness gate: **{payload['gates']['strong_robustness']}**.",
        "",
        payload["inference_limit"],
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
