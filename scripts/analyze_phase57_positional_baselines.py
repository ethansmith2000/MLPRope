#!/usr/bin/env python
"""Analyze the frozen Phase-57 recognized positional-baseline table."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "sweep_configs" / "phase57_positional_baselines"
RESULT_ROOT = ROOT / "results" / "phase57_positional_baselines"
STEP = 100_000
FINAL_START = 7_168
FINAL_BLOCKS = 1_024
ARMS = (
    "rope",
    "scalar-qkpre",
    "nope",
    "fixed-input-sinusoid",
    "learned-absolute",
    "partial-rope-25",
    "alibi",
)
LABELS = {
    "rope": "standard RoPE",
    "scalar-qkpre": "scalar pre-Q/K + RoPE",
    "nope": "NoPE",
    "fixed-input-sinusoid": "fixed input sinusoid",
    "learned-absolute": "learned absolute",
    "partial-rope-25": "25% partial RoPE",
    "alibi": "ALiBi",
}


def _configs() -> dict[str, dict]:
    result = {}
    for path in sorted(CONFIG_ROOT.glob("*.json")):
        arm = path.stem.split("-", 1)[1]
        result[arm] = json.loads(path.read_text())
    if tuple(result) != ARMS:
        raise ValueError(f"Phase-57 config matrix is incomplete or reordered: {tuple(result)}")
    for arm, config in result.items():
        required = {
            "max_train_steps": STEP,
            "learning_rate": 1.2e-3,
            "training_length": 1_024,
            "per_device_train_batch_size": 32,
            "seed": 123,
            "paired_initialization_seed": 123,
            "final_validation_start_batch": FINAL_START,
            "num_final_validation_batches": FINAL_BLOCKS,
            "save_final_model": False,
        }
        for key, expected in required.items():
            if config.get(key) != expected:
                raise ValueError(f"Unexpected {arm} config {key}: {config.get(key)!r}")
    return result


def _losses(run_dir: Path) -> np.ndarray:
    path = run_dir / "evaluation_details" / f"step_{STEP:08d}_context_001024.json"
    payload = json.loads(path.read_text())
    values = np.asarray(payload["losses"], dtype=np.float64)
    if (
        payload.get("evaluation_kind") != "final_holdout"
        or payload.get("evaluation_start_batch") != FINAL_START
        or values.shape != (FINAL_BLOCKS,)
        or not np.isfinite(values).all()
    ):
        raise ValueError(f"Invalid final evaluation details: {path}")
    return values


def _interval(
    values: np.ndarray,
    seed: int,
    *,
    block_size: int | None = None,
) -> list[float]:
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


def _health(run_dir: Path) -> dict:
    marker = json.loads((run_dir / "COMPLETED").read_text())
    if int(marker.get("completed_steps", -1)) != STEP:
        raise ValueError(f"Incomplete run: {run_dir}")
    summary = json.loads((run_dir / "training_summary.json").read_text())
    provenance = json.loads((run_dir / "run_provenance.json").read_text())
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
        "total_parameters": int(provenance["parameter_counts"]["total"]),
        "position_parameters": int(provenance["parameter_counts"]["position_params"]),
        "target_tokens_per_second": float(summary["target_tokens_per_second"]),
        "elapsed_seconds": float(summary["elapsed_seconds"]),
        "peak_allocated_mib": float(summary["peak_allocated_mib"]),
        "peak_reserved_mib": float(summary["peak_reserved_mib"]),
        "metrics_finite": all(math.isfinite(value) for value in numeric),
    }


def _contrast(values: np.ndarray, reference: np.ndarray, seed: int) -> dict:
    delta = values - reference
    return {
        "mean_delta": float(delta.mean()),
        "iid_bootstrap_ci95": _interval(delta, seed),
        "contiguous_block_32_bootstrap_ci95": _interval(
            delta,
            seed + 1,
            block_size=32,
        ),
    }


def _cleanup_summary() -> dict:
    path = RESULT_ROOT / "checkpoint_cleanup_events.jsonl"
    events = [json.loads(line) for line in path.read_text().splitlines() if line] if path.is_file() else []
    return {
        "event_count": len(events),
        "reclaimed_bytes": sum(int(event["reclaimed_bytes"]) for event in events),
        "runs_with_removed_checkpoints": sorted(
            {
                event["run_name"]
                for event in events
                if event.get("removed")
            }
        ),
    }


def analyze() -> dict:
    configs = _configs()
    run_dirs = {arm: Path(configs[arm]["output_dir"]) for arm in ARMS}
    losses = {arm: _losses(run_dirs[arm]) for arm in ARMS}
    health = {arm: _health(run_dirs[arm]) for arm in ARMS}
    rows = {}
    for index, arm in enumerate(ARMS):
        rows[arm] = {
            "label": LABELS[arm],
            "mean_nll": float(losses[arm].mean()),
            "block_nll_standard_error": float(
                losses[arm].std(ddof=1) / math.sqrt(losses[arm].size)
            ),
            "versus_rope": _contrast(losses[arm], losses["rope"], 57_000 + 20 * index),
            "versus_scalar_qkpre": _contrast(
                losses[arm],
                losses["scalar-qkpre"],
                57_010 + 20 * index,
            ),
            **health[arm],
        }
    ranking = sorted(ARMS, key=lambda arm: rows[arm]["mean_nll"])
    scalar_delta = rows["scalar-qkpre"]["versus_rope"]
    payload = {
        "scope": "phase57_recognized_positional_baselines",
        "protocol": "paper/POSITIONAL_BASELINE_PROTOCOL.md",
        "training_seed": 123,
        "training_steps": STEP,
        "sequence_batch": 32,
        "context": 1_024,
        "learning_rate": 1.2e-3,
        "final_holdout": {
            "start_batch": FINAL_START,
            "blocks": FINAL_BLOCKS,
            "previously_inspected": False,
        },
        "methods": rows,
        "ranking_best_to_worst": ranking,
        "primary_confirmation": {
            "scalar_minus_rope": scalar_delta,
            "negative_mean": scalar_delta["mean_delta"] < 0,
            "block_32_interval_excludes_zero": scalar_delta[
                "contiguous_block_32_bootstrap_ci95"
            ][1]
            < 0,
        },
        "all_training_metrics_finite": all(
            item["metrics_finite"] for item in health.values()
        ),
        "checkpoint_cleanup": _cleanup_summary(),
        "inference_limit": (
            "All arms use one training seed. Paired block intervals quantify final-holdout "
            "precision, not training-seed variability. The fixed input, partial-RoPE, and "
            "ALiBi cells are controlled implementations rather than claims of exact external "
            "recipe reproduction; ALiBi also uses a different attention kernel."
        ),
    }
    return payload


def _ci_text(contrast: dict) -> str:
    low, high = contrast["contiguous_block_32_bootstrap_ci95"]
    return f"{contrast['mean_delta']:+.6f} [{low:+.6f}, {high:+.6f}]"


def render(payload: dict) -> str:
    lines = [
        "# Phase 57: recognized positional baselines",
        "",
        "| Method | Final NLL | Delta vs RoPE (block-32 95% CI) | Delta vs scalar (block-32 95% CI) | Position params | ktok/s | Peak alloc GiB |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in payload["ranking_best_to_worst"]:
        row = payload["methods"][arm]
        lines.append(
            f"| {row['label']} | {row['mean_nll']:.6f} | "
            f"`{_ci_text(row['versus_rope'])}` | "
            f"`{_ci_text(row['versus_scalar_qkpre'])}` | "
            f"{row['position_parameters']:,} | "
            f"{row['target_tokens_per_second'] / 1000:.1f} | "
            f"{row['peak_allocated_mib'] / 1024:.2f} |"
        )
    primary = payload["primary_confirmation"]["scalar_minus_rope"]
    iid = primary["iid_bootstrap_ci95"]
    block = primary["contiguous_block_32_bootstrap_ci95"]
    cleanup = payload["checkpoint_cleanup"]
    lines.extend(
        [
            "",
            "## Primary confirmation",
            "",
            f"- Scalar pre-Q/K minus RoPE: `{primary['mean_delta']:+.6f}` NLL.",
            f"- IID 95% interval: `[{iid[0]:+.6f}, {iid[1]:+.6f}]`.",
            f"- Contiguous-block-32 95% interval: `[{block[0]:+.6f}, {block[1]:+.6f}]`.",
            f"- All training metrics finite: **{payload['all_training_metrics_finite']}**.",
            "",
            "## Artifact closeout",
            "",
            f"Recovery cleanup reclaimed `{cleanup['reclaimed_bytes'] / 2**30:.2f}` GiB "
            f"across `{len(cleanup['runs_with_removed_checkpoints'])}` completed runs. "
            "No final weights were saved.",
            "",
            payload["inference_limit"],
            "",
        ]
    )
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

