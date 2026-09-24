#!/usr/bin/env python
"""Analyze the frozen Phase-59 modern-backbone transfer pair."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "sweep_configs" / "phase59_modern_backbone"
RESULT_ROOT = ROOT / "results" / "phase59_modern_backbone"
STEP = 100_000
FINAL_START = 9_216
FINAL_BLOCKS = 1_024
ARMS = ("rope", "scalar-qkpre")
LABELS = {
    "rope": "modern + standard RoPE",
    "scalar-qkpre": "modern + scalar pre-Q/K + RoPE",
}


def _configs() -> dict[str, dict]:
    result = {}
    for path in sorted(CONFIG_ROOT.glob("*.json")):
        arm = path.stem.split("-", 1)[1]
        result[arm] = json.loads(path.read_text())
    if tuple(result) != ARMS:
        raise ValueError(f"Phase-59 config matrix is incomplete: {tuple(result)}")
    for arm, config in result.items():
        required = {
            "backbone_variant": "modern",
            "ff_hidden_dim": 2_048,
            "max_train_steps": STEP,
            "learning_rate": 1.2e-3,
            "training_length": 1_024,
            "per_device_train_batch_size": 32,
            "seed": 123,
            "paired_initialization_seed": 123,
            "qk_projection_bias": False,
            "qk_norm_mode": "method_aware_rms",
            "final_validation_start_batch": FINAL_START,
            "num_final_validation_batches": FINAL_BLOCKS,
            "save_final_model": False,
            "use_rope": True,
        }
        for key, expected in required.items():
            if config.get(key) != expected:
                raise ValueError(f"Unexpected {arm} config {key}: {config.get(key)!r}")
    return result


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _numeric_finite(rows: list[dict]) -> bool:
    return all(
        math.isfinite(float(value))
        for row in rows
        for value in row.values()
        if isinstance(value, (int, float))
    )


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
        indices = rng.integers(0, values.size, size=(2_000, values.size))
        samples.append(values[indices].mean(axis=1))
    return [float(value) for value in np.quantile(np.concatenate(samples), (0.025, 0.975))]


def _contrast(values: np.ndarray, reference: np.ndarray, seed: int) -> dict:
    delta = values - reference
    return {
        "mean_delta": float(delta.mean()),
        "iid_bootstrap_ci95": _interval(delta, seed),
        "contiguous_block_32_bootstrap_ci95": _interval(
            delta, seed + 1, block_size=32
        ),
    }


def _gate_values(row: dict) -> list[float]:
    return [
        float(value)
        for key, value in sorted(row.items())
        if key.endswith("/qk_preprojection/gate")
    ]


def _development(rows: list[dict]) -> list[dict]:
    result = []
    for row in rows:
        if row.get("evaluation_kind") != "development":
            continue
        gates = _gate_values(row)
        item = {
            "step": int(row["step"]),
            "mean_nll": float(row["eval_loss/context_1024"]),
        }
        if gates:
            item.update(
                {
                    "effective_gate_min": min(gates),
                    "effective_gate_max": max(gates),
                    "effective_gate_mean": float(np.mean(gates)),
                }
            )
        result.append(item)
    return result


def _optimizer_finite(run_dir: Path) -> bool | None:
    path = run_dir / "intervention_optimization.jsonl"
    if not path.is_file():
        return None
    return _numeric_finite(_jsonl(path))


def _health(run_dir: Path) -> dict:
    marker = json.loads((run_dir / "COMPLETED").read_text())
    if int(marker.get("completed_steps", -1)) != STEP:
        raise ValueError(f"Incomplete run: {run_dir}")
    metrics = _jsonl(run_dir / "metrics.jsonl")
    development = _development(metrics)
    if not development or development[-1]["step"] != STEP:
        raise ValueError(f"Missing step-{STEP} development metrics: {run_dir}")
    summary = json.loads((run_dir / "training_summary.json").read_text())
    provenance = json.loads((run_dir / "run_provenance.json").read_text())
    return {
        "total_parameters": int(provenance["parameter_counts"]["total"]),
        "non_embedding_head_parameters": int(
            provenance["parameter_counts"]["non_embed"]
        ),
        "position_parameters": int(
            provenance["parameter_counts"]["position_params"]
        ),
        "target_tokens_per_second": float(summary["target_tokens_per_second"]),
        "elapsed_seconds": float(summary["elapsed_seconds"]),
        "peak_allocated_mib": float(summary["peak_allocated_mib"]),
        "peak_reserved_mib": float(summary["peak_reserved_mib"]),
        "metrics_finite": _numeric_finite(metrics),
        "optimizer_metrics_finite": _optimizer_finite(run_dir),
        "development": development,
    }


def _cleanup_summary() -> dict:
    path = RESULT_ROOT / "checkpoint_cleanup_events.jsonl"
    events = _jsonl(path) if path.is_file() else []
    return {
        "event_count": len(events),
        "reclaimed_bytes": sum(int(event["reclaimed_bytes"]) for event in events),
        "runs_with_removed_checkpoints": sorted(
            {event["run_name"] for event in events if event.get("removed")}
        ),
    }


def analyze() -> dict:
    configs = _configs()
    run_dirs = {arm: Path(configs[arm]["output_dir"]) for arm in ARMS}
    losses = {arm: _losses(run_dirs[arm]) for arm in ARMS}
    health = {arm: _health(run_dirs[arm]) for arm in ARMS}
    contrast = _contrast(losses["scalar-qkpre"], losses["rope"], 59_000)
    rope_curve = {
        point["step"]: point["mean_nll"]
        for point in health["rope"]["development"]
    }
    scalar_curve = [
        {
            **point,
            "delta_vs_rope": point["mean_nll"] - rope_curve[point["step"]],
        }
        for point in health["scalar-qkpre"]["development"]
    ]
    health["scalar-qkpre"]["development"] = scalar_curve
    all_finite = all(
        item["metrics_finite"]
        and item["optimizer_metrics_finite"] is not False
        for item in health.values()
    )
    transfer_pass = bool(
        contrast["mean_delta"] <= -0.010
        and contrast["contiguous_block_32_bootstrap_ci95"][1] < 0
        and all_finite
    )
    return {
        "scope": "phase59_modern_backbone_transfer",
        "protocol": "paper/MODERN_BACKBONE_PROTOCOL.md",
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
        "methods": {
            arm: {
                "label": LABELS[arm],
                "mean_nll": float(losses[arm].mean()),
                "block_nll_standard_error": float(
                    losses[arm].std(ddof=1) / math.sqrt(losses[arm].size)
                ),
                **health[arm],
            }
            for arm in ARMS
        },
        "scalar_minus_rope": contrast,
        "transfer_gate": {
            "materiality_threshold": -0.010,
            "mean_pass": contrast["mean_delta"] <= -0.010,
            "block_32_interval_below_zero": contrast[
                "contiguous_block_32_bootstrap_ci95"
            ][1]
            < 0,
            "all_metrics_finite": all_finite,
            "pass": transfer_pass,
            "next": (
                "eligible for a separately frozen modern rank-32 arm"
                if transfer_pass
                else "do not auto-launch rank 32 or architecture decomposition"
            ),
        },
        "checkpoint_cleanup": _cleanup_summary(),
        "inference_limit": (
            "This is a one-training-seed bundled architecture transfer. Paired "
            "block intervals quantify final-stream precision, not seed variability, "
            "and do not attribute the result to an individual backbone component."
        ),
    }


def render(payload: dict) -> str:
    contrast = payload["scalar_minus_rope"]
    iid = contrast["iid_bootstrap_ci95"]
    block = contrast["contiguous_block_32_bootstrap_ci95"]
    cleanup = payload["checkpoint_cleanup"]
    lines = [
        "# Phase 59: modern-backbone transfer",
        "",
        "| Arm | Final NLL | Position params | Total params | Non-embedding/head params | ktok/s | Peak alloc GiB |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        row = payload["methods"][arm]
        lines.append(
            f"| {row['label']} | {row['mean_nll']:.6f} | "
            f"{row['position_parameters']:,} | {row['total_parameters']:,} | "
            f"{row['non_embedding_head_parameters']:,} | "
            f"{row['target_tokens_per_second'] / 1000:.1f} | "
            f"{row['peak_allocated_mib'] / 1024:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Registered transfer gate",
            "",
            f"- Scalar minus RoPE: `{contrast['mean_delta']:+.6f}` NLL.",
            f"- IID 95% interval: `[{iid[0]:+.6f}, {iid[1]:+.6f}]`.",
            f"- Contiguous-block-32 95% interval: `[{block[0]:+.6f}, {block[1]:+.6f}]`.",
            f"- Transfer gate passed: **{payload['transfer_gate']['pass']}**.",
            f"- Continuation: {payload['transfer_gate']['next']}.",
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
