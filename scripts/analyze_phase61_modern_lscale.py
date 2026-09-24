#!/usr/bin/env python
"""Analyze the frozen Phase-61 modern larger-scale transfer pair."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "sweep_configs" / "phase61_modern_lscale"
RESULT_ROOT = ROOT / "results" / "phase61_modern_lscale"
STEP = 100_000
FINAL_START = 11_264
FINAL_BLOCKS = 1_024
ARMS = ("rope", "scalar-qkpre")
LABELS = {
    "rope": "modern L + standard RoPE",
    "scalar-qkpre": "modern L + scalar pre-Q/K + RoPE",
}


def _configs() -> dict[str, dict]:
    result = {}
    for path in sorted(CONFIG_ROOT.glob("*.json")):
        arm = path.stem.split("-", 1)[1]
        result[arm] = json.loads(path.read_text())
    if tuple(result) != ARMS:
        raise ValueError(f"Phase-61 config matrix is incomplete: {tuple(result)}")
    required = {
        "backbone_variant": "modern",
        "hidden_size": 1_024,
        "depth": 12,
        "n_head": 8,
        "ff_hidden_dim": 2_752,
        "max_train_steps": STEP,
        "learning_rate": 1.2e-3,
        "training_length": 1_024,
        "per_device_train_batch_size": 32,
        "gradient_accumulation_steps": 1,
        "seed": 123,
        "paired_initialization_seed": 123,
        "qk_projection_bias": False,
        "qk_norm_mode": "method_aware_rms",
        "final_validation_start_batch": FINAL_START,
        "num_final_validation_batches": FINAL_BLOCKS,
        "save_final_model": False,
        "use_rope": True,
    }
    for arm, config in result.items():
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


def _interval(values: np.ndarray, seed: int, block_size: int | None = None) -> list[float]:
    if block_size is not None:
        if values.size % block_size:
            raise ValueError("Block size must divide evaluation examples")
        values = values.reshape(-1, block_size).mean(axis=1)
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(10):
        indices = rng.integers(0, values.size, size=(2_000, values.size))
        samples.append(values[indices].mean(axis=1))
    return [float(x) for x in np.quantile(np.concatenate(samples), (0.025, 0.975))]


def _development(rows: list[dict]) -> list[dict]:
    output = []
    for row in rows:
        if row.get("evaluation_kind") != "development":
            continue
        gates = [
            float(value)
            for key, value in sorted(row.items())
            if key.endswith("/qk_preprojection/gate")
        ]
        item = {"step": int(row["step"]), "mean_nll": float(row["eval_loss/context_1024"])}
        if gates:
            item.update(
                effective_gate_min=min(gates),
                effective_gate_max=max(gates),
                effective_gate_mean=float(np.mean(gates)),
            )
        output.append(item)
    return output


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
    optimizer_path = run_dir / "intervention_optimization.jsonl"
    return {
        "total_parameters": int(provenance["parameter_counts"]["total"]),
        "position_parameters": int(provenance["parameter_counts"]["position_params"]),
        "target_tokens_per_second": float(summary["target_tokens_per_second"]),
        "elapsed_seconds": float(summary["elapsed_seconds"]),
        "peak_allocated_mib": float(summary["peak_allocated_mib"]),
        "peak_reserved_mib": float(summary["peak_reserved_mib"]),
        "metrics_finite": _numeric_finite(metrics),
        "optimizer_metrics_finite": (
            _numeric_finite(_jsonl(optimizer_path)) if optimizer_path.is_file() else None
        ),
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
    losses = {arm: _losses(Path(configs[arm]["output_dir"])) for arm in ARMS}
    health = {arm: _health(Path(configs[arm]["output_dir"])) for arm in ARMS}
    delta = losses["scalar-qkpre"] - losses["rope"]
    contrast = {
        "mean_delta": float(delta.mean()),
        "iid_bootstrap_ci95": _interval(delta, 61_000),
        "contiguous_block_32_bootstrap_ci95": _interval(delta, 61_001, 32),
    }
    rope_curve = {x["step"]: x["mean_nll"] for x in health["rope"]["development"]}
    health["scalar-qkpre"]["development"] = [
        {**x, "delta_vs_rope": x["mean_nll"] - rope_curve[x["step"]]}
        for x in health["scalar-qkpre"]["development"]
    ]
    late = [
        x["delta_vs_rope"]
        for x in health["scalar-qkpre"]["development"]
        if x["step"] >= 80_000
    ]
    all_finite = all(
        x["metrics_finite"] and x["optimizer_metrics_finite"] is not False
        for x in health.values()
    )
    positive = bool(
        contrast["contiguous_block_32_bootstrap_ci95"][1] < 0
        and len(late) == 5
        and all(x < 0 for x in late)
        and all_finite
    )
    material = bool(positive and contrast["mean_delta"] <= -0.010)
    return {
        "scope": "phase61_modern_lscale_transfer",
        "protocol": "paper/MODERN_LSCALE_PROTOCOL.md",
        "training_seed": 123,
        "training_steps": STEP,
        "sequence_batch": 32,
        "context": 1_024,
        "learning_rate": 1.2e-3,
        "final_holdout": {"start_batch": FINAL_START, "blocks": FINAL_BLOCKS},
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
        "late_curve": {"steps": [80_000, 85_000, 90_000, 95_000, 100_000], "deltas": late},
        "registered_decisions": {
            "all_metrics_finite": all_finite,
            "positive_scale_transfer": positive,
            "material_scale_transfer": material,
            "materiality_threshold": -0.010,
            "next": (
                "scale transfer supports the primary method; stop local tuning"
                if positive
                else "stop scale expansion and diagnose before any new scale run"
            ),
        },
        "checkpoint_cleanup": _cleanup_summary(),
        "inference_limit": (
            "One training seed; paired block intervals measure final-stream precision, "
            "not seed variability or attribution within the modern bundle."
        ),
    }


def render(payload: dict) -> str:
    contrast = payload["scalar_minus_rope"]
    iid = contrast["iid_bootstrap_ci95"]
    block = contrast["contiguous_block_32_bootstrap_ci95"]
    decision = payload["registered_decisions"]
    cleanup = payload["checkpoint_cleanup"]
    lines = [
        "# Phase 61: modern larger-scale transfer",
        "",
        "| Arm | Final NLL | Position params | Total params | ktok/s | Peak alloc GiB | Final gate mean |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        row = payload["methods"][arm]
        gate = row["development"][-1].get("effective_gate_mean")
        gate_text = f"{gate:.6f}" if gate is not None else "--"
        lines.append(
            f"| {row['label']} | {row['mean_nll']:.6f} | {row['position_parameters']:,} | "
            f"{row['total_parameters']:,} | {row['target_tokens_per_second'] / 1000:.1f} | "
            f"{row['peak_allocated_mib'] / 1024:.2f} | {gate_text} |"
        )
    lines.extend(
        [
            "", "## Registered decisions", "",
            f"- Scalar minus RoPE: `{contrast['mean_delta']:+.6f}` NLL.",
            f"- IID 95% interval: `[{iid[0]:+.6f}, {iid[1]:+.6f}]`.",
            f"- Block-32 95% interval: `[{block[0]:+.6f}, {block[1]:+.6f}]`.",
            f"- Positive scale transfer: **{decision['positive_scale_transfer']}**.",
            f"- Material (`<= -0.010`) scale transfer: **{decision['material_scale_transfer']}**.",
            "", "## Development curve", "",
            "| Step | Scalar minus RoPE NLL | Scalar gate mean |",
            "|---:|---:|---:|",
        ]
    )
    for point in payload["methods"]["scalar-qkpre"]["development"]:
        lines.append(
            f"| {point['step']:,} | {point['delta_vs_rope']:+.6f} | "
            f"{point['effective_gate_mean']:.6f} |"
        )
    lines.extend(
        [
            "", "## Artifact closeout", "",
            f"Recovery cleanup reclaimed `{cleanup['reclaimed_bytes'] / 2**30:.2f}` GiB "
            f"across `{len(cleanup['runs_with_removed_checkpoints'])}` completed runs. "
            "No final weights were saved.",
            "", payload["inference_limit"], "",
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
