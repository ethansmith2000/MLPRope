#!/usr/bin/env python
"""Analyze the frozen Phase-58 scalar-gate initialization scout."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "sweep_configs" / "phase58_gate_initialization"
RESULT_ROOT = ROOT / "results" / "phase58_gate_initialization"
STEP = 20_000
FINAL_START = 8_192
FINAL_BLOCKS = 1_024
ARMS = (
    "alpha1-direct",
    "alpha01-direct",
    "alpha01-scaled",
    "alpha01-fixed",
)
LABELS = {
    "alpha1-direct": "learned alpha, init 1.0",
    "alpha01-direct": "learned alpha, init 0.1",
    "alpha01-scaled": "alpha=0.1g, g init 1.0",
    "alpha01-fixed": "fixed alpha=0.1",
}


def _configs() -> dict[str, dict]:
    result = {}
    for path in sorted(CONFIG_ROOT.glob("*.json")):
        arm = path.stem.split("-", 1)[1]
        result[arm] = json.loads(path.read_text())
    if tuple(result) != ARMS:
        raise ValueError(f"Phase-58 config matrix is incomplete: {tuple(result)}")
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
            "checkpointing_steps": None,
            "save_final_model": False,
            "use_rope": True,
        }
        for key, expected in required.items():
            if config.get(key) != expected:
                raise ValueError(f"Unexpected {arm} config {key}: {config.get(key)!r}")
    return result


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


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
            delta,
            seed + 1,
            block_size=32,
        ),
    }


def _numeric_metrics_finite(rows: list[dict]) -> bool:
    return all(
        math.isfinite(float(value))
        for row in rows
        for value in row.values()
        if isinstance(value, (int, float))
    )


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
        result.append(
            {
                "step": int(row["step"]),
                "mean_nll": float(row["eval_loss/context_1024"]),
                "effective_gate_min": min(gates),
                "effective_gate_max": max(gates),
                "effective_gate_mean": float(np.mean(gates)),
            }
        )
    return result


def _optimizer_summary(run_dir: Path) -> dict | None:
    path = run_dir / "intervention_optimization.jsonl"
    if not path.is_file():
        return None
    rows = _jsonl(path)
    prefix = "optimization/pre_qk_sinusoid_adapter"
    selected = [row for row in rows if 200 <= int(row["step"]) <= 2_000]
    if not selected:
        raise ValueError(f"No post-warmup optimization telemetry in {path}")

    def median(suffix: str) -> float:
        return float(np.median([row[f"{prefix}/{suffix}"] for row in selected]))

    return {
        "window": [200, 2_000],
        "samples": len(selected),
        "median_parameter_update_rms": median("parameter_update/rms"),
        "median_carrier_function_step_rms": median("carrier_function_step/rms"),
        "median_carrier_to_parameter_update_rms_ratio": median(
            "carrier_function_to_parameter_update_rms_ratio"
        ),
        "metrics_finite": _numeric_metrics_finite(rows),
    }


def _health(run_dir: Path, config: dict) -> dict:
    marker = json.loads((run_dir / "COMPLETED").read_text())
    if int(marker.get("completed_steps", -1)) != STEP:
        raise ValueError(f"Incomplete run: {run_dir}")
    metrics = _jsonl(run_dir / "metrics.jsonl")
    development = _development(metrics)
    if not development or development[-1]["step"] != STEP:
        raise ValueError(f"Missing step-{STEP} development metrics: {run_dir}")
    summary = json.loads((run_dir / "training_summary.json").read_text())
    provenance = json.loads((run_dir / "run_provenance.json").read_text())
    effective_gates = _gate_values(
        next(
            row
            for row in reversed(metrics)
            if row.get("evaluation_kind") == "development" and row.get("step") == STEP
        )
    )
    gate_scale = float(config["qk_preprojection"]["gate_output_scale"])
    return {
        "total_parameters": int(provenance["parameter_counts"]["total"]),
        "position_parameters": int(provenance["parameter_counts"]["position_params"]),
        "target_tokens_per_second": float(summary["target_tokens_per_second"]),
        "elapsed_seconds": float(summary["elapsed_seconds"]),
        "peak_allocated_mib": float(summary["peak_allocated_mib"]),
        "peak_reserved_mib": float(summary["peak_reserved_mib"]),
        "metrics_finite": _numeric_metrics_finite(metrics),
        "development": development,
        "final_effective_gate_range": [min(effective_gates), max(effective_gates)],
        "final_raw_gate_range": [
            min(effective_gates) / gate_scale,
            max(effective_gates) / gate_scale,
        ],
        "optimizer_telemetry": _optimizer_summary(run_dir),
    }


def analyze() -> dict:
    configs = _configs()
    run_dirs = {arm: Path(configs[arm]["output_dir"]) for arm in ARMS}
    losses = {arm: _losses(run_dirs[arm]) for arm in ARMS}
    health = {arm: _health(run_dirs[arm], configs[arm]) for arm in ARMS}
    rows = {}
    reference = losses["alpha1-direct"]
    for index, arm in enumerate(ARMS):
        contrast = _contrast(losses[arm], reference, 58_000 + 10 * index)
        rows[arm] = {
            "label": LABELS[arm],
            "mean_nll": float(losses[arm].mean()),
            "block_nll_standard_error": float(
                losses[arm].std(ddof=1) / math.sqrt(losses[arm].size)
            ),
            "versus_alpha1_direct": contrast,
            "passes_promotion_rule": bool(
                arm != "alpha1-direct"
                and contrast["mean_delta"] <= -0.003
                and contrast["contiguous_block_32_bootstrap_ci95"][1] < 0
            ),
            **health[arm],
        }
    ranking = sorted(ARMS, key=lambda arm: rows[arm]["mean_nll"])
    promoted = [arm for arm in ranking if rows[arm]["passes_promotion_rule"]]
    secondary_contrasts = {
        "alpha01_direct_minus_scaled": _contrast(
            losses["alpha01-direct"], losses["alpha01-scaled"], 58_100
        ),
        "alpha01_direct_minus_fixed": _contrast(
            losses["alpha01-direct"], losses["alpha01-fixed"], 58_110
        ),
        "alpha01_scaled_minus_fixed": _contrast(
            losses["alpha01-scaled"], losses["alpha01-fixed"], 58_120
        ),
    }
    parent_curve = {
        point["step"]: point["mean_nll"]
        for point in rows["alpha1-direct"]["development"]
    }
    for arm in ARMS:
        rows[arm]["development_delta_vs_alpha1_direct"] = [
            {
                "step": point["step"],
                "mean_nll_delta": point["mean_nll"] - parent_curve[point["step"]],
            }
            for point in rows[arm]["development"]
        ]
    return {
        "scope": "phase58_scalar_gate_initialization_scout",
        "protocol": "paper/GATE_INITIALIZATION_PROTOCOL.md",
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
        "primary_contrast": rows["alpha01-direct"]["versus_alpha1_direct"],
        "secondary_contrasts": secondary_contrasts,
        "promotion_candidates": promoted[:1],
        "all_training_metrics_finite": all(
            row["metrics_finite"]
            and (
                row["optimizer_telemetry"] is None
                or row["optimizer_telemetry"]["metrics_finite"]
            )
            for row in health.values()
        ),
        "inference_limit": (
            "This is a one-seed, 20k design scout. Paired block intervals quantify "
            "endpoint-stream precision, not training-seed variability or mature-horizon "
            "performance. At most one passing alternative may advance to 100k."
        ),
    }


def _ci_text(contrast: dict) -> str:
    low, high = contrast["contiguous_block_32_bootstrap_ci95"]
    return f"{contrast['mean_delta']:+.6f} [{low:+.6f}, {high:+.6f}]"


def render(payload: dict) -> str:
    lines = [
        "# Phase 58: scalar-carrier initialization scout",
        "",
        "| Arm | Final NLL | Delta vs alpha1 direct (block-32 95% CI) | Final effective gate range | Function-step / parameter-step | Promote? |",
        "|---|---:|---:|---:|---:|:---:|",
    ]
    for arm in payload["ranking_best_to_worst"]:
        row = payload["methods"][arm]
        gates = row["final_effective_gate_range"]
        telemetry = row["optimizer_telemetry"]
        ratio = "n/a" if telemetry is None else f"{telemetry['median_carrier_to_parameter_update_rms_ratio']:.4f}"
        lines.append(
            f"| {row['label']} | {row['mean_nll']:.6f} | "
            f"`{_ci_text(row['versus_alpha1_direct'])}` | "
            f"`[{gates[0]:.4f}, {gates[1]:.4f}]` | {ratio} | "
            f"{'yes' if row['passes_promotion_rule'] else 'no'} |"
        )
    primary = payload["primary_contrast"]
    secondary = payload["secondary_contrasts"]
    iid = primary["iid_bootstrap_ci95"]
    block = primary["contiguous_block_32_bootstrap_ci95"]
    candidates = payload["promotion_candidates"]
    lines.extend(
        [
            "",
            "## Registered readout",
            "",
            f"- Direct alpha=0.1 minus direct alpha=1.0: `{primary['mean_delta']:+.6f}` NLL.",
            f"- IID 95% interval: `[{iid[0]:+.6f}, {iid[1]:+.6f}]`.",
            f"- Contiguous-block-32 95% interval: `[{block[0]:+.6f}, {block[1]:+.6f}]`.",
            f"- Direct alpha=0.1 minus scaled alpha=0.1g: `{_ci_text(secondary['alpha01_direct_minus_scaled'])}`.",
            f"- Scaled alpha=0.1g minus fixed alpha=0.1: `{_ci_text(secondary['alpha01_scaled_minus_fixed'])}`.",
            f"- Promotion candidate: `{candidates[0]}`." if candidates else "- No arm passed the frozen promotion rule.",
            f"- All training and optimizer metrics finite: **{payload['all_training_metrics_finite']}**.",
            "",
            "No checkpoints or final weights were requested for this short scout.",
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
