#!/usr/bin/env python
"""Analyze the frozen Phase-56 LR boundary check."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "sweep_configs" / "phase56_lr_boundary"
NEW_ROOT = ROOT / "model-output" / "position_bias_phase56_lr_boundary"
REFERENCE_ROOT = ROOT / "model-output" / "position_bias_phase55_lr_robustness"
RESULT_ROOT = ROOT / "results" / "phase56_lr_boundary"
STEP = 100_000
SELECTION_START = 6_144
SELECTION_BLOCKS = 1_024
DEVELOPMENT_START = 0
DEVELOPMENT_BLOCKS = 128
CURRENT_LR = 6.0e-4
BOUNDARY_LR = 1.2e-3
ARMS = ("rope", "scalar-qkpre")
REFERENCE_RUNS = {
    "rope": REFERENCE_ROOT / "phase55-rope-lr6e4-seed123-b32-s100000-h768d8",
    "scalar-qkpre": REFERENCE_ROOT / "phase55-scalar-qkpre-lr6e4-seed123-b32-s100000-h768d8",
}


def _configs() -> dict[str, dict]:
    result = {}
    for path in sorted(CONFIG_ROOT.glob("*.json")):
        config = json.loads(path.read_text())
        arm = "scalar-qkpre" if config["qk_preprojection"]["enabled"] else "rope"
        result[arm] = config
    if set(result) != set(ARMS):
        raise ValueError("Phase-56 config matrix is incomplete")
    return result


def _losses(run_dir: Path, *, development: bool) -> np.ndarray:
    suffix = "_development_context_001024.json" if development else "_context_001024.json"
    path = run_dir / "evaluation_details" / f"step_{STEP:08d}{suffix}"
    payload = json.loads(path.read_text())
    values = np.asarray(payload["losses"], dtype=np.float64)
    expected_kind = "development" if development else "final_holdout"
    expected_start = DEVELOPMENT_START if development else SELECTION_START
    expected_blocks = DEVELOPMENT_BLOCKS if development else SELECTION_BLOCKS
    if (
        payload.get("evaluation_kind") != expected_kind
        or payload.get("evaluation_start_batch") != expected_start
        or values.shape != (expected_blocks,)
        or not np.isfinite(values).all()
    ):
        raise ValueError(f"Invalid evaluation details: {path}")
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


def _health(run_dir: Path) -> dict:
    marker = json.loads((run_dir / "COMPLETED").read_text())
    if int(marker.get("completed_steps", -1)) != STEP:
        raise ValueError(f"Incomplete run: {run_dir}")
    summary = json.loads((run_dir / "training_summary.json").read_text())
    rows = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines() if line]
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
    configs = _configs()
    run_dirs = {arm: Path(configs[arm]["output_dir"]) for arm in ARMS}
    health = {arm: _health(run_dirs[arm]) for arm in ARMS}
    development = {
        CURRENT_LR: {arm: _losses(REFERENCE_RUNS[arm], development=True) for arm in ARMS},
        BOUNDARY_LR: {arm: _losses(run_dirs[arm], development=True) for arm in ARMS},
    }
    selection = {
        CURRENT_LR: {arm: _losses(REFERENCE_RUNS[arm], development=False) for arm in ARMS},
        BOUNDARY_LR: {arm: _losses(run_dirs[arm], development=False) for arm in ARMS},
    }
    both_finite = all(item["metrics_finite"] for item in health.values())
    chosen_lr = min(
        (CURRENT_LR, BOUNDARY_LR),
        key=lambda lr: development[lr]["rope"].mean(),
    ) if both_finite else CURRENT_LR
    boundary_delta = selection[BOUNDARY_LR]["scalar-qkpre"] - selection[BOUNDARY_LR]["rope"]
    payload = {
        "scope": "phase56_lr_boundary",
        "training_seed": 123,
        "training_steps": STEP,
        "sequence_batch": 32,
        "context": 1_024,
        "development": {
            f"{lr:.1e}": {arm: float(development[lr][arm].mean()) for arm in ARMS}
            for lr in (CURRENT_LR, BOUNDARY_LR)
        },
        "selection_evidence": {
            "start_batch": SELECTION_START,
            "blocks": SELECTION_BLOCKS,
            "grid": {
                f"{lr:.1e}": {arm: float(selection[lr][arm].mean()) for arm in ARMS}
                for lr in (CURRENT_LR, BOUNDARY_LR)
            },
            "boundary_scalar_minus_rope": {
                "mean_delta": float(boundary_delta.mean()),
                "iid_bootstrap_ci95": _interval(boundary_delta, 56_000),
                "contiguous_block_32_bootstrap_ci95": _interval(
                    boundary_delta, 56_100, block_size=32
                ),
            },
        },
        "run_health": health,
        "decision": {
            "both_boundary_runs_finite": both_finite,
            "selected_common_learning_rate": chosen_lr,
            "selection_basis": "lowest RoPE 100k development NLL among 6e-4 and 1.2e-3, conditional on both new arms being finite",
            "boundary_scalar_advantage_negative": float(boundary_delta.mean()) < 0,
            "stop_lr_expansion": True,
        },
        "future_confirmatory_holdout": {
            "start_batch": 7_168,
            "blocks": 1_024,
            "inspected_in_phase56": False,
        },
        "inference_limit": (
            "The selected LR is the best of a resource-bounded candidate set, not a claimed optimizer optimum. "
            "Adam betas, warmup, weight decay, clipping, and schedule were not tuned."
        ),
    }
    return payload


def render(payload: dict) -> str:
    lines = [
        "# Phase 56: learning-rate boundary check",
        "",
        "| Peak LR | RoPE development NLL | Scalar development NLL | RoPE selection NLL | Scalar selection NLL |",
        "|---:|---:|---:|---:|---:|",
    ]
    for key in ("6.0e-04", "1.2e-03"):
        dev = payload["development"][key]
        selection = payload["selection_evidence"]["grid"][key]
        lines.append(
            f"| `{key}` | {dev['rope']:.6f} | {dev['scalar-qkpre']:.6f} | "
            f"{selection['rope']:.6f} | {selection['scalar-qkpre']:.6f} |"
        )
    contrast = payload["selection_evidence"]["boundary_scalar_minus_rope"]
    iid = contrast["iid_bootstrap_ci95"]
    block = contrast["contiguous_block_32_bootstrap_ci95"]
    decision = payload["decision"]
    lines.extend([
        "",
        "## Frozen decisions",
        "",
        f"- Boundary scalar minus RoPE: `{contrast['mean_delta']:+.6f}`.",
        f"- IID 95% interval: `[{iid[0]:+.6f}, {iid[1]:+.6f}]`.",
        f"- Block-32 95% interval: `[{block[0]:+.6f}, {block[1]:+.6f}]`.",
        f"- Both boundary runs finite: **{decision['both_boundary_runs_finite']}**.",
        f"- Selected common LR: `{decision['selected_common_learning_rate']:.1e}`.",
        "- LR expansion stops here by protocol.",
        "",
        payload["inference_limit"],
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    payload = analyze()
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    (RESULT_ROOT / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    report = render(payload)
    (RESULT_ROOT / "REPORT.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
