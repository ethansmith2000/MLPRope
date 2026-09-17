#!/usr/bin/env python
"""Analyze the Phase-52 bias simplification and scalar-anchor close-out."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "model-output" / "position_bias_phase52_bias_anchor_closeout"
RESULT_ROOT = ROOT / "results" / "phase52_bias_anchor_closeout"
STEP = 100_000
HOLDOUT_START = 5_120
BLOCKS = 1_024
MATERIALITY = 0.003
NEW_RUNS = {
    "scalar-qk-bias": RUN_ROOT / "phase52-scalar-qk-bias-seed123-b32-s100000-h768d8",
    "qk-readout-r32-no-anchor": RUN_ROOT / "phase52-qk-readout-r32-no-anchor-seed123-b32-s100000-h768d8",
}
REFERENCES = ("rope", "scalar-qkpre", "qk-readout-r32")
CONTRASTS = {
    "scalar_qk_bias_minus_rope": ("scalar-qk-bias", "rope"),
    "scalar_qk_bias_minus_scalar": ("scalar-qk-bias", "scalar-qkpre"),
    "scalar_qk_bias_minus_rank32": ("scalar-qk-bias", "qk-readout-r32"),
    "rank32_no_anchor_minus_rope": ("qk-readout-r32-no-anchor", "rope"),
    "rank32_no_anchor_minus_scalar": (
        "qk-readout-r32-no-anchor", "scalar-qkpre"
    ),
    "rank32_no_anchor_minus_rank32": (
        "qk-readout-r32-no-anchor", "qk-readout-r32"
    ),
    "rank32_no_anchor_minus_scalar_qk_bias": (
        "qk-readout-r32-no-anchor", "scalar-qk-bias"
    ),
}


def _interval(samples: np.ndarray) -> list[float]:
    return [float(x) for x in np.quantile(samples, (0.025, 0.975))]


def _iid_ci(delta: np.ndarray, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    chunks = []
    for _ in range(10):
        indices = rng.integers(0, delta.size, size=(2_000, delta.size))
        chunks.append(delta[indices].mean(axis=1))
    return _interval(np.concatenate(chunks))


def _block_ci(delta: np.ndarray, seed: int, block_length: int = 32) -> list[float]:
    if delta.size % block_length:
        raise ValueError("Holdout size must divide the block length")
    means = delta.reshape(-1, block_length).mean(axis=1)
    rng = np.random.default_rng(seed)
    chunks = []
    for _ in range(10):
        indices = rng.integers(0, means.size, size=(2_000, means.size))
        chunks.append(means[indices].mean(axis=1))
    return _interval(np.concatenate(chunks))


def _new_losses(arm: str) -> np.ndarray:
    run_dir = NEW_RUNS[arm]
    if not (run_dir / "COMPLETED").is_file():
        raise RuntimeError(f"Incomplete Phase-52 run: {run_dir}")
    path = run_dir / "evaluation_details" / f"step_{STEP:08d}_context_001024.json"
    payload = json.loads(path.read_text())
    losses = np.asarray(payload["losses"], dtype=np.float64)
    if (
        payload.get("evaluation_kind") != "final_holdout"
        or payload.get("evaluation_start_batch") != HOLDOUT_START
        or losses.shape != (BLOCKS,)
        or not np.isfinite(losses).all()
    ):
        raise ValueError(f"Invalid Phase-52 evaluation details: {path}")
    return losses


def _reference_losses(arm: str) -> np.ndarray:
    path = RESULT_ROOT / f"reference_{arm}.json"
    payload = json.loads(path.read_text())
    losses = np.asarray(payload["losses"], dtype=np.float64)
    if (
        payload.get("evaluation_start_batch") != HOLDOUT_START
        or payload.get("evaluation_blocks") != BLOCKS
        or losses.shape != (BLOCKS,)
        or not np.isfinite(losses).all()
    ):
        raise ValueError(f"Invalid Phase-52 reference evaluation: {path}")
    return losses


def _parameter_counts(arm: str) -> dict:
    payload = json.loads((NEW_RUNS[arm] / "run_provenance.json").read_text())
    return payload["parameter_counts"]


def _final_metrics(arm: str) -> dict:
    rows = [
        json.loads(line)
        for line in (NEW_RUNS[arm] / "metrics.jsonl").read_text().splitlines()
        if line
    ]
    return next(row for row in reversed(rows) if row.get("evaluation_kind") == "final_holdout")


def analyze() -> dict:
    losses = {arm: _new_losses(arm) for arm in NEW_RUNS}
    losses.update({arm: _reference_losses(arm) for arm in REFERENCES})
    comparisons = {}
    for index, (name, (candidate, reference)) in enumerate(CONTRASTS.items()):
        delta = losses[candidate] - losses[reference]
        comparisons[name] = {
            "candidate": candidate,
            "reference": reference,
            "mean_delta": float(delta.mean()),
            "iid_bootstrap_ci95": _iid_ci(delta, 52_000 + index),
            "contiguous_block_32_bootstrap_ci95": _block_ci(
                delta, 52_100 + index
            ),
        }
    bias_metrics = _final_metrics("scalar-qk-bias")
    bias_values = {
        key: float(value)
        for key, value in bias_metrics.items()
        if key.startswith("architecture/") and key.endswith(("/q_rms", "/k_rms"))
    }
    all_finite = all(
        math.isfinite(float(value))
        for value in list(losses.values())
        for value in value
    ) and all(math.isfinite(value) for value in bias_values.values())
    bias_vs_rank32 = comparisons["scalar_qk_bias_minus_rank32"]["mean_delta"]
    no_anchor_vs_rank32 = comparisons["rank32_no_anchor_minus_rank32"]["mean_delta"]
    return {
        "scope": "phase52_bias_anchor_closeout",
        "training_seed": 123,
        "paired_initialization_seed": 123,
        "training_steps": STEP,
        "sequence_batch": 32,
        "context": 1_024,
        "holdout": {"start_batch": HOLDOUT_START, "blocks": BLOCKS},
        "artifact_policy": {
            "periodic_checkpoints": False,
            "final_model_weights": False,
            "compact_metrics_and_per_sequence_losses": True,
        },
        "runs": {
            arm: {
                "mean_loss": float(values.mean()),
                "parameter_counts": _parameter_counts(arm) if arm in NEW_RUNS else None,
            }
            for arm, values in losses.items()
        },
        "comparisons": comparisons,
        "qk_projection_bias_rms": bias_values,
        "gates": {
            "descriptive_match_margin_nll": MATERIALITY,
            "scalar_qk_bias_within_margin_of_rank32": bias_vs_rank32 <= MATERIALITY,
            "rank32_no_anchor_within_margin_of_rank32": no_anchor_vs_rank32 <= MATERIALITY,
            "qk_projection_biases_moved_from_zero": bool(
                bias_values and max(bias_values.values()) > 1e-4
            ),
            "all_values_finite": all_finite,
        },
        "inference_note": (
            "All arms use the same 1,024 untouched validation blocks, so paired "
            "example and contiguous-block intervals quantify holdout-sample "
            "precision. This is a seed-123 design close-out, not a new training-seed "
            "replication; Phase 50 supplies the three-seed evidence for the parent "
            "rank-32 method. The 0.003-NLL margin is a descriptive engineering "
            "criterion, not a formal equivalence test."
        ),
    }


def render(payload: dict) -> str:
    lines = [
        "# Phase 52: bias and scalar-anchor close-out",
        "",
        "## Endpoint NLL on untouched blocks 5120–6143",
        "",
        "| Arm | NLL |",
        "|---|---:|",
    ]
    for arm, run in payload["runs"].items():
        lines.append(f"| {arm} | {run['mean_loss']:.6f} |")
    lines.extend(
        [
            "",
            "## Paired contrasts",
            "",
            "Negative deltas favor the candidate.",
            "",
            "| Contrast | Delta | IID 95% interval | Block-32 95% interval |",
            "|---|---:|---:|---:|",
        ]
    )
    for name, result in payload["comparisons"].items():
        iid = result["iid_bootstrap_ci95"]
        block = result["contiguous_block_32_bootstrap_ci95"]
        lines.append(
            f"| {name} | {result['mean_delta']:+.6f} | "
            f"[{iid[0]:+.6f}, {iid[1]:+.6f}] | "
            f"[{block[0]:+.6f}, {block[1]:+.6f}] |"
        )
    lines.extend(
        [
            "",
            "## Frozen interpretation rules",
            "",
            f"- Bias simplification within +0.003 NLL of rank-32: "
            f"**{payload['gates']['scalar_qk_bias_within_margin_of_rank32']}**.",
            f"- No-anchor rank-32 within +0.003 NLL of rank-32: "
            f"**{payload['gates']['rank32_no_anchor_within_margin_of_rank32']}**.",
            f"- Q/K biases moved from zero: "
            f"**{payload['gates']['qk_projection_biases_moved_from_zero']}**.",
            "",
            "## Inference limit",
            "",
            payload["inference_note"],
            "",
            "Neither new run saved periodic checkpoints or final model weights.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    payload = analyze()
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    (RESULT_ROOT / "results.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    (RESULT_ROOT / "REPORT.md").write_text(render(payload))
    print(render(payload))


if __name__ == "__main__":
    main()
