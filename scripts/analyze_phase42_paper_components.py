#!/usr/bin/env python
"""Analyze the matched Phase-42 paper component matrix."""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "model-output" / "position_bias_phase42_paper_components"
RESULT_ROOT = ROOT / "results" / "phase42_paper_components"
ARMS = (
    "nope",
    "rope",
    "qkpre-nope",
    "qkpre-rope",
    "qkpre-fixed-rope",
    "qkpre-global-rope",
    "qkpre-first-rope",
    "input-rope",
    "addrope-fixed-nope",
    "addrope-fixed-postrope",
)


def run_dir(arm: str) -> Path:
    return OUTPUT_ROOT / f"phase42-{arm}-seed123-b32-s100000-h768d8"


def final_losses(path: Path) -> np.ndarray:
    detail = path / "evaluation_details/step_00100000_context_001024.json"
    payload = json.loads(detail.read_text())
    if payload.get("evaluation_kind") != "final_holdout":
        raise ValueError(f"Unexpected evaluation kind in {detail}")
    if payload.get("evaluation_start_batch") != 2_048:
        raise ValueError(f"Unexpected holdout start in {detail}")
    values = np.asarray(payload["losses"], dtype=np.float64)
    if values.shape != (1_024,):
        raise ValueError(f"Expected 1,024 losses in {detail}, got {values.shape}")
    return values


def bootstrap_ci(delta: np.ndarray, *, seed: int = 42) -> list[float]:
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(10):
        indices = rng.integers(0, delta.size, size=(2_000, delta.size))
        samples.append(delta[indices].mean(axis=1))
    low, high = np.quantile(np.concatenate(samples), (0.025, 0.975))
    return [float(low), float(high)]


def delta_summary(delta: np.ndarray) -> dict:
    return {
        "delta": float(delta.mean()),
        "paired_example_bootstrap_ci95": bootstrap_ci(delta),
        "num_paired_examples": int(delta.size),
    }


def development_curve(path: Path) -> dict[str, float]:
    curve = {}
    with (path / "metrics.jsonl").open() as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("evaluation_kind") != "development":
                continue
            step = record.get("step")
            loss = record.get("eval_loss/context_1024", record.get("eval_loss"))
            if isinstance(step, int) and isinstance(loss, (int, float)):
                curve[str(step)] = float(loss)
    return curve


def final_record(path: Path) -> dict:
    record = None
    with (path / "metrics.jsonl").open() as handle:
        for line in handle:
            candidate = json.loads(line)
            if candidate.get("evaluation_kind") == "final_holdout":
                record = candidate
    if record is None:
        raise RuntimeError(f"Final holdout record missing from {path}")
    return record


def gate_values(path: Path) -> list[float]:
    record = final_record(path)
    return [
        float(value)
        for key, value in sorted(record.items())
        if key.endswith("/qk_preprojection/gate")
        and isinstance(value, (int, float))
    ]


def run_summary(path: Path, losses: np.ndarray) -> dict:
    training = json.loads((path / "training_summary.json").read_text())
    provenance = json.loads((path / "run_provenance.json").read_text())
    counts = provenance.get("parameter_counts", {})
    if not counts and provenance.get("launches"):
        counts = provenance["launches"][-1].get("parameter_counts", {})
    gates = gate_values(path)
    return {
        "final_holdout_loss": float(losses.mean()),
        "development_curve": development_curve(path),
        "target_tokens_per_second": training["target_tokens_per_second"],
        "elapsed_seconds": training["elapsed_seconds"],
        "peak_reserved_mib": training["peak_reserved_mib"],
        "parameter_counts": counts,
        "gate_values": gates,
        "gate_range": [min(gates), max(gates)] if gates else None,
    }


def analyze() -> dict:
    missing = [
        str(run_dir(arm))
        for arm in ARMS
        if not (run_dir(arm) / "COMPLETED").is_file()
    ]
    if missing:
        raise RuntimeError(f"Phase 42 incomplete: {missing}")
    losses = {arm: final_losses(run_dir(arm)) for arm in ARMS}
    runs = {arm: run_summary(run_dir(arm), losses[arm]) for arm in ARMS}

    contrast_pairs = {
        "qkpre-rope_minus_rope": ("qkpre-rope", "rope"),
        "qkpre-nope_minus_nope": ("qkpre-nope", "nope"),
        "qkpre-rope_minus_qkpre-nope": ("qkpre-rope", "qkpre-nope"),
        "rope_minus_nope": ("rope", "nope"),
        "qkpre-fixed-rope_minus_qkpre-rope": (
            "qkpre-fixed-rope",
            "qkpre-rope",
        ),
        "qkpre-global-rope_minus_qkpre-rope": (
            "qkpre-global-rope",
            "qkpre-rope",
        ),
        "qkpre-first-rope_minus_qkpre-rope": (
            "qkpre-first-rope",
            "qkpre-rope",
        ),
        "input-rope_minus_qkpre-rope": ("input-rope", "qkpre-rope"),
        "addrope-fixed-nope_minus_rope": ("addrope-fixed-nope", "rope"),
        "addrope-fixed-nope_minus_qkpre-rope": (
            "addrope-fixed-nope",
            "qkpre-rope",
        ),
        "addrope-fixed-postrope_minus_rope": (
            "addrope-fixed-postrope",
            "rope",
        ),
        "addrope-fixed-postrope_minus_qkpre-rope": (
            "addrope-fixed-postrope",
            "qkpre-rope",
        ),
    }
    contrasts = {
        name: delta_summary(losses[candidate] - losses[reference])
        for name, (candidate, reference) in contrast_pairs.items()
    }
    interaction = (
        losses["qkpre-rope"]
        - losses["rope"]
        - losses["qkpre-nope"]
        + losses["nope"]
    )
    contrasts["carrier_by_rope_interaction"] = delta_summary(interaction)
    return {
        "scope": "phase42_batch32_paper_components",
        "context": 1_024,
        "sequence_batch": 32,
        "training_steps": 100_000,
        "nominal_training_tokens": 3_276_800_000,
        "seed": 123,
        "final_holdout_start_batch": 2_048,
        "final_holdout_examples": 1_024,
        "runs": runs,
        "contrasts": contrasts,
        "caveat": (
            "Paired-example intervals quantify holdout precision within one "
            "training seed; they do not estimate training-seed uncertainty."
        ),
    }


def render(results: dict) -> str:
    lines = [
        "# Phase 42: batch-32 paper component matrix",
        "",
        "All runs use h768/d8, context 1024, sequence batch 32, seed 123, "
        "100k updates, and the same disjoint 1,024-example holdout.",
        "",
        "| Arm | Final NLL | tokens/s | Position params | Gate range |",
        "|---|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        run = results["runs"][arm]
        gate_range = run["gate_range"]
        gate_text = (
            "--"
            if gate_range is None
            else f"[{gate_range[0]:.4f}, {gate_range[1]:.4f}]"
        )
        lines.append(
            f"| {arm} | {run['final_holdout_loss']:.6f} | "
            f"{run['target_tokens_per_second']:.0f} | "
            f"{run['parameter_counts'].get('position_params', 0)} | "
            f"{gate_text} |"
        )
    lines.extend(
        [
            "",
            "## Paired contrasts",
            "",
            "Negative deltas favor the first named arm. The factorial interaction "
            "is `(C+R - R) - (C - N)`.",
            "",
            "| Contrast | Delta | Paired 95% interval |",
            "|---|---:|---:|",
        ]
    )
    for name, contrast in results["contrasts"].items():
        low, high = contrast["paired_example_bootstrap_ci95"]
        lines.append(
            f"| {name} | {contrast['delta']:+.6f} | "
            f"[{low:+.6f}, {high:+.6f}] |"
        )
    lines.extend(["", results["caveat"], ""])
    return "\n".join(lines)


if __name__ == "__main__":
    results = analyze()
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    (RESULT_ROOT / "summary.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n"
    )
    report = render(results)
    (RESULT_ROOT / "REPORT.md").write_text(report)
    print(report)
