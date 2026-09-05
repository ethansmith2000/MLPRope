#!/usr/bin/env python
"""Analyze the frozen Phase-38 paired evidence matrix."""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "model-output" / "position_bias_phase38_evidence_200k"
RESULT_ROOT = ROOT / "results" / "phase38_evidence_200k"
PAIR_KEYS = (
    "scale-h1024d12-seed123",
    "rep-h768d8-seed456",
    "rep-h768d8-seed789",
    "noqknorm-h768d8-seed123",
)
PHASE33_ROOT = ROOT / "model-output" / "position_bias_phase33_static_qkpre_200k"


def run_dir(pair: str, arm: str) -> Path:
    return OUTPUT_ROOT / f"phase38-{pair}-{arm}-s200000"


def final_losses(path: Path) -> list[float]:
    detail = path / "evaluation_details/step_00200000_context_001024.json"
    payload = json.loads(detail.read_text())
    if payload.get("evaluation_kind") != "final_holdout":
        raise ValueError(f"Unexpected evaluation kind in {detail}")
    if payload.get("evaluation_start_batch") != 2_048:
        raise ValueError(f"Unexpected holdout start in {detail}")
    values = [float(value) for value in payload["losses"]]
    if len(values) != 1_024:
        raise ValueError(f"Expected 1,024 losses in {detail}, got {len(values)}")
    return values


def bootstrap_ci(delta: np.ndarray, *, seed: int = 38) -> list[float]:
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(10):
        indices = rng.integers(0, delta.size, size=(2_000, delta.size))
        means.append(delta[indices].mean(axis=1))
    samples = np.concatenate(means)
    low, high = np.quantile(samples, (0.025, 0.975))
    return [float(low), float(high)]


def paired_summary(candidate: list[float], reference: list[float]) -> dict:
    delta = np.asarray(candidate, dtype=np.float64) - np.asarray(
        reference, dtype=np.float64
    )
    return {
        "candidate_loss": statistics.fmean(candidate),
        "reference_loss": statistics.fmean(reference),
        "delta_candidate_minus_reference": float(delta.mean()),
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


def final_gate_range(path: Path) -> list[float] | None:
    final = None
    with (path / "metrics.jsonl").open() as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("evaluation_kind") == "final_holdout":
                final = record
    if final is None:
        return None
    gates = [
        float(value)
        for key, value in final.items()
        if key.endswith("/qk_preprojection/gate")
        and isinstance(value, (int, float))
    ]
    return [min(gates), max(gates)] if gates else None


def run_summary(path: Path, losses: list[float]) -> dict:
    training = json.loads((path / "training_summary.json").read_text())
    provenance = json.loads((path / "run_provenance.json").read_text())
    launches = provenance.get("launches", [])
    counts = launches[-1].get("parameter_counts", {}) if launches else {}
    return {
        "final_holdout_loss": statistics.fmean(losses),
        "development_curve": development_curve(path),
        "target_tokens_per_second": training["target_tokens_per_second"],
        "elapsed_seconds": training["elapsed_seconds"],
        "peak_reserved_mib": training["peak_reserved_mib"],
        "parameter_counts": counts,
        "gate_range": final_gate_range(path),
    }


def phase33_delta() -> float:
    baseline = PHASE33_ROOT / "phase33-rope-fixed-seed123-s200000-h768d8"
    candidate = PHASE33_ROOT / "phase33-qkpre-tied-rope-seed123-s200000-h768d8"
    return paired_summary(final_losses(candidate), final_losses(baseline))[
        "delta_candidate_minus_reference"
    ]


def analyze() -> dict:
    results = {}
    for pair in PAIR_KEYS:
        baseline_dir = run_dir(pair, "rope")
        candidate_dir = run_dir(pair, "qkpre")
        incomplete = [
            str(path)
            for path in (baseline_dir, candidate_dir)
            if not (path / "COMPLETED").is_file()
        ]
        if incomplete:
            raise RuntimeError(f"Phase 38 incomplete: {incomplete}")
        baseline_losses = final_losses(baseline_dir)
        candidate_losses = final_losses(candidate_dir)
        results[pair] = {
            "baseline": run_summary(baseline_dir, baseline_losses),
            "candidate": run_summary(candidate_dir, candidate_losses),
            "paired": paired_summary(candidate_losses, baseline_losses),
        }

    mature_by_seed = {
        "123": phase33_delta(),
        "456": results["rep-h768d8-seed456"]["paired"][
            "delta_candidate_minus_reference"
        ],
        "789": results["rep-h768d8-seed789"]["paired"][
            "delta_candidate_minus_reference"
        ],
    }
    mature_mean = statistics.fmean(mature_by_seed.values())
    scale = results["scale-h1024d12-seed123"]["paired"]
    noqk = results["noqknorm-h768d8-seed123"]["paired"]
    return {
        "scope": "phase38_pre_qk_evidence_strengthening",
        "training_steps": 200_000,
        "context": 1_024,
        "final_holdout_start_batch": 2_048,
        "final_holdout_examples": 1_024,
        "pairs": results,
        "mature_h768_by_seed": mature_by_seed,
        "mature_h768_mean_delta": mature_mean,
        "gates": {
            "mature_replication_pass": (
                all(value < 0 for value in mature_by_seed.values())
                and mature_mean <= -0.03
            ),
            "scale_transfer_pass": (
                scale["delta_candidate_minus_reference"] <= -0.03
                and scale["paired_example_bootstrap_ci95"][1] < 0
            ),
            "noqknorm_robustness_pass": (
                noqk["delta_candidate_minus_reference"] <= -0.02
                and noqk["paired_example_bootstrap_ci95"][1] < 0
            ),
        },
        "caveat": (
            "Paired-example intervals estimate holdout precision within a seed. "
            "The three h768 seed deltas, not 3,072 examples pooled as independent, "
            "are the unit for training-seed robustness."
        ),
    }


def render(results: dict) -> str:
    lines = [
        "# Phase 38: pre-Q/K evidence strengthening",
        "",
        "All contrasts are scalar pre-Q/K + fixed RoPE minus matched fixed RoPE.",
        "Negative deltas favor the candidate.",
        "",
        "| Pair | Baseline | Candidate | Delta | Paired bootstrap 95% CI |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for pair in PAIR_KEYS:
        row = results["pairs"][pair]
        paired = row["paired"]
        low, high = paired["paired_example_bootstrap_ci95"]
        lines.append(
            f"| {pair} | {paired['reference_loss']:.6f} | "
            f"{paired['candidate_loss']:.6f} | "
            f"{paired['delta_candidate_minus_reference']:+.6f} | "
            f"[{low:+.6f}, {high:+.6f}] |"
        )
    lines.extend(
        [
            "",
            "## Mature h768 replication",
            "",
            "| Seed | Delta |",
            "| ---: | ---: |",
        ]
    )
    for seed, delta in results["mature_h768_by_seed"].items():
        lines.append(f"| {seed} | {delta:+.6f} |")
    lines.extend(
        [
            f"| mean | {results['mature_h768_mean_delta']:+.6f} |",
            "",
            "## Predeclared gates",
            "",
        ]
    )
    for name, passed in results["gates"].items():
        lines.append(f"- {name}: **{'PASS' if passed else 'FAIL'}**")
    lines.append("")
    lines.append(results["caveat"])
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    results = analyze()
    markdown = render(results)
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    (RESULT_ROOT / "phase38_analysis.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n"
    )
    (RESULT_ROOT / "PHASE38_RESULTS.md").write_text(markdown)
    print(markdown)


if __name__ == "__main__":
    main()
