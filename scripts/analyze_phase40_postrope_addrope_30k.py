#!/usr/bin/env python
"""Compare post-RoPE AddRoPE with the completed Phase-39 orderings."""

from __future__ import annotations

import json
import os
import statistics
from pathlib import Path

import numpy as np


ROOT = Path(
    os.environ.get("MLPROPE_RESULT_REPO_ROOT", Path(__file__).resolve().parents[1])
).resolve()
PHASE39_ROOT = ROOT / "model-output" / "position_bias_phase39_carrier_location_30k"
PHASE40_ROOT = ROOT / "model-output" / "position_bias_phase40_postrope_addrope_30k"
RESULT_ROOT = ROOT / "results" / "phase40_postrope_addrope_30k"
RUNS = {
    "rope-fixed": PHASE39_ROOT / "phase39-rope-fixed-seed123-s30000-h768d8",
    "qkpre-rope": PHASE39_ROOT / "phase39-qkpre-rope-seed123-s30000-h768d8",
    "addrope-direct-nope": PHASE39_ROOT
    / "phase39-addrope-direct-nope-seed123-s30000-h768d8",
    "addrope-fixed-prerope": PHASE39_ROOT
    / "phase39-addrope-fixed-rope-seed123-s30000-h768d8",
    "addrope-direct-prerope": PHASE39_ROOT
    / "phase39-addrope-direct-rope-seed123-s30000-h768d8",
    "addrope-fixed-postrope": PHASE40_ROOT
    / "phase40-addrope-fixed-postrope-seed123-s30000-h768d8",
    "addrope-direct-postrope": PHASE40_ROOT
    / "phase40-addrope-direct-postrope-seed123-s30000-h768d8",
}


def final_losses(path: Path) -> list[float]:
    if not (path / "COMPLETED").is_file():
        raise RuntimeError(f"Incomplete run: {path}")
    detail = path / "evaluation_details/step_00030000_context_001024.json"
    payload = json.loads(detail.read_text())
    if payload.get("evaluation_kind") != "final_holdout":
        raise ValueError(f"Unexpected evaluation kind in {detail}")
    if payload.get("evaluation_start_batch") != 2_048:
        raise ValueError(f"Unexpected holdout start in {detail}")
    values = [float(value) for value in payload["losses"]]
    if len(values) != 1_024:
        raise ValueError(f"Expected 1,024 losses in {detail}, got {len(values)}")
    return values


def bootstrap_ci(delta: np.ndarray, *, seed: int = 40) -> list[float]:
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(10):
        indices = rng.integers(0, delta.size, size=(2_000, delta.size))
        samples.append(delta[indices].mean(axis=1))
    low, high = np.quantile(np.concatenate(samples), (0.025, 0.975))
    return [float(low), float(high)]


def paired(candidate: list[float], reference: list[float]) -> dict:
    delta = np.asarray(candidate, dtype=np.float64) - np.asarray(
        reference, dtype=np.float64
    )
    return {
        "candidate_loss": statistics.fmean(candidate),
        "reference_loss": statistics.fmean(reference),
        "delta": float(delta.mean()),
        "ci95": bootstrap_ci(delta),
    }


def run_efficiency(path: Path) -> dict:
    summary = json.loads((path / "training_summary.json").read_text())
    provenance = json.loads((path / "run_provenance.json").read_text())
    counts = provenance.get("parameter_counts", {})
    return {
        "tokens_per_second": float(summary["target_tokens_per_second"]),
        "peak_reserved_mib": float(summary["peak_reserved_mib"]),
        "position_params": int(counts.get("position_params", 0)),
    }


def analyze() -> dict:
    losses = {name: final_losses(path) for name, path in RUNS.items()}
    comparisons = {
        "fixed_postrope_minus_fixed_prerope": paired(
            losses["addrope-fixed-postrope"],
            losses["addrope-fixed-prerope"],
        ),
        "direct_postrope_minus_direct_prerope": paired(
            losses["addrope-direct-postrope"],
            losses["addrope-direct-prerope"],
        ),
        "direct_postrope_minus_direct_nope": paired(
            losses["addrope-direct-postrope"],
            losses["addrope-direct-nope"],
        ),
        "direct_postrope_minus_fixed_postrope": paired(
            losses["addrope-direct-postrope"],
            losses["addrope-fixed-postrope"],
        ),
        "direct_postrope_minus_rope_fixed": paired(
            losses["addrope-direct-postrope"],
            losses["rope-fixed"],
        ),
        "direct_postrope_minus_qkpre_rope": paired(
            losses["addrope-direct-postrope"],
            losses["qkpre-rope"],
        ),
    }
    return {
        "scope": "phase40_postrope_addrope_ordering",
        "training_steps": 30_000,
        "seed": 123,
        "runs": {
            name: {
                "loss": statistics.fmean(values),
                **run_efficiency(RUNS[name]),
            }
            for name, values in losses.items()
        },
        "comparisons": comparisons,
        "caveat": (
            "Phase 40 reuses completed Phase-39 controls with identical data, "
            "initialization, optimizer, and evaluation protocol. It remains a "
            "one-training-seed mechanism screen."
        ),
    }


def render(results: dict) -> str:
    lines = [
        "# Phase 40: post-RoPE AddRoPE ordering",
        "",
        "Negative deltas favor the first named arm.",
        "",
        "| Run | Final loss | tok/s | position params |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name, run in results["runs"].items():
        lines.append(
            f"| {name} | {run['loss']:.6f} | "
            f"{run['tokens_per_second']:.0f} | {run['position_params']} |"
        )
    lines.extend(
        [
            "",
            "## Paired ordering contrasts",
            "",
            "| Contrast | Delta | 95% paired interval |",
            "| --- | ---: | ---: |",
        ]
    )
    for name, result in results["comparisons"].items():
        low, high = result["ci95"]
        lines.append(
            f"| {name} | {result['delta']:+.6f} | "
            f"[{low:+.6f}, {high:+.6f}] |"
        )
    lines.extend(["", results["caveat"], ""])
    return "\n".join(lines)


def main() -> None:
    results = analyze()
    markdown = render(results)
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    (RESULT_ROOT / "phase40_analysis.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n"
    )
    (RESULT_ROOT / "PHASE40_RESULTS.md").write_text(markdown)
    print(markdown)


if __name__ == "__main__":
    main()
