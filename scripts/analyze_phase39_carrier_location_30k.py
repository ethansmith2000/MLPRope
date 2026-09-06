#!/usr/bin/env python
"""Analyze the paired Phase-39A carrier-location comparison."""

from __future__ import annotations

import json
import os
import statistics
from pathlib import Path

import numpy as np


ROOT = Path(
    os.environ.get("MLPROPE_RESULT_REPO_ROOT", Path(__file__).resolve().parents[1])
).resolve()
OUTPUT_ROOT = ROOT / "model-output" / "position_bias_phase39_carrier_location_30k"
RESULT_ROOT = ROOT / "results" / "phase39_carrier_location_30k"
ARMS = (
    "rope-fixed",
    "qkpre-rope",
    "input-rope",
    "addrope-fixed-rope",
    "addrope-direct-nope",
    "addrope-direct-rope",
)


def run_dir(arm: str) -> Path:
    return OUTPUT_ROOT / f"phase39-{arm}-seed123-s30000-h768d8"


def final_record(path: Path) -> dict:
    record = None
    with (path / "metrics.jsonl").open() as handle:
        for line in handle:
            candidate = json.loads(line)
            if candidate.get("evaluation_kind") == "final_holdout":
                record = candidate
    if record is None:
        raise RuntimeError(f"Final holdout record is missing from {path}")
    return record


def final_losses(path: Path) -> list[float]:
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


def bootstrap_ci(delta: np.ndarray, *, seed: int = 39) -> list[float]:
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(10):
        indices = rng.integers(0, delta.size, size=(2_000, delta.size))
        samples.append(delta[indices].mean(axis=1))
    low, high = np.quantile(np.concatenate(samples), (0.025, 0.975))
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


def compact_position_diagnostics(record: dict) -> dict:
    suffixes = (
        "/gate",
        "/amplitude_q_min",
        "/amplitude_q_max",
        "/amplitude_k_min",
        "/amplitude_k_max",
        "/phase_q_abs_max",
        "/phase_k_abs_max",
        "/input_mixture_position_to_content_rms_ratio",
        "/projected_q_mixture_position_to_content_rms_ratio",
        "/projected_k_mixture_position_to_content_rms_ratio",
        "/addend_q_to_q_rms_ratio",
        "/addend_k_to_k_rms_ratio",
        "/direct_amplitude_q/min",
        "/direct_amplitude_q/max",
        "/direct_amplitude_q/nonpositive_fraction",
        "/direct_amplitude_k/min",
        "/direct_amplitude_k/max",
        "/direct_amplitude_k/nonpositive_fraction",
        "/direct_phase_q/rms",
        "/direct_phase_q/abs_max",
        "/direct_phase_k/rms",
        "/direct_phase_k/abs_max",
    )
    return {
        key: float(value)
        for key, value in record.items()
        if key.startswith("position/")
        and key.endswith(suffixes)
        and isinstance(value, (int, float))
    }


def run_summary(path: Path, losses: list[float]) -> dict:
    training = json.loads((path / "training_summary.json").read_text())
    provenance = json.loads((path / "run_provenance.json").read_text())
    launches = provenance.get("launches", [])
    counts = provenance.get("parameter_counts", {})
    if not counts and launches:
        counts = launches[-1].get("parameter_counts", {})
    record = final_record(path)
    return {
        "final_holdout_loss": statistics.fmean(losses),
        "development_curve": development_curve(path),
        "target_tokens_per_second": training["target_tokens_per_second"],
        "elapsed_seconds": training["elapsed_seconds"],
        "peak_reserved_mib": training["peak_reserved_mib"],
        "parameter_counts": counts,
        "position_diagnostics": compact_position_diagnostics(record),
    }


def analyze() -> dict:
    losses = {}
    runs = {}
    for arm in ARMS:
        path = run_dir(arm)
        if not (path / "COMPLETED").is_file():
            raise RuntimeError(f"Phase 39 incomplete: {path}")
        losses[arm] = final_losses(path)
        runs[arm] = run_summary(path, losses[arm])

    contrasts = {
        # Overall ranking against the common fixed-RoPE reference.
        f"{arm}_minus_rope-fixed": paired_summary(
            losses[arm], losses["rope-fixed"]
        )
        for arm in ARMS
        if arm != "rope-fixed"
    }
    contrasts.update(
        {
            # Does repeated attention-local access beat a one-shot residual cue?
            "qkpre-rope_minus_input-rope": paired_summary(
                losses["qkpre-rope"], losses["input-rope"]
            ),
            # Pre-projection versus native head-space placement at fixed carriers.
            "qkpre-rope_minus_addrope-fixed-rope": paired_summary(
                losses["qkpre-rope"], losses["addrope-fixed-rope"]
            ),
            # Best pre-Q/K method versus the strongest direct AddRoPE ordering.
            "qkpre-rope_minus_addrope-direct-nope": paired_summary(
                losses["qkpre-rope"], losses["addrope-direct-nope"]
            ),
            # Does learning Q/K amplitude and phase improve the native carrier?
            "addrope-direct-rope_minus_addrope-fixed-rope": paired_summary(
                losses["addrope-direct-rope"], losses["addrope-fixed-rope"]
            ),
            # Does standard RoPE complement direct AddRoPE?
            "addrope-direct-rope_minus_addrope-direct-nope": paired_summary(
                losses["addrope-direct-rope"], losses["addrope-direct-nope"]
            ),
        }
    )
    return {
        "scope": "phase39_carrier_location_and_direct_addrope",
        "training_steps": 30_000,
        "context": 1_024,
        "seed": 123,
        "final_holdout_start_batch": 2_048,
        "final_holdout_examples": 1_024,
        "runs": runs,
        "contrasts": contrasts,
        "caveat": (
            "Paired-example intervals quantify holdout precision within one "
            "training seed. Phase 39A is a breadth screen, not seed-robust "
            "publication evidence."
        ),
    }


def render(results: dict) -> str:
    lines = [
        "# Phase 39A: carrier location and direct AddRoPE",
        "",
        "All runs use h768/d8, seed 123, 30k steps, method-aware Q/K RMSNorm, "
        "and the same disjoint 1,024-example holdout.",
        "",
        "| Arm | Final loss | vs fixed RoPE | 95% paired interval | tok/s | position params |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm in ARMS:
        run = results["runs"][arm]
        if arm == "rope-fixed":
            delta = "reference"
            interval = "—"
        else:
            contrast = results["contrasts"][f"{arm}_minus_rope-fixed"]
            delta = f"{contrast['delta_candidate_minus_reference']:+.6f}"
            low, high = contrast["paired_example_bootstrap_ci95"]
            interval = f"[{low:+.6f}, {high:+.6f}]"
        counts = run["parameter_counts"]
        lines.append(
            f"| {arm} | {run['final_holdout_loss']:.6f} | {delta} | "
            f"{interval} | {run['target_tokens_per_second']:.0f} | "
            f"{counts.get('position_params', 0)} |"
        )
    lines.extend(
        [
            "",
            "## Mechanistic contrasts",
            "",
            "Negative values favor the first named arm.",
            "",
            "| Contrast | Delta | 95% paired interval |",
            "| --- | ---: | ---: |",
        ]
    )
    special = (
        "qkpre-rope_minus_input-rope",
        "qkpre-rope_minus_addrope-fixed-rope",
        "qkpre-rope_minus_addrope-direct-nope",
        "addrope-direct-rope_minus_addrope-fixed-rope",
        "addrope-direct-rope_minus_addrope-direct-nope",
    )
    for name in special:
        contrast = results["contrasts"][name]
        low, high = contrast["paired_example_bootstrap_ci95"]
        lines.append(
            f"| {name} | {contrast['delta_candidate_minus_reference']:+.6f} | "
            f"[{low:+.6f}, {high:+.6f}] |"
        )
    lines.extend(["", results["caveat"], ""])
    return "\n".join(lines)


def main() -> None:
    results = analyze()
    markdown = render(results)
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    (RESULT_ROOT / "phase39_analysis.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n"
    )
    (RESULT_ROOT / "PHASE39_RESULTS.md").write_text(markdown)
    print(markdown)


if __name__ == "__main__":
    main()
