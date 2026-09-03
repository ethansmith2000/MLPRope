#!/usr/bin/env python
"""Analyze the paired seed-123 Phase-32 AddRoPE EMA coupling screen."""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT / "model-output" / "position_bias_phase32_ema_coupling"
RESULT_ROOT = ROOT / "results" / "phase32_dynamic_ema_coupling"
SEED = 123
STEPS = 15_000
ARMS = (
    "addrope-content-pointwise",
    "addrope-content-ema-scalar",
    "addrope-content-ema-per-head",
    "addrope-content-ema-per-dim",
)
CONTRASTS = (
    (
        "addrope-content-ema-scalar",
        "addrope-content-pointwise",
        "value of a single causal timescale",
    ),
    (
        "addrope-content-ema-per-head",
        "addrope-content-pointwise",
        "value of head-specific causal timescales",
    ),
    (
        "addrope-content-ema-per-dim",
        "addrope-content-pointwise",
        "value of latent-specific causal timescales",
    ),
    (
        "addrope-content-ema-per-head",
        "addrope-content-ema-scalar",
        "incremental value of head-specific decay",
    ),
    (
        "addrope-content-ema-per-dim",
        "addrope-content-ema-scalar",
        "incremental value of latent-specific decay",
    ),
    (
        "addrope-content-ema-per-dim",
        "addrope-content-ema-per-head",
        "latent-specific versus head-specific decay",
    ),
)


def run_dir(arm: str) -> Path:
    return OUTPUT_ROOT / f"phase32-{arm}-seed{SEED}-s{STEPS}-h768d8"


def _metrics_rows(arm: str) -> list[dict]:
    return [
        json.loads(line)
        for line in (run_dir(arm) / "metrics.jsonl").read_text().splitlines()
        if line.strip()
    ]


def load_final_losses(arm: str) -> list[float]:
    path = (
        run_dir(arm)
        / "evaluation_details"
        / "step_00015000_context_001024.json"
    )
    payload = json.loads(path.read_text())
    if payload["evaluation_kind"] != "final_holdout":
        raise ValueError(f"Unexpected evaluation kind in {path}")
    if payload["evaluation_start_batch"] != 2_048:
        raise ValueError(f"Unexpected holdout offset in {path}")
    losses = [float(value) for value in payload["losses"]]
    if len(losses) != 256:
        raise ValueError(f"Expected 256 final-holdout losses in {path}")
    return losses


def load_final_metrics(arm: str) -> dict:
    final = None
    for row in _metrics_rows(arm):
        if row.get("evaluation_kind") == "final_holdout":
            final = row
    if final is None:
        raise ValueError(f"No final-holdout metrics for {arm}")
    return final


def load_development_curve(arm: str) -> dict[str, float]:
    curve = {}
    for row in _metrics_rows(arm):
        if row.get("evaluation_kind") == "development":
            curve[str(int(row["step"]))] = float(row["eval_loss"])
    expected = {"5000", "10000", "15000"}
    if set(curve) != expected:
        raise ValueError(f"Unexpected development checkpoints for {arm}: {curve}")
    return curve


def paired_summary(candidate: list[float], reference: list[float]) -> dict:
    differences = [
        candidate_loss - reference_loss
        for candidate_loss, reference_loss in zip(candidate, reference, strict=True)
    ]
    mean = statistics.fmean(differences)
    std = statistics.stdev(differences)
    half_width = 1.96 * std / math.sqrt(len(differences))
    return {
        "candidate_loss": statistics.fmean(candidate),
        "reference_loss": statistics.fmean(reference),
        "delta_candidate_minus_reference": mean,
        "paired_example_ci95": [mean - half_width, mean + half_width],
        "num_paired_examples": len(differences),
    }


def triage(delta: float) -> str:
    if delta <= -0.01:
        return "survive"
    if delta >= 0.01:
        return "prune"
    return "unresolved"


def mechanism_health(metrics: dict) -> dict:
    wanted_suffixes = (
        "hyper_ema_decay_mean",
        "hyper_ema_decay_min",
        "hyper_ema_decay_max",
        "hyper_ema_effective_window_mean",
        "hyper_phase_delta_q/rms",
        "hyper_phase_delta_k/rms",
        "hyper_amplitude_delta_q/rms",
        "hyper_amplitude_delta_k/rms",
    )
    selected = {}
    for suffix in wanted_suffixes:
        values = [
            float(value)
            for key, value in metrics.items()
            if key.startswith("position/layer_") and key.endswith(f"/{suffix}")
        ]
        if values:
            selected[suffix] = {
                "layer_min": min(values),
                "layer_mean": statistics.fmean(values),
                "layer_max": max(values),
            }
    return selected


def analyze() -> dict:
    missing = [
        str(run_dir(arm) / "COMPLETED")
        for arm in ARMS
        if not (run_dir(arm) / "COMPLETED").is_file()
    ]
    if missing:
        raise RuntimeError(
            "Phase-32 screen is incomplete; missing markers:\n" + "\n".join(missing)
        )
    losses = {arm: load_final_losses(arm) for arm in ARMS}
    arms = {}
    for arm in ARMS:
        training = json.loads((run_dir(arm) / "training_summary.json").read_text())
        arms[arm] = {
            "final_loss": statistics.fmean(losses[arm]),
            "development_curve": load_development_curve(arm),
            "elapsed_seconds": training["elapsed_seconds"],
            "target_tokens_per_second": training["target_tokens_per_second"],
            "peak_allocated_mib": training["peak_allocated_mib"],
            "mechanism_health": mechanism_health(load_final_metrics(arm)),
        }
    contrasts = {}
    for candidate, reference, question in CONTRASTS:
        summary = paired_summary(losses[candidate], losses[reference])
        summary.update(
            {
                "candidate": candidate,
                "reference": reference,
                "question": question,
                "negative_favors_candidate": True,
                "triage": triage(summary["delta_candidate_minus_reference"]),
            }
        )
        contrasts[f"{candidate}_vs_{reference}"] = summary
    return {
        "scope": "single_seed_15k_ema_coefficient_axis_screen",
        "seed": SEED,
        "primary_context": 1_024,
        "development_checkpoints": [5_000, 10_000, 15_000],
        "final_holdout_examples": 256,
        "final_holdout_start_batch": 2_048,
        "ema_decay_init": 0.9,
        "ema_definition": "learned bias-corrected causal EMA over the shared 128-d content projection",
        "coefficient_structures": {
            "scalar": "one decay per layer",
            "per_head": "one decay per layer and attention head",
            "per_dim": "one decay per layer and projected-content dimension, shared across heads",
        },
        "triage_thresholds": {
            "survive": "delta <= -0.01",
            "prune": "delta >= +0.01",
            "unresolved": "otherwise",
            "note": "single-seed screen; replicate only a clear survivor",
        },
        "arms": arms,
        "contrasts": contrasts,
    }


def render_markdown(results: dict) -> str:
    lines = [
        "# Phase-32 AddRoPE EMA coefficient-axis screen",
        "",
        "This is a paired seed-123, 15k-step screen. All four arms run",
        "sequentially under one lifetime GPU claim. Final loss uses the disjoint",
        "256-example holdout beginning at validation batch 2,048. Negative",
        "deltas favor the candidate.",
        "",
        "| Arm | 5k dev | 10k dev | 15k dev | Final | Tok/s | Peak MiB |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm in ARMS:
        values = results["arms"][arm]
        curve = values["development_curve"]
        lines.append(
            f"| {arm} | {curve['5000']:.6f} | {curve['10000']:.6f} | "
            f"{curve['15000']:.6f} | {values['final_loss']:.6f} | "
            f"{values['target_tokens_per_second']:,.0f} | "
            f"{values['peak_allocated_mib']:,.0f} |"
        )
    lines.extend(
        [
            "",
            "| Candidate | Direct control | Final delta | 95% paired-example CI | Triage |",
            "| --- | --- | ---: | --- | --- |",
        ]
    )
    for candidate, reference, _ in CONTRASTS:
        contrast = results["contrasts"][f"{candidate}_vs_{reference}"]
        low, high = contrast["paired_example_ci95"]
        lines.append(
            f"| {candidate} | {reference} | "
            f"{contrast['delta_candidate_minus_reference']:+.6f} | "
            f"[{low:+.6f}, {high:+.6f}] | {contrast['triage']} |"
        )
    lines.extend(
        [
            "",
            "The JSON companion includes learned decay/window diagnostics,",
            "carrier-delta magnitudes, elapsed time, throughput, and peak memory.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    results = analyze()
    rendered = render_markdown(results)
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    (RESULT_ROOT / "phase32_analysis.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n"
    )
    (RESULT_ROOT / "PHASE32_RESULTS.md").write_text(rendered)
    print(rendered)


if __name__ == "__main__":
    main()
