#!/usr/bin/env python
"""Analyze the paired, single-seed phase-26 mechanism breadth screen."""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT / "model-output" / "position_bias_phase26_breadth"
RESULT_ROOT = ROOT / "results" / "phase26_position_breadth"
SEED = 123
STEPS = 5_000
ARMS = (
    "rope-fixed",
    "nope",
    "addrope-a10",
    "posgain-q",
    "posgain-k",
    "posgain-qk",
    "qkpre-nope",
    "qkpre-rope",
    "clock-pointwise",
    "clock-causalconv",
)
REFERENCES = {
    "nope": "rope-fixed",
    "addrope-a10": "rope-fixed",
    "posgain-q": "rope-fixed",
    "posgain-k": "rope-fixed",
    "posgain-qk": "rope-fixed",
    "qkpre-nope": "nope",
    "qkpre-rope": "rope-fixed",
    "clock-pointwise": "rope-fixed",
    "clock-causalconv": "rope-fixed",
}


def run_dir(arm: str) -> Path:
    return OUTPUT_ROOT / f"phase26-{arm}-seed{SEED}-s{STEPS}-h768d8"


def load_losses(arm: str) -> list[float]:
    path = (
        run_dir(arm)
        / "evaluation_details"
        / "step_00005000_context_001024.json"
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
    path = run_dir(arm) / "metrics.jsonl"
    final = None
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if row.get("evaluation_kind") == "final_holdout":
            final = row
    if final is None:
        raise ValueError(f"No final-holdout metrics in {path}")
    return final


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
    """Broad-screen label; this is deliberately not a replication claim."""
    if delta <= -0.02:
        return "survive"
    if delta >= 0.01:
        return "prune"
    if abs(delta) < 0.01:
        return "unresolved"
    return "weak_benefit" if delta < 0 else "weak_harm"


def mechanism_health(metrics: dict) -> dict:
    selected = {}
    suffixes = {
        "position_gain": (
            "q_gain_min",
            "q_gain_max",
            "q_near_bound_fraction",
            "k_gain_min",
            "k_gain_max",
            "k_near_bound_fraction",
        ),
        "rotary_clock": (
            "speed_min",
            "speed_max",
            "clock_final_drift_rms",
            "phase_delta_abs_max",
        ),
        "qk_preprojection": (
            "gate",
            "input_rms",
            "projected_q_rms",
            "projected_k_rms",
        ),
    }
    for mechanism, names in suffixes.items():
        if not any(f"/{mechanism}/" in key for key in metrics):
            continue
        summary = {}
        for name in names:
            values = [
                float(value)
                for key, value in metrics.items()
                if f"/{mechanism}/" in key and key.endswith(f"/{name}")
            ]
            if values:
                summary[name] = {
                    "layer_min": min(values),
                    "layer_mean": statistics.fmean(values),
                    "layer_max": max(values),
                }
        selected[mechanism] = summary
    return selected


def analyze() -> dict:
    missing = [
        str(run_dir(arm) / "COMPLETED")
        for arm in ARMS
        if not (run_dir(arm) / "COMPLETED").is_file()
    ]
    if missing:
        raise RuntimeError(
            "Phase-26 breadth screen is incomplete; missing markers:\n"
            + "\n".join(missing)
        )
    losses = {arm: load_losses(arm) for arm in ARMS}
    metrics = {arm: load_final_metrics(arm) for arm in ARMS}
    arms = {}
    for arm in ARMS:
        summary = json.loads((run_dir(arm) / "training_summary.json").read_text())
        arms[arm] = {
            "loss": statistics.fmean(losses[arm]),
            "target_tokens_per_second": summary["target_tokens_per_second"],
            "mechanism_health": mechanism_health(metrics[arm]),
        }
    contrasts = {}
    for candidate, reference in REFERENCES.items():
        result = paired_summary(losses[candidate], losses[reference])
        result.update(
            {
                "candidate": candidate,
                "reference": reference,
                "negative_favors_candidate": True,
                "triage": triage(result["delta_candidate_minus_reference"]),
            }
        )
        contrasts[f"{candidate}_vs_{reference}"] = result
    return {
        "scope": "single_seed_breadth_screen",
        "seed": SEED,
        "primary_context": 1_024,
        "final_holdout_examples": 256,
        "final_holdout_start_batch": 2_048,
        "triage_thresholds": {
            "survive": "delta <= -0.02",
            "prune": "delta >= +0.01",
            "unresolved": "abs(delta) < 0.01",
            "note": "only survivors are eligible for later seed replication",
        },
        "arms": arms,
        "contrasts": contrasts,
    }


def render_markdown(results: dict) -> str:
    lines = [
        "# Phase-26 dynamic-position breadth screen",
        "",
        "This is a paired seed-123 mechanism screen, not a multi-seed result.",
        "Loss uses the disjoint 256-example holdout beginning at validation",
        "batch 2,048. Deltas are candidate minus its direct control; negative",
        "is better. Only clear survivors should receive seed replication.",
        "",
        "| Arm | Final loss | Target tok/s | Direct control | Delta | Triage |",
        "| --- | ---: | ---: | --- | ---: | --- |",
    ]
    lines.append(
        f"| rope-fixed | {results['arms']['rope-fixed']['loss']:.6f} | "
        f"{results['arms']['rope-fixed']['target_tokens_per_second']:,.0f} | — | — | control |"
    )
    for arm in ARMS[1:]:
        contrast = results["contrasts"][f"{arm}_vs_{REFERENCES[arm]}"]
        lines.append(
            f"| {arm} | {results['arms'][arm]['loss']:.6f} | "
            f"{results['arms'][arm]['target_tokens_per_second']:,.0f} | "
            f"{REFERENCES[arm]} | "
            f"{contrast['delta_candidate_minus_reference']:+.6f} | "
            f"{contrast['triage']} |"
        )
    lines.extend(
        [
            "",
            "The JSON companion contains paired-example confidence intervals",
            "and layer-aggregated mechanism-health diagnostics.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    results = analyze()
    rendered = render_markdown(results)
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    (RESULT_ROOT / "phase26_analysis.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n"
    )
    (RESULT_ROOT / "PHASE26_RESULTS.md").write_text(rendered)
    print(rendered)


if __name__ == "__main__":
    main()
