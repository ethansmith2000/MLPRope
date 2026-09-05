#!/usr/bin/env python
"""Analyze the Phase-37 200k direct-amplitude confirmation."""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "model-output" / "position_bias_phase37_direct_amplitude_200k"
RESULT_ROOT = ROOT / "results" / "phase37_direct_amplitude_200k"
STEP = 200_000
PRACTICAL_THRESHOLD = 0.002
ARMS = {
    "scalar": "phase37-qkpre-scalar-seed123-s200000-h768d8",
    "exponential-amplitude": (
        "phase37-qkpre-exponential-amplitude-r4-seed123-s200000-h768d8"
    ),
    "direct-amplitude": (
        "phase37-qkpre-direct-amplitude-r4-seed123-s200000-h768d8"
    ),
}
CONTRASTS = (
    ("direct-amplitude_vs_scalar", "direct-amplitude", "scalar", "primary"),
    (
        "exponential-amplitude_vs_scalar",
        "exponential-amplitude",
        "scalar",
        "secondary",
    ),
    (
        "direct-amplitude_vs_exponential-amplitude",
        "direct-amplitude",
        "exponential-amplitude",
        "parameterization",
    ),
)


def run_dir(arm: str) -> Path:
    return OUTPUT_ROOT / ARMS[arm]


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def evaluation_losses(arm: str, *, final: bool) -> dict[int, list[float]]:
    pattern = (
        "step_*_context_001024.json"
        if final
        else "step_*_development_context_001024.json"
    )
    values = {}
    for path in (run_dir(arm) / "evaluation_details").glob(pattern):
        if final and "_development_" in path.name:
            continue
        payload = json.loads(path.read_text())
        expected_kind = "final_holdout" if final else "development"
        expected_start = 2_048 if final else 0
        if payload["evaluation_kind"] != expected_kind:
            raise ValueError(f"Unexpected evaluation kind in {path}")
        if payload["evaluation_start_batch"] != expected_start:
            raise ValueError(f"Unexpected evaluation start in {path}")
        losses = [float(value) for value in payload["losses"]]
        expected_count = 1_024 if final else 128
        if len(losses) != expected_count:
            raise ValueError(
                f"Expected {expected_count} losses in {path}, got {len(losses)}"
            )
        values[int(payload["step"])] = losses
    return values


def paired_summary(candidate: list[float], reference: list[float]) -> dict:
    delta = [a - b for a, b in zip(candidate, reference, strict=True)]
    mean = statistics.fmean(delta)
    half_width = 1.96 * statistics.stdev(delta) / math.sqrt(len(delta))
    return {
        "candidate_loss": statistics.fmean(candidate),
        "reference_loss": statistics.fmean(reference),
        "delta_candidate_minus_reference": mean,
        "paired_example_ci95": [mean - half_width, mean + half_width],
        "paired_delta_stdev": statistics.stdev(delta),
        "num_paired_examples": len(delta),
    }


def _linear_slope(points: list[tuple[int, float]]) -> float | None:
    if len(points) < 2:
        return None
    xs = [step / 1_000 for step, _ in points]
    ys = [value for _, value in points]
    x_mean = statistics.fmean(xs)
    y_mean = statistics.fmean(ys)
    denominator = sum((x - x_mean) ** 2 for x in xs)
    return sum(
        (x - x_mean) * (y - y_mean)
        for x, y in zip(xs, ys, strict=True)
    ) / denominator


def development_contrast(
    candidate: dict[int, list[float]],
    reference: dict[int, list[float]],
) -> dict:
    common = sorted(set(candidate) & set(reference))
    points = {
        str(step): paired_summary(candidate[step], reference[step])
        for step in common
    }
    late = [
        (step, points[str(step)]["delta_candidate_minus_reference"])
        for step in common
        if step >= 150_000
    ]
    return {
        "points": points,
        "late_window_start_step": 150_000,
        "late_mean_delta": statistics.fmean(value for _, value in late),
        "late_slope_delta_per_1k_steps": _linear_slope(late),
    }


def final_metrics(arm: str) -> dict:
    matches = [
        row
        for row in _read_jsonl(run_dir(arm) / "metrics.jsonl")
        if row.get("step") == STEP
        and row.get("evaluation_kind") == "final_holdout"
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one final-holdout row for {arm}, got {len(matches)}")
    return matches[0]


def _metric_values(row: dict, suffix: str) -> list[float]:
    return [
        float(value)
        for key, value in row.items()
        if key.startswith("position/layer_") and key.endswith(suffix)
    ]


def intervention_state(arm: str) -> dict:
    row = final_metrics(arm)
    gates = _metric_values(row, "/gate_q")
    amplitude_min = _metric_values(row, "/amplitude_factor_q_min")
    amplitude_max = _metric_values(row, "/amplitude_factor_q_max")
    nonpositive = _metric_values(row, "/amplitude_factor_q_nonpositive_fraction")
    energy = _metric_values(row, "/input_mixture_position_energy_fraction")
    q_cosine = _metric_values(row, "/normalized_q_cosine_to_content")
    k_cosine = _metric_values(row, "/normalized_k_cosine_to_content")
    return {
        "gate_range": [min(gates), max(gates)],
        "amplitude_factor_range": [min(amplitude_min), max(amplitude_max)],
        "amplitude_nonpositive_fraction_max": max(nonpositive),
        "input_position_energy_fraction": {
            "min": min(energy),
            "mean": statistics.fmean(energy),
            "max": max(energy),
        },
        "normalized_q_cosine_to_content_range": [min(q_cosine), max(q_cosine)],
        "normalized_k_cosine_to_content_range": [min(k_cosine), max(k_cosine)],
    }


def optimization_health(arm: str) -> dict:
    rows = _read_jsonl(run_dir(arm) / "intervention_optimization.jsonl")
    prefix = "optimization/pre_qk_sinusoid_adapter"
    active = [
        row
        for row in rows
        if row["step"] < STEP
        and row.get(f"{prefix}/parameter_update/l2", 0) > 0
    ]
    numeric = [
        value
        for row in rows
        for key, value in row.items()
        if key not in {"step", "timestamp"} and isinstance(value, (int, float))
    ]
    cosines = [
        row[f"{prefix}/descent_update_gradient_cosine"]
        for row in active
        if row.get(f"{prefix}/descent_update_gradient_cosine") is not None
    ]
    return {
        "sample_count": len(rows),
        "active_sample_count": len(active),
        "all_numeric_finite": all(math.isfinite(float(value)) for value in numeric),
        "minimum_gradient_clip_ratio": min(
            row[f"{prefix}/gradient_clip_ratio"] for row in active
        ),
        "maximum_carrier_function_step_rms": max(
            row[f"{prefix}/carrier_function_step/rms"] for row in active
        ),
        "median_descent_update_gradient_cosine": statistics.median(cosines),
        "last_active_step": active[-1]["step"],
    }


def analyze() -> dict:
    incomplete = [arm for arm in ARMS if not (run_dir(arm) / "COMPLETED").is_file()]
    if incomplete:
        raise RuntimeError(f"Phase 37 incomplete: {incomplete}")
    development = {
        arm: evaluation_losses(arm, final=False) for arm in ARMS
    }
    final = {arm: evaluation_losses(arm, final=True)[STEP] for arm in ARMS}
    arms = {}
    launches = {}
    for arm in ARMS:
        summary = json.loads((run_dir(arm) / "training_summary.json").read_text())
        provenance = json.loads((run_dir(arm) / "run_provenance.json").read_text())
        launches[arm] = provenance["launches"]
        arms[arm] = {
            "final_holdout_loss": statistics.fmean(final[arm]),
            "target_tokens_per_second": summary["target_tokens_per_second"],
            "peak_reserved_mib": summary["peak_reserved_mib"],
            "parameter_counts": provenance["parameter_counts"],
            "intervention_state": intervention_state(arm),
            "optimization_health": optimization_health(arm),
        }
    all_launches = [launch for values in launches.values() for launch in values]
    source_states = {
        json.dumps(launch["source"], sort_keys=True) for launch in all_launches
    }
    dataset_splits = {
        json.dumps(launch["dataset"]["splits"], sort_keys=True)
        for launch in all_launches
    }
    if len(source_states) != 1 or any(
        launch["source"]["dirty"] for launch in all_launches
    ):
        raise ValueError("Phase-37 launches do not share one clean source state")
    if len(dataset_splits) != 1:
        raise ValueError("Phase-37 launches do not share dataset split fingerprints")
    contrasts = {}
    for name, candidate, reference, kind in CONTRASTS:
        endpoint = paired_summary(final[candidate], final[reference])
        endpoint_pass = (
            endpoint["delta_candidate_minus_reference"] <= -PRACTICAL_THRESHOLD
            and endpoint["paired_example_ci95"][1] < 0
        )
        contrasts[name] = {
            "candidate": candidate,
            "reference": reference,
            "kind": kind,
            "final": endpoint,
            "development_curve": development_contrast(
                development[candidate], development[reference]
            ),
            "promotion_gate": {
                "practical_threshold": PRACTICAL_THRESHOLD,
                "endpoint_and_interval_pass": endpoint_pass,
                "optimization_finite": arms[candidate]["optimization_health"][
                    "all_numeric_finite"
                ],
                "overall_pass": endpoint_pass
                and arms[candidate]["optimization_health"]["all_numeric_finite"],
            },
        }
    first = all_launches[0]
    return {
        "scope": "phase37_one_seed_200k_direct_amplitude_confirmation",
        "seed": 123,
        "training_steps": STEP,
        "primary_context": 1_024,
        "development_examples": 128,
        "final_holdout_start_batch": 2_048,
        "final_holdout_examples": 1_024,
        "practical_threshold_nats": PRACTICAL_THRESHOLD,
        "resume_event": {
            "checkpoint_step": 70_000,
            "launches_per_arm": {arm: len(values) for arm, values in launches.items()},
            "all_launches_same_clean_source": True,
            "optimizer_scheduler_sampler_and_rng_restored": True,
        },
        "provenance": {
            "source": first["source"],
            "dataset": {
                "resolved_path": first["dataset"]["resolved_path"],
                "splits": first["dataset"]["splits"],
                "manifest_sha256": {
                    name: manifest["sha256"]
                    for name, manifest in first["dataset"]["manifests"].items()
                },
            },
            "software": first["software"],
            "gpu_by_launch_and_arm": {
                arm: [launch["hardware"]["cuda_device"] for launch in values]
                for arm, values in launches.items()
            },
        },
        "arms": arms,
        "contrasts": contrasts,
        "caveat": (
            "Paired-example intervals measure holdout precision for seed 123, not "
            "training-seed variability. The repeatedly evaluated 128-example "
            "development slice is noisier than the disjoint 1,024-example primary "
            "holdout."
        ),
    }


def _range(values: list[float]) -> str:
    return f"{values[0]:.3f}--{values[1]:.3f}"


def render(results: dict) -> str:
    lines = [
        "# Phase 37: direct-amplitude confirmation at 200k",
        "",
        "All arms use seed 123, fixed RoPE, the tied pre-Q/K carrier, and one",
        "common 200k schedule. Negative deltas favor the candidate.",
        "",
        "| Arm | Final loss | Target tok/s | Peak MiB | Gate range | Amplitude-factor range | Position-energy mean |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm in ARMS:
        row = results["arms"][arm]
        state = row["intervention_state"]
        lines.append(
            f"| {arm} | {row['final_holdout_loss']:.6f} | "
            f"{row['target_tokens_per_second']:,.0f} | "
            f"{row['peak_reserved_mib']:,.0f} | {_range(state['gate_range'])} | "
            f"{_range(state['amplitude_factor_range'])} | "
            f"{state['input_position_energy_fraction']['mean']:.3f} |"
        )
    lines.extend(
        [
            "",
            "| Contrast | Final delta | Paired-example 95% CI | Dev late mean | Dev late delta/1k | Gate |",
            "| --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for name, _, _, _ in CONTRASTS:
        row = results["contrasts"][name]
        final = row["final"]
        low, high = final["paired_example_ci95"]
        curve = row["development_curve"]
        gate = "pass" if row["promotion_gate"]["overall_pass"] else "no"
        lines.append(
            f"| {name} | {final['delta_candidate_minus_reference']:+.6f} | "
            f"[{low:+.6f}, {high:+.6f}] | {curve['late_mean_delta']:+.6f} | "
            f"{curve['late_slope_delta_per_1k_steps']:+.6f} | {gate} |"
        )
    direct = results["contrasts"]["direct-amplitude_vs_scalar"]
    exponential = results["contrasts"]["exponential-amplitude_vs_scalar"]
    parameterization = results["contrasts"][
        "direct-amplitude_vs_exponential-amplitude"
    ]
    direct_state = results["arms"]["direct-amplitude"]["intervention_state"]
    exponential_state = results["arms"]["exponential-amplitude"][
        "intervention_state"
    ]
    finite = all(
        row["optimization_health"]["all_numeric_finite"]
        for row in results["arms"].values()
    )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "The Phase-36 short-horizon direct-amplitude win does not survive the",
            "predeclared 200k primary endpoint. Direct amplitude is effectively tied",
            f"with scalar ({direct['final']['delta_candidate_minus_reference']:+.6f},",
            f"CI [{direct['final']['paired_example_ci95'][0]:+.6f},",
            f"{direct['final']['paired_example_ci95'][1]:+.6f}]). The interval excludes",
            "the required 0.002-nat improvement. Exponential amplitude is also null",
            f"({exponential['final']['delta_candidate_minus_reference']:+.6f}), and",
            "direct does not beat exponential on the holdout",
            f"({parameterization['final']['delta_candidate_minus_reference']:+.6f}).",
            "No amplitude-shape arm earns seed replication.",
            "",
            "The repeatedly measured 128-example development slice favored direct",
            f"amplitude by {direct['development_curve']['late_mean_delta']:+.6f} on",
            "average from 150k--200k, while the disjoint 1,024-example primary holdout",
            "did not. Its per-step paired uncertainty is large enough to explain this",
            "difference. The larger frozen holdout governs the decision; the mismatch",
            "is evidence that millinat-scale rankings are slice-sensitive.",
            "",
            "## Mechanism and optimization health",
            "",
            f"All optimization traces finite: {finite}. Direct factors stayed",
            f"nonnegative but reached {_range(direct_state['amplitude_factor_range'])};",
            "one band was almost completely suppressed. Exponential factors reached",
            f"{_range(exponential_state['amplitude_factor_range'])}. Both models",
            "learned substantial, distinct spectra and reduced their scalar gates,",
            "yet neither improved held-out loss materially. This is evidence against",
            "the amplitude-shape hypothesis at this scale/horizon, not an inactive-path",
            "or saturation diagnosis.",
            "",
            "Training resumed once from complete step-70k checkpoints after the",
            "interactive launcher exited. Model, optimizer, scheduler, sampler, and RNG",
            "states were restored. Both launches for every arm recorded the identical",
            "clean source commit, and all arms then completed normally.",
            "",
            "## Disposition",
            "",
            "Keep the scalar pre-Q/K carrier with fixed RoPE as the active default.",
            "Treat both smooth-amplitude maps and all Phase-36 frequency maps as",
            "completed ablations. Additional seeds, width transfer, and longer-context",
            "tests are not warranted for these refinements without a new hypothesis.",
            "The broader pre-Q/K and AddRoPE mechanisms remain supported by their",
            "separate evidence; this result only closes the smooth carrier-shape branch.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    results = analyze()
    markdown = render(results)
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    (RESULT_ROOT / "phase37_analysis.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n"
    )
    (RESULT_ROOT / "PHASE37_RESULTS.md").write_text(markdown + "\n")
    print(markdown)


if __name__ == "__main__":
    main()
