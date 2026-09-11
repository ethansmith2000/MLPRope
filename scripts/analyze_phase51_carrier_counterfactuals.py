#!/usr/bin/env python
"""Aggregate completed Phase-51 carrier counterfactuals across seeds."""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results" / "phase51_carrier_mechanism"
INTERVENTIONS = (
    "direct_mean_only",
    "direct_mean_removed",
    "direct_zero",
    "scalar_zero",
    "all_zero",
)
T_CRITICAL = {1: 12.706204736432095, 2: 4.302652729911275}


def _interval(values: list[float]) -> list[float] | None:
    if len(values) < 2:
        return None
    critical = T_CRITICAL[len(values) - 1]
    radius = critical * statistics.stdev(values) / math.sqrt(len(values))
    mean = statistics.mean(values)
    return [mean - radius, mean + radius]


def main() -> None:
    runs = {}
    for seed in (123, 456, 789):
        path = RESULT_ROOT / f"counterfactual_seed{seed}.json"
        if path.is_file():
            runs[str(seed)] = json.loads(path.read_text())
    if not runs:
        raise RuntimeError("No completed Phase-51 counterfactual results")

    comparisons = {}
    for intervention in INTERVENTIONS:
        by_seed = {
            seed: run["interventions"][intervention]["delta_vs_full"]["mean"]
            for seed, run in runs.items()
        }
        values = list(by_seed.values())
        comparisons[intervention] = {
            "by_seed": by_seed,
            "seed_mean_delta": statistics.mean(values),
            "seed_sample_std": statistics.stdev(values) if len(values) > 1 else None,
            "seed_t_interval95": _interval(values),
        }
    payload = {
        "schema_version": 1,
        "available_seeds": [int(seed) for seed in runs],
        "comparisons": comparisons,
        "inference_note": (
            "The seed is the replication unit. Per-checkpoint example/block "
            "intervals are holdout-sampling sensitivity analyses only."
        ),
    }
    (RESULT_ROOT / "counterfactual_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    lines = [
        "# Phase 51: carrier counterfactual summary",
        "",
        f"Available training seeds: {', '.join(runs)}.",
        "Positive deltas mean the intervention hurts relative to the trained full path.",
        "",
        "| Intervention | Seed deltas | Mean | Seed t interval |",
        "|---|---:|---:|---:|",
    ]
    for intervention, comparison in comparisons.items():
        seed_values = ", ".join(
            f"{seed}: {value:+.6f}"
            for seed, value in comparison["by_seed"].items()
        )
        interval = comparison["seed_t_interval95"]
        interval_text = (
            "n/a"
            if interval is None
            else f"[{interval[0]:+.6f}, {interval[1]:+.6f}]"
        )
        lines.append(
            f"| {intervention} | {seed_values} | "
            f"{comparison['seed_mean_delta']:+.6f} | {interval_text} |"
        )
    lines.extend(["", payload["inference_note"], ""])
    report = "\n".join(lines)
    (RESULT_ROOT / "COUNTERFACTUAL_SUMMARY.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
