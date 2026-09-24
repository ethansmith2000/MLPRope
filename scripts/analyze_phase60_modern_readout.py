#!/usr/bin/env python
"""Analyze the frozen Phase-60 modern dedicated-readout cohort."""

from __future__ import annotations

import json
import math
from pathlib import Path
import statistics

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "sweep_configs" / "phase60_modern_readout"
RESULT_ROOT = ROOT / "results" / "phase60_modern_readout"
STEP = 100_000
FINAL_START = 10_240
FINAL_BLOCKS = 1_024
MATERIALITY = -0.003
ARMS = ("scalar", "rank32", "rank128")
LABELS = {
    "scalar": "modern scalar pre-Q/K + RoPE",
    "rank32": "modern scalar + rank-32 Q/K readout + RoPE",
    "rank128": "modern scalar + rank-128 Q/K readout + RoPE",
}
EXPECTED_MODES = {
    "scalar": ("tied_scalar", 32, 1.0),
    "rank32": ("low_rank_qk_residual", 32, 6.367487169620489),
    "rank128": ("low_rank_qk_residual", 128, 2.449489742783178),
}
FUNCTION_KEY = "optimization/pre_qk_sinusoid_adapter/carrier_function_step/rms"


def _configs() -> dict[str, dict]:
    result = {}
    for path in sorted(CONFIG_ROOT.glob("*.json")):
        arm = path.stem.split("-", 1)[1]
        result[arm] = json.loads(path.read_text())
    if tuple(result) != ARMS:
        raise ValueError(f"Phase-60 config matrix is incomplete: {tuple(result)}")
    for arm, config in result.items():
        required = {
            "backbone_variant": "modern",
            "ff_hidden_dim": 2_048,
            "max_train_steps": STEP,
            "learning_rate": 1.2e-3,
            "training_length": 1_024,
            "per_device_train_batch_size": 32,
            "seed": 123,
            "paired_initialization_seed": 123,
            "qk_projection_bias": False,
            "qk_norm_mode": "method_aware_rms",
            "final_validation_start_batch": FINAL_START,
            "num_final_validation_batches": FINAL_BLOCKS,
            "save_final_model": False,
            "use_rope": True,
        }
        for key, expected in required.items():
            if config.get(key) != expected:
                raise ValueError(f"Unexpected {arm} config {key}: {config.get(key)!r}")
        mode, rank, multiplier = EXPECTED_MODES[arm]
        qkpre = config["qk_preprojection"]
        if qkpre["mode"] != mode or qkpre["rank"] != rank:
            raise ValueError(f"Unexpected {arm} readout configuration: {qkpre}")
        if not math.isclose(
            float(qkpre["readout_lr_multiplier"]), multiplier, rel_tol=0, abs_tol=1e-12
        ):
            raise ValueError(f"Unexpected {arm} readout LR multiplier")
    return result


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _numeric_finite(rows: list[dict]) -> bool:
    return all(
        math.isfinite(float(value))
        for row in rows
        for value in row.values()
        if isinstance(value, (int, float))
    )


def _losses(run_dir: Path) -> np.ndarray:
    path = run_dir / "evaluation_details" / f"step_{STEP:08d}_context_001024.json"
    payload = json.loads(path.read_text())
    values = np.asarray(payload["losses"], dtype=np.float64)
    if (
        payload.get("evaluation_kind") != "final_holdout"
        or payload.get("evaluation_start_batch") != FINAL_START
        or values.shape != (FINAL_BLOCKS,)
        or not np.isfinite(values).all()
    ):
        raise ValueError(f"Invalid final evaluation details: {path}")
    return values


def _interval(
    values: np.ndarray,
    seed: int,
    *,
    block_size: int | None = None,
) -> list[float]:
    if block_size is not None:
        if values.size % block_size:
            raise ValueError("Block size must divide evaluation examples")
        values = values.reshape(-1, block_size).mean(axis=1)
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(10):
        indices = rng.integers(0, values.size, size=(2_000, values.size))
        samples.append(values[indices].mean(axis=1))
    return [
        float(value)
        for value in np.quantile(np.concatenate(samples), (0.025, 0.975))
    ]


def _contrast(values: np.ndarray, reference: np.ndarray, seed: int) -> dict:
    delta = values - reference
    return {
        "mean_delta": float(delta.mean()),
        "iid_bootstrap_ci95": _interval(delta, seed),
        "contiguous_block_32_bootstrap_ci95": _interval(
            delta, seed + 1, block_size=32
        ),
    }


def _values(row: dict, suffix: str) -> list[float]:
    return [
        float(value)
        for key, value in sorted(row.items())
        if key.endswith(suffix) and isinstance(value, (int, float))
    ]


def _mean_or_none(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _carrier_summary(row: dict) -> dict:
    gates = _values(row, "/qk_preprojection/gate")
    result = {
        "effective_gate_min": min(gates),
        "effective_gate_mean": float(np.mean(gates)),
        "effective_gate_max": max(gates),
    }
    diagnostic_suffixes = {
        "input_position_to_content_rms_mean": (
            "/qk_preprojection/input_mixture_position_to_content_rms_ratio"
        ),
        "projected_q_position_to_content_rms_mean": (
            "/qk_preprojection/projected_q_mixture_position_to_content_rms_ratio"
        ),
        "projected_k_position_to_content_rms_mean": (
            "/qk_preprojection/projected_k_mixture_position_to_content_rms_ratio"
        ),
        "normalized_q_cosine_to_content_mean": (
            "/qk_preprojection/normalized_q_cosine_to_content"
        ),
        "normalized_k_cosine_to_content_mean": (
            "/qk_preprojection/normalized_k_cosine_to_content"
        ),
        "direct_q_rms_mean": "/qk_preprojection/direct_q_rms",
        "direct_k_rms_mean": "/qk_preprojection/direct_k_rms",
    }
    for name, suffix in diagnostic_suffixes.items():
        result[name] = _mean_or_none(_values(row, suffix))
    return result


def _development(rows: list[dict]) -> list[dict]:
    result = []
    for row in rows:
        if row.get("evaluation_kind") != "development":
            continue
        result.append(
            {
                "step": int(row["step"]),
                "mean_nll": float(row["eval_loss/context_1024"]),
                **_carrier_summary(row),
            }
        )
    return result


def _function_steps(rows: list[dict]) -> dict[int, float]:
    return {
        int(row["step"]): float(row[FUNCTION_KEY])
        for row in rows
        if isinstance(row.get(FUNCTION_KEY), (int, float))
        and float(row[FUNCTION_KEY]) > 0
    }


def _health(run_dir: Path) -> dict:
    marker = json.loads((run_dir / "COMPLETED").read_text())
    if int(marker.get("completed_steps", -1)) != STEP:
        raise ValueError(f"Incomplete run: {run_dir}")
    metrics = _jsonl(run_dir / "metrics.jsonl")
    optimizer_path = run_dir / "intervention_optimization.jsonl"
    optimizer = _jsonl(optimizer_path)
    development = _development(metrics)
    if not development or development[-1]["step"] != STEP:
        raise ValueError(f"Missing step-{STEP} development metrics: {run_dir}")
    final_row = next(
        row for row in reversed(metrics) if row.get("evaluation_kind") == "final_holdout"
    )
    summary = json.loads((run_dir / "training_summary.json").read_text())
    provenance = json.loads((run_dir / "run_provenance.json").read_text())
    return {
        "total_parameters": int(provenance["parameter_counts"]["total"]),
        "non_embedding_head_parameters": int(
            provenance["parameter_counts"]["non_embed"]
        ),
        "position_parameters": int(
            provenance["parameter_counts"]["position_params"]
        ),
        "target_tokens_per_second": float(summary["target_tokens_per_second"]),
        "elapsed_seconds": float(summary["elapsed_seconds"]),
        "peak_allocated_mib": float(summary["peak_allocated_mib"]),
        "peak_reserved_mib": float(summary["peak_reserved_mib"]),
        "metrics_finite": _numeric_finite(metrics),
        "optimizer_metrics_finite": _numeric_finite(optimizer),
        "development": development,
        "final_carrier_summary": _carrier_summary(final_row),
        "function_steps": _function_steps(optimizer),
    }


def _cleanup_summary() -> dict:
    path = RESULT_ROOT / "checkpoint_cleanup_events.jsonl"
    events = _jsonl(path) if path.is_file() else []
    return {
        "event_count": len(events),
        "reclaimed_bytes": sum(int(event["reclaimed_bytes"]) for event in events),
        "runs_with_removed_checkpoints": sorted(
            {event["run_name"] for event in events if event.get("removed")}
        ),
    }


def analyze() -> dict:
    configs = _configs()
    run_dirs = {arm: Path(configs[arm]["output_dir"]) for arm in ARMS}
    losses = {arm: _losses(run_dirs[arm]) for arm in ARMS}
    health = {arm: _health(run_dirs[arm]) for arm in ARMS}
    comparisons = {
        "rank32_minus_scalar": _contrast(losses["rank32"], losses["scalar"], 60_032),
        "rank128_minus_scalar": _contrast(losses["rank128"], losses["scalar"], 60_128),
        "rank128_minus_rank32": _contrast(losses["rank128"], losses["rank32"], 60_160),
    }

    scalar_curve = {
        point["step"]: point["mean_nll"] for point in health["scalar"]["development"]
    }
    rank32_curve = {
        point["step"]: point["mean_nll"] for point in health["rank32"]["development"]
    }
    for arm in ("rank32", "rank128"):
        health[arm]["development"] = [
            {
                **point,
                "delta_vs_scalar": point["mean_nll"] - scalar_curve[point["step"]],
                **(
                    {"delta_vs_rank32": point["mean_nll"] - rank32_curve[point["step"]]}
                    if arm == "rank128"
                    else {}
                ),
            }
            for point in health[arm]["development"]
        ]

    r32_steps = health["rank32"].pop("function_steps")
    r128_steps = health["rank128"].pop("function_steps")
    health["scalar"].pop("function_steps")
    ratios = {
        step: r128_steps[step] / r32_steps[step]
        for step in sorted(r32_steps.keys() & r128_steps.keys())
    }
    mature_ratios = [
        ratio for step, ratio in ratios.items() if 5_000 <= step <= 95_000
    ]
    if not mature_ratios:
        raise ValueError("No shared mature rank function-step diagnostics")
    ratio_median = float(statistics.median(mature_ratios))
    function_match = 0.8 <= ratio_median <= 1.25

    late_steps = (80_000, 85_000, 90_000, 95_000, 100_000)
    r32_development = {
        point["step"]: point for point in health["rank32"]["development"]
    }
    r128_development = {
        point["step"]: point for point in health["rank128"]["development"]
    }
    late_r32 = [r32_development[step]["delta_vs_scalar"] for step in late_steps]
    late_r128 = [r128_development[step]["delta_vs_rank32"] for step in late_steps]
    all_finite = all(
        item["metrics_finite"] and item["optimizer_metrics_finite"]
        for item in health.values()
    )
    r32_contrast = comparisons["rank32_minus_scalar"]
    r128_contrast = comparisons["rank128_minus_rank32"]
    r32_pass = bool(
        r32_contrast["mean_delta"] <= MATERIALITY
        and r32_contrast["contiguous_block_32_bootstrap_ci95"][1] < 0
        and all(delta < 0 for delta in late_r32)
        and all_finite
    )
    r128_pass = bool(
        function_match
        and r128_contrast["mean_delta"] <= MATERIALITY
        and r128_contrast["contiguous_block_32_bootstrap_ci95"][1] < 0
        and all(delta < 0 for delta in late_r128)
        and all_finite
    )

    return {
        "scope": "phase60_modern_dedicated_readout",
        "protocol": "paper/MODERN_READOUT_PROTOCOL.md",
        "training_seed": 123,
        "training_steps": STEP,
        "sequence_batch": 32,
        "context": 1_024,
        "learning_rate": 1.2e-3,
        "final_holdout": {
            "start_batch": FINAL_START,
            "blocks": FINAL_BLOCKS,
            "previously_inspected": False,
        },
        "methods": {
            arm: {
                "label": LABELS[arm],
                "mean_nll": float(losses[arm].mean()),
                "block_nll_standard_error": float(
                    losses[arm].std(ddof=1) / math.sqrt(losses[arm].size)
                ),
                **health[arm],
            }
            for arm in ARMS
        },
        "comparisons": comparisons,
        "rank_function_step_scale_r128_over_r32": {
            "step_5000_to_95000_median": ratio_median,
            "mature_min": min(mature_ratios),
            "mature_max": max(mature_ratios),
            "passes_registered_match": function_match,
            "ratios": {str(step): value for step, value in ratios.items()},
        },
        "late_curve": {
            "steps": list(late_steps),
            "rank32_minus_scalar": late_r32,
            "rank128_minus_rank32": late_r128,
        },
        "registered_decisions": {
            "materiality_threshold": MATERIALITY,
            "all_metrics_finite": all_finite,
            "rank32_transfers": r32_pass,
            "rank128_is_necessary": r128_pass,
            "next": (
                "prefer rank128 as the modern extension"
                if r128_pass
                else (
                    "retain rank32 as the efficient modern extension"
                    if r32_pass
                    else "retain the scalar primary method; do not tune the readout on this window"
                )
            ),
        },
        "checkpoint_cleanup": _cleanup_summary(),
        "inference_limit": (
            "This conditional modern-backbone extension cohort uses one training seed. "
            "Paired block intervals measure endpoint-stream precision, not training-seed "
            "uncertainty. The inherited controlled-backbone capacity control is not a "
            "new modern-backbone capacity control."
        ),
    }


def render(payload: dict) -> str:
    cleanup = payload["checkpoint_cleanup"]
    lines = [
        "# Phase 60: modern dedicated carrier readouts",
        "",
        "| Arm | Final NLL | Position params | Total params | ktok/s | Peak alloc GiB | Final gate mean |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        row = payload["methods"][arm]
        carrier = row["final_carrier_summary"]
        lines.append(
            f"| {row['label']} | {row['mean_nll']:.6f} | "
            f"{row['position_parameters']:,} | {row['total_parameters']:,} | "
            f"{row['target_tokens_per_second'] / 1000:.1f} | "
            f"{row['peak_allocated_mib'] / 1024:.2f} | "
            f"{carrier['effective_gate_mean']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Paired final-window contrasts",
            "",
            "Negative deltas favor the first named arm.",
            "",
            "| Contrast | Mean delta | IID 95% interval | Block-32 95% interval |",
            "|---|---:|---:|---:|",
        ]
    )
    for name, comparison in payload["comparisons"].items():
        iid = comparison["iid_bootstrap_ci95"]
        block = comparison["contiguous_block_32_bootstrap_ci95"]
        lines.append(
            f"| {name} | {comparison['mean_delta']:+.6f} | "
            f"[{iid[0]:+.6f}, {iid[1]:+.6f}] | "
            f"[{block[0]:+.6f}, {block[1]:+.6f}] |"
        )
    scale = payload["rank_function_step_scale_r128_over_r32"]
    decisions = payload["registered_decisions"]
    lines.extend(
        [
            "",
            "## Registered decisions",
            "",
            f"- Median rank-128/rank-32 carrier-function-step ratio: "
            f"`{scale['step_5000_to_95000_median']:.3f}`; registered match: "
            f"**{scale['passes_registered_match']}**.",
            f"- Rank 32 transfers beyond scalar: **{decisions['rank32_transfers']}**.",
            f"- Rank 128 is necessary beyond rank 32: **{decisions['rank128_is_necessary']}**.",
            f"- Decision: {decisions['next']}.",
            "",
            "## Artifact closeout",
            "",
            f"Recovery cleanup reclaimed `{cleanup['reclaimed_bytes'] / 2**30:.2f}` GiB "
            f"across `{len(cleanup['runs_with_removed_checkpoints'])}` completed runs. "
            "No final weights were saved.",
            "",
            payload["inference_limit"],
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    payload = analyze()
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    (RESULT_ROOT / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    report = render(payload)
    (RESULT_ROOT / "REPORT.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
