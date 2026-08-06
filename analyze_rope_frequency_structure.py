#!/usr/bin/env python
"""Fit compact static structure to the completed phase-20 frequency tables."""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import torch

from position import build_rope_frequencies


ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT / "model-output" / "position_bias_phase20_rope_frequency"
SEEDS = (123, 456, 789)
ARMS = ("layer-shared", "layer-head")


def run_dir(arm: str, seed: int) -> Path:
    return OUTPUT_ROOT / f"phase20-{arm}-seed{seed}-s5000-h768d8"


def load_table(arm: str, seed: int) -> torch.Tensor:
    state = torch.load(
        run_dir(arm, seed) / "pytorch_model.bin",
        map_location="cpu",
        weights_only=True,
    )
    layers = [
        state[f"blocks.{layer}.attn.rope_log_frequency_delta"].float()
        for layer in range(8)
    ]
    return torch.stack(layers)


def explained_fraction(target: torch.Tensor, reconstruction: torch.Tensor) -> float:
    total = target.square().sum()
    if total == 0:
        return 1.0
    return float(1.0 - (target - reconstruction).square().sum() / total)


def row_fit(target: torch.Tensor, design: torch.Tensor) -> tuple[float, torch.Tensor]:
    rows = target.reshape(-1, target.shape[-1])
    coefficients = torch.linalg.lstsq(design, rows.T).solution
    reconstructed = (design @ coefficients).T.reshape_as(target)
    return explained_fraction(target, reconstructed), reconstructed


def spline_design(pair_count: int, knots: int = 8) -> torch.Tensor:
    positions = torch.linspace(0.0, 1.0, pair_count)
    knot_positions = torch.linspace(0.0, 1.0, knots)
    spacing = float(knot_positions[1] - knot_positions[0])
    return (
        1.0
        - (positions[:, None] - knot_positions[None, :]).abs() / spacing
    ).clamp_min(0.0)


def summarize_table(table: torch.Tensor) -> dict:
    pair_count = table.shape[-1]
    log_frequency = build_rope_frequencies(pair_count * 2, 10_000.0).log()
    x = (log_frequency - log_frequency.mean()) / log_frequency.std()
    affine = torch.stack((torch.ones_like(x), x), dim=-1)
    spline = spline_design(pair_count)
    affine_explained, _ = row_fit(table, affine)
    spline_explained, _ = row_fit(table, spline)
    rows = table.reshape(-1, pair_count)
    singular_values = torch.linalg.svdvals(rows)
    energy = singular_values.square()
    cumulative = energy.cumsum(0) / energy.sum().clamp_min(1e-30)
    rank90 = int((cumulative < 0.9).sum().item() + 1)
    summary = {
        "raw_rms": float(table.square().mean().sqrt()),
        "raw_min": float(table.min()),
        "raw_max": float(table.max()),
        "affine_log_frequency_explained": affine_explained,
        "spline8_explained": spline_explained,
        "spectral_row_rank90": rank90,
    }
    if table.shape[1] > 1:
        head_shared = table.mean(dim=1, keepdim=True).expand_as(table)
        summary["head_shared_explained"] = explained_fraction(table, head_shared)
    return summary


def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left.flatten()
    right = right.flatten()
    return float(
        torch.dot(left, right)
        / (left.norm() * right.norm()).clamp_min(1e-30)
    )


def analyze() -> dict:
    results: dict = {"arms": {}}
    for arm in ARMS:
        tables = {seed: load_table(arm, seed) for seed in SEEDS}
        summaries = {str(seed): summarize_table(table) for seed, table in tables.items()}
        seed_cosines = {
            f"{left}_{right}": cosine(tables[left], tables[right])
            for index, left in enumerate(SEEDS)
            for right in SEEDS[index + 1 :]
        }
        mean_summary = {
            key: statistics.fmean(summary[key] for summary in summaries.values())
            for key in next(iter(summaries.values()))
        }
        results["arms"][arm] = {
            "by_seed": summaries,
            "mean": mean_summary,
            "cross_seed_cosine": seed_cosines,
            "mean_cross_seed_cosine": statistics.fmean(seed_cosines.values()),
        }
    return results


def render_markdown(results: dict) -> str:
    lines = [
        "# Phase-20 learned-frequency structure fits",
        "",
        "Explained fractions measure reconstruction energy of the trained raw",
        "log-frequency deltas. These are descriptive fits, not loss results.",
        "",
        "| Arm | Affine in log omega | Spline-8 | Head-shared | Spectral rank-90 | Cross-seed cosine |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm in ARMS:
        result = results["arms"][arm]
        mean = result["mean"]
        head_shared = mean.get("head_shared_explained")
        head_text = "—" if head_shared is None else f"{head_shared:.4f}"
        lines.append(
            f"| {arm} | {mean['affine_log_frequency_explained']:.4f} | "
            f"{mean['spline8_explained']:.4f} | {head_text} | "
            f"{mean['spectral_row_rank90']:.1f} | "
            f"{result['mean_cross_seed_cosine']:.4f} |"
        )
    lines.extend(
        [
            "",
            "Low cross-seed cosine would indicate that the reproducible loss",
            "effect does not select a reproducible detailed spectrum, weakening",
            "the case for fitting a richer static function to any one seed.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    results = analyze()
    rendered = render_markdown(results)
    (OUTPUT_ROOT / "frequency_structure.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n"
    )
    (OUTPUT_ROOT / "FREQUENCY_STRUCTURE.md").write_text(rendered)
    print(rendered)


if __name__ == "__main__":
    main()

