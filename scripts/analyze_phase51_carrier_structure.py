#!/usr/bin/env python
"""Measure whether trained rank-32 Q/K carriers are mostly position-constant.

This analysis reads final model weights only and writes compact JSON/Markdown.
It creates no model checkpoint. The realized attention intervention is evaluated
separately by ``evaluate_phase51_carrier_counterfactuals.py``.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from position.basis import interleaved_fourier_basis


RESULT_ROOT = ROOT / "results" / "phase51_carrier_mechanism"
RUNS = {
    123: ROOT / "model-output" / "position_bias_phase49_mature_qk_readout"
    / "phase49-qk-readout-r32-seed123-b32-s100000-h768d8",
    456: ROOT / "model-output" / "position_bias_phase50_training_seed_replication"
    / "phase50-qk-readout-r32-seed456-b32-s100000-h768d8",
    789: ROOT / "model-output" / "position_bias_phase50_training_seed_replication"
    / "phase50-qk-readout-r32-seed789-b32-s100000-h768d8",
}


def _energy_fraction_in_mean(value: torch.Tensor) -> float:
    numerator = value.mean(dim=0).square().sum()
    denominator = value.square().sum(dim=-1).mean()
    return float((numerator / denominator.clamp_min(1e-30)).item())


def _rms(value: torch.Tensor) -> float:
    return float(value.square().mean().sqrt().item())


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    denominator = left.norm() * right.norm()
    return float((left.dot(right) / denominator.clamp_min(1e-30)).item())


def _analyze_run(seed: int, run_dir: Path) -> dict:
    config_path = run_dir / "training_config.json"
    weights_path = run_dir / "pytorch_model.bin"
    config = json.loads(config_path.read_text())
    qk_config = config["qk_preprojection"]
    if qk_config["mode"] != "low_rank_qk_residual":
        raise ValueError(f"Unexpected carrier mode in {config_path}")
    length = int(config["training_length"]) - 1
    basis = interleaved_fourier_basis(
        length,
        int(qk_config["basis_dim"]),
        float(qk_config["theta"]),
    )
    state = torch.load(weights_path, map_location="cpu", weights_only=True, mmap=True)
    layers = []
    for layer in range(int(config["depth"])):
        prefix = f"blocks.{layer}.attn"
        gate = state[f"{prefix}.qk_preprojection.gate"].float()
        down = state[f"{prefix}.qk_preprojection.down.weight"].float()
        q_up = state[f"{prefix}.qk_preprojection.q_up.weight"].float()
        k_up = state[f"{prefix}.qk_preprojection.k_up.weight"].float()
        q_projection = state[f"{prefix}.to_q.weight"].float()
        k_projection = state[f"{prefix}.to_k.weight"].float()

        latent = basis @ down.T
        q_direct = latent @ q_up.T
        k_direct = latent @ k_up.T
        anchor = basis * gate
        q_scalar = anchor @ q_projection.T
        k_scalar = anchor @ k_projection.T
        q_total = q_scalar + q_direct
        k_total = k_scalar + k_direct
        q_mean = q_direct.mean(dim=0)
        k_mean = k_direct.mean(dim=0)
        layer_result = {
            "layer": layer,
            "gate": float(gate.item()),
            "latent_mean_energy_fraction": _energy_fraction_in_mean(latent),
            "q_direct_mean_energy_fraction": _energy_fraction_in_mean(q_direct),
            "k_direct_mean_energy_fraction": _energy_fraction_in_mean(k_direct),
            "q_total_mean_energy_fraction": _energy_fraction_in_mean(q_total),
            "k_total_mean_energy_fraction": _energy_fraction_in_mean(k_total),
            "q_direct_rms": _rms(q_direct),
            "k_direct_rms": _rms(k_direct),
            "q_scalar_rms": _rms(q_scalar),
            "k_scalar_rms": _rms(k_scalar),
            "q_scalar_to_direct_rms": _rms(q_scalar) / max(_rms(q_direct), 1e-30),
            "k_scalar_to_direct_rms": _rms(k_scalar) / max(_rms(k_direct), 1e-30),
            "direct_mean_qk_cosine": _cosine(q_mean, k_mean),
        }
        if not all(
            math.isfinite(value)
            for key, value in layer_result.items()
            if key != "layer"
        ):
            raise ValueError(f"Non-finite carrier statistic in seed {seed} layer {layer}")
        layers.append(layer_result)

    def field_summary(field: str) -> dict:
        values = [float(layer[field]) for layer in layers]
        return {
            "mean": statistics.mean(values),
            "min": min(values),
            "max": max(values),
        }

    return {
        "seed": seed,
        "run_dir": str(run_dir.relative_to(ROOT)),
        "weights": {
            "path": str(weights_path.relative_to(ROOT)),
            "size_bytes": weights_path.stat().st_size,
            "mtime_ns": weights_path.stat().st_mtime_ns,
        },
        "carrier_length": length,
        "definition": (
            "||mean_p C(p)||_2^2 / mean_p ||C(p)||_2^2 over positions "
            "p=0,...,training_length-2"
        ),
        "layers": layers,
        "summary": {
            field: field_summary(field)
            for field in (
                "latent_mean_energy_fraction",
                "q_direct_mean_energy_fraction",
                "k_direct_mean_energy_fraction",
                "q_total_mean_energy_fraction",
                "k_total_mean_energy_fraction",
                "q_scalar_to_direct_rms",
                "k_scalar_to_direct_rms",
                "direct_mean_qk_cosine",
            )
        },
    }


def _render(payload: dict) -> str:
    lines = [
        "# Phase 51: trained carrier structure",
        "",
        "This is a weight-only diagnostic. It does not establish that the positional",
        "mean causes the loss improvement; the registered checkpoint counterfactual",
        "evaluations test that separately.",
        "",
        "The reported fraction is",
        "`||mean_p C(p)||^2 / mean_p ||C(p)||^2` over the 1,023 positions actually",
        "entering attention for each 1,024-token language-model block.",
        "",
        "| Seed | Q direct mean-energy range | K direct mean-energy range | Q scalar/direct RMS range | K scalar/direct RMS range |",
        "|---:|---:|---:|---:|---:|",
    ]
    for seed, run in payload["runs"].items():
        summary = run["summary"]
        q_mean = summary["q_direct_mean_energy_fraction"]
        k_mean = summary["k_direct_mean_energy_fraction"]
        q_ratio = summary["q_scalar_to_direct_rms"]
        k_ratio = summary["k_scalar_to_direct_rms"]
        lines.append(
            f"| {seed} | {q_mean['min']:.2%}--{q_mean['max']:.2%} | "
            f"{k_mean['min']:.2%}--{k_mean['max']:.2%} | "
            f"{q_ratio['min']:.5f}--{q_ratio['max']:.5f} | "
            f"{k_ratio['min']:.5f}--{k_ratio['max']:.5f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "A direct carrier dominated by its positional mean is close to a learned",
            "constant Q/K vector before QK normalization and RoPE. RoPE rotates such a",
            "vector by position, so it is not functionally position-free: its pure",
            "position--position numerator is a relative Toeplitz kernel",
            "`b_q^T R(r-p) b_k`. It can also create content--position cross terms.",
            "",
            "The very small scalar/direct RMS ratios show that the mature rank-32",
            "extension is structurally dominated by its dedicated projected branch.",
            "They do not by themselves prove that the scalar anchor is causally",
            "unnecessary, because QK normalization and all downstream hidden states",
            "change jointly when either branch is removed.",
            "",
            "## Decision use",
            "",
            "- If positional-mean-only evaluation preserves the gain and the",
            "  mean-removed component does not, train a matched constant-carrier/QK-bias",
            "  control before claiming that a rich Fourier readout is responsible.",
            "- If mean removal preserves the gain, retain the Fourier interpretation and",
            "  characterize the centered component.",
            "- If scalar removal is neutral only at evaluation but a no-anchor training",
            "  control loses, interpret the scalar as optimization scaffolding.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--require-seeds",
        type=int,
        nargs="*",
        default=None,
        help="Fail unless these seed checkpoints exist; default analyzes all available.",
    )
    args = parser.parse_args()
    required = set(args.require_seeds or ())
    available = {
        seed: run_dir
        for seed, run_dir in RUNS.items()
        if (run_dir / "COMPLETED").is_file()
        and (run_dir / "pytorch_model.bin").is_file()
    }
    missing = sorted(required - set(available))
    if missing:
        raise RuntimeError(f"Required completed checkpoints are missing: {missing}")
    if not available:
        raise RuntimeError("No completed rank-32 checkpoints are available")
    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "analysis_kind": "weight_only_noncausal_structure_diagnostic",
        "runs": {
            str(seed): _analyze_run(seed, run_dir)
            for seed, run_dir in sorted(available.items())
        },
    }
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    (RESULT_ROOT / "carrier_structure.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    report = _render(payload)
    (RESULT_ROOT / "CARRIER_STRUCTURE.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
