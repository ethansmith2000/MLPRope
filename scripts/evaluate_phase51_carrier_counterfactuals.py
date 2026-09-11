#!/usr/bin/env python
"""Evaluate causal ablations of a trained rank-32 projected Q/K carrier.

The script is intended to run under ``gpu-claim`` with one visible GPU. It
loads an existing final model, writes only compact losses/statistics, and does
not create or modify any model weights or checkpoints.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import datasets
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import default_data_collator


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from position import QKPreprojectionPosition
from train_gpt import make_model


RESULT_ROOT = ROOT / "results" / "phase51_carrier_mechanism"
RUNS = {
    123: ROOT / "model-output" / "position_bias_phase49_mature_qk_readout"
    / "phase49-qk-readout-r32-seed123-b32-s100000-h768d8",
    456: ROOT / "model-output" / "position_bias_phase50_training_seed_replication"
    / "phase50-qk-readout-r32-seed456-b32-s100000-h768d8",
    789: ROOT / "model-output" / "position_bias_phase50_training_seed_replication"
    / "phase50-qk-readout-r32-seed789-b32-s100000-h768d8",
}
INTERVENTIONS = (
    "full",
    "direct_mean_only",
    "direct_mean_removed",
    "direct_zero",
    "scalar_zero",
    "all_zero",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _quantile_interval(samples: np.ndarray) -> list[float]:
    return [float(value) for value in np.quantile(samples, (0.025, 0.975))]


def _iid_bootstrap(delta: np.ndarray, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(10):
        indices = rng.integers(0, delta.size, size=(2_000, delta.size))
        means.append(delta[indices].mean(axis=1))
    return _quantile_interval(np.concatenate(means))


def _contiguous_block_bootstrap(
    delta: np.ndarray,
    *,
    block_length: int,
    seed: int,
) -> list[float]:
    if delta.size % block_length:
        raise ValueError("Counterfactual holdout must divide the block length")
    block_means = delta.reshape(-1, block_length).mean(axis=1)
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(10):
        indices = rng.integers(
            0,
            block_means.size,
            size=(2_000, block_means.size),
        )
        means.append(block_means[indices].mean(axis=1))
    return _quantile_interval(np.concatenate(means))


def _set_intervention(model: torch.nn.Module, intervention: str) -> int:
    count = 0
    for module in model.modules():
        if isinstance(module, QKPreprojectionPosition):
            module.set_evaluation_intervention(intervention)
            count += 1
    return count


@torch.inference_mode()
def _evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    intervention: str,
    device: torch.device,
) -> tuple[np.ndarray, float]:
    count = _set_intervention(model, intervention)
    if count == 0:
        raise RuntimeError("No Q/K preprojection modules found")
    start = time.perf_counter()
    losses = []
    for batch in loader:
        tokens = batch["input_ids"].to(device, non_blocking=True)
        input_ids = tokens[:, :-1]
        targets = tokens[:, 1:]
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(input_ids=input_ids)
        token_losses = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]).float(),
            targets.reshape(-1),
            reduction="none",
        ).reshape(targets.shape)
        losses.append(token_losses.mean(dim=-1).cpu())
        del logits, token_losses, tokens, input_ids, targets
    torch.cuda.synchronize(device)
    values = torch.cat(losses).double().numpy()
    if not np.isfinite(values).all():
        raise ValueError(f"Non-finite losses for {intervention}")
    return values, time.perf_counter() - start


def _render(payload: dict) -> str:
    full = np.asarray(payload["interventions"]["full"]["losses"])
    lines = [
        f"# Phase 51 carrier counterfactuals — seed {payload['seed']}",
        "",
        "All interventions use the same trained checkpoint and holdout blocks.",
        "QK normalization and every downstream hidden state are recomputed.",
        "Negative deltas relative to `full` are better.",
        "",
        "| Intervention | NLL | Delta vs full | IID 95% interval | Block-32 95% interval |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in INTERVENTIONS:
        result = payload["interventions"][name]
        delta = np.asarray(result["losses"]) - full
        lines.append(
            f"| {name} | {result['mean_loss']:.6f} | {delta.mean():+.6f} | "
            f"[{result['delta_vs_full']['iid_bootstrap_ci95'][0]:+.6f}, "
            f"{result['delta_vs_full']['iid_bootstrap_ci95'][1]:+.6f}] | "
            f"[{result['delta_vs_full']['contiguous_block_32_ci95'][0]:+.6f}, "
            f"{result['delta_vs_full']['contiguous_block_32_ci95'][1]:+.6f}] |"
        )
    validation = payload["full_reproduction_check"]
    lines.extend(
        [
            "",
            "## Full-path reproduction check",
            "",
            f"Original saved mean NLL: `{validation['saved_mean_loss']:.6f}`; "
            f"fresh full-path mean: `{validation['fresh_mean_loss']:.6f}`; "
            f"difference: `{validation['mean_difference']:+.8f}`.",
            "",
            "## Interpretation limits",
            "",
            "These are checkpoint interventions, not independently trained models.",
            "A neutral removal can identify what the trained network currently needs,",
            "but it cannot show whether a branch was useful as optimization scaffolding.",
            "The block-32 interval is a sensitivity analysis for adjacent validation",
            "blocks that may share source-document context; it is not a second corpus",
            "or training-seed replicate.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=tuple(RUNS), required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Counterfactual evaluation requires a visible CUDA GPU")
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")

    output_json = RESULT_ROOT / f"counterfactual_seed{args.seed}.json"
    output_report = RESULT_ROOT / f"COUNTERFACTUAL_SEED{args.seed}.md"
    if output_json.exists() and not args.overwrite:
        print(f"Already complete: {output_json}")
        return

    run_dir = RUNS[args.seed]
    if not (run_dir / "COMPLETED").is_file():
        raise RuntimeError(f"Run is not complete: {run_dir}")
    config = json.loads((run_dir / "training_config.json").read_text())
    weights_path = run_dir / "pytorch_model.bin"
    state = torch.load(weights_path, map_location="cpu", weights_only=True, mmap=True)
    vocab_size = int(state["token_embedding.weight"].shape[0])
    model = make_model(SimpleNamespace(**config), vocab_size)
    model.load_state_dict(state, strict=True)
    del state
    model.requires_grad_(False)
    model.eval()
    device = torch.device("cuda", 0)
    model.to(device)

    dataset_path = Path(config["tokenized_dataset_path"])
    dataset = datasets.load_from_disk(str(dataset_path))["validation"]
    start_batch = int(config["final_validation_start_batch"])
    blocks = int(config["num_final_validation_batches"])
    if blocks != 1_024:
        raise ValueError(f"Expected 1,024 final blocks, got {blocks}")
    evaluation = dataset.select(range(start_batch, start_batch + blocks))
    loader = DataLoader(
        evaluation,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=default_data_collator,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=bool(args.num_workers),
    )

    original_path = run_dir / "evaluation_details" / (
        "step_00100000_context_001024.json"
    )
    original_payload = json.loads(original_path.read_text())
    original = np.asarray(original_payload["losses"], dtype=np.float64)
    if original.shape != (blocks,):
        raise ValueError(f"Unexpected original loss shape in {original_path}")

    intervention_results = {}
    full = None
    for index, intervention in enumerate(INTERVENTIONS):
        losses, elapsed = _evaluate(
            model,
            loader,
            intervention=intervention,
            device=device,
        )
        if full is None:
            full = losses
        delta = losses - full
        intervention_results[intervention] = {
            "mean_loss": float(losses.mean()),
            "losses": [float(value) for value in losses],
            "elapsed_seconds": elapsed,
            "delta_vs_full": {
                "mean": float(delta.mean()),
                "iid_bootstrap_ci95": _iid_bootstrap(
                    delta,
                    51_000 + args.seed * 10 + index,
                ),
                "contiguous_block_8_ci95": _contiguous_block_bootstrap(
                    delta,
                    block_length=8,
                    seed=51_100 + args.seed * 10 + index,
                ),
                "contiguous_block_32_ci95": _contiguous_block_bootstrap(
                    delta,
                    block_length=32,
                    seed=51_200 + args.seed * 10 + index,
                ),
            },
        }

    mean_difference = float(full.mean() - original.mean())
    if abs(mean_difference) > 5e-4:
        raise RuntimeError(
            "Fresh full-path evaluation does not reproduce the saved endpoint: "
            f"difference={mean_difference:+.8f}"
        )
    manifest_paths = [
        dataset_path / name
        for name in (
            "mlprope_cache_manifest.json",
            ".tokenized-cache-manifest.json",
            "MANIFEST.json",
        )
        if (dataset_path / name).is_file()
    ]
    if not manifest_paths:
        raise RuntimeError(f"No dataset manifest found in {dataset_path}")
    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "analysis_kind": "trained_checkpoint_causal_intervention",
        "seed": args.seed,
        "run_dir": str(run_dir.relative_to(ROOT)),
        "weights": {
            "path": str(weights_path.relative_to(ROOT)),
            "size_bytes": weights_path.stat().st_size,
            "mtime_ns": weights_path.stat().st_mtime_ns,
        },
        "dataset": {
            "path": str(dataset_path),
            "manifests": {
                path.name: {"path": str(path), "sha256": _sha256(path)}
                for path in manifest_paths
            },
            "start_block": start_batch,
            "blocks": blocks,
        },
        "hardware": {
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(device),
            "batch_size": args.batch_size,
        },
        "full_reproduction_check": {
            "saved_mean_loss": float(original.mean()),
            "fresh_mean_loss": float(full.mean()),
            "mean_difference": mean_difference,
            "mean_absolute_per_block_difference": float(np.abs(full - original).mean()),
            "max_absolute_per_block_difference": float(np.abs(full - original).max()),
            "tolerance_on_mean_difference": 5e-4,
        },
        "interventions": intervention_results,
        "interpretation_limit": (
            "Checkpoint ablations recompute normalization and downstream states but "
            "do not distinguish endpoint necessity from training-time scaffolding."
        ),
    }
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    report = _render(payload)
    output_report.write_text(report)
    print(report)


if __name__ == "__main__":
    main()
