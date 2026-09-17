#!/usr/bin/env python
"""Evaluate retained carrier checkpoints under common sinusoid-origin shifts."""

from __future__ import annotations

import argparse
import hashlib
import json
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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from position.basis import interleaved_fourier_basis
from scripts.evaluate_phase53_paper_mechanism import (
    HOLDOUT_START,
    LOSS_BLOCKS,
    _run_dir,
)
from train_gpt import make_model


RESULT_ROOT = ROOT / "results" / "phase54_position_origin"
ARMS = ("scalar-qkpre", "qk-readout-r32")
SEEDS = (123, 456, 789)
OFFSETS = (0, 1, 4, 16, 64, 256, 1_024, 4_096)
EVAL_BATCH_SIZE = 8


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _carrier_modules(model) -> list:
    modules = []
    for block in model.blocks:
        module = block.attn.qk_preprojection
        if module is None:
            raise ValueError("Phase 54 requires a pre-Q/K carrier checkpoint")
        modules.append(module)
    return modules


def _shifted_basis(module, offset: int) -> torch.Tensor:
    basis = module.basis
    shifted = interleaved_fourier_basis(
        basis.extent + offset,
        basis.basis_dim,
        basis.theta,
    )[offset : offset + basis.extent]
    return shifted.to(device=basis.basis.device, dtype=torch.float32)


@torch.inference_mode()
def _evaluate_offset(model, loader, modules: list, offset: int) -> np.ndarray:
    for module in modules:
        module.basis.basis = _shifted_basis(module, offset)
    losses = []
    for batch in loader:
        tokens = batch["input_ids"].to("cuda", non_blocking=True)
        inputs = tokens[:, :-1]
        targets = tokens[:, 1:]
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(input_ids=inputs)
        token_loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]).float(),
            targets.reshape(-1),
            reduction="none",
        ).reshape_as(targets)
        losses.extend(token_loss.double().mean(dim=1).cpu().tolist())
    if len(losses) != LOSS_BLOCKS:
        raise ValueError(f"Expected {LOSS_BLOCKS} blocks, got {len(losses)}")
    result = np.asarray(losses, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError(f"Non-finite loss at offset {offset}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True, choices=SEEDS)
    parser.add_argument("--arm", required=True, choices=ARMS)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Phase 54 requires a visible CUDA GPU")

    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    stem = RESULT_ROOT / f"seed{args.seed}_{args.arm}"
    output_json = stem.with_suffix(".json")
    output_npz = stem.with_suffix(".npz")
    if output_json.is_file() and output_npz.is_file() and not args.overwrite:
        print(f"Already complete: seed{args.seed}_{args.arm}")
        return

    run_dir = _run_dir(args.seed, args.arm)
    config_path = run_dir / "training_config.json"
    weights_path = run_dir / "pytorch_model.bin"
    if not (run_dir / "COMPLETED").is_file():
        raise RuntimeError(f"Incomplete source run: {run_dir}")
    config = json.loads(config_path.read_text())
    state = torch.load(weights_path, map_location="cpu", weights_only=True, mmap=True)
    vocab_size = int(state["token_embedding.weight"].shape[0])
    model = make_model(SimpleNamespace(**config), vocab_size)
    model.load_state_dict(state, strict=True)
    del state
    model.requires_grad_(False).eval().to("cuda")
    modules = _carrier_modules(model)

    validation = datasets.load_from_disk(config["tokenized_dataset_path"])["validation"]
    evaluation = validation.select(range(HOLDOUT_START, HOLDOUT_START + LOSS_BLOCKS))
    loader = DataLoader(
        evaluation,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        collate_fn=default_data_collator,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=bool(args.num_workers),
    )

    started = time.perf_counter()
    loss_matrix = np.stack(
        [_evaluate_offset(model, loader, modules, offset) for offset in OFFSETS]
    )
    torch.cuda.synchronize()
    reference_basis = interleaved_fourier_basis(
        config["model_position_extent"], config["hidden_size"],
        float(config["qk_preprojection"]["theta"]),
    )
    carrier_cosines = []
    for offset in OFFSETS:
        shifted = interleaved_fourier_basis(
            config["model_position_extent"] + offset,
            config["hidden_size"],
            float(config["qk_preprojection"]["theta"]),
        )[offset : offset + config["model_position_extent"]]
        carrier_cosines.append(float(F.cosine_similarity(reference_basis, shifted).mean()))

    temporary_npz = output_npz.with_suffix(".npz.tmp")
    with temporary_npz.open("wb") as handle:
        np.savez_compressed(
            handle,
            offsets=np.asarray(OFFSETS, dtype=np.int64),
            per_block_mean_loss=loss_matrix,
        )
    temporary_npz.replace(output_npz)
    payload = {
        "scope": "phase54_position_origin",
        "seed": args.seed,
        "arm": args.arm,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "source_run": str(run_dir.relative_to(ROOT)),
        "source_config_sha256": _sha256(config_path),
        "source_weights_sha256": _sha256(weights_path),
        "analysis_script_sha256": _sha256(Path(__file__)),
        "array_file": output_npz.name,
        "array_file_sha256": _sha256(output_npz),
        "holdout": {"start_batch": HOLDOUT_START, "blocks": LOSS_BLOCKS},
        "offsets": list(OFFSETS),
        "carrier_cosines": carrier_cosines,
        "mean_loss": [float(row.mean()) for row in loss_matrix],
        "elapsed_seconds": time.perf_counter() - started,
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "gpu": torch.cuda.get_device_name(0),
        "artifact_policy": {
            "source_weights_read_only": True,
            "new_weights_or_checkpoints": False,
            "compressed_per_block_arrays": True,
        },
    }
    temporary_json = output_json.with_suffix(".json.tmp")
    temporary_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary_json.replace(output_json)
    print(
        f"seed{args.seed}_{args.arm}: "
        + " ".join(
            f"c={offset}:{loss:.6f}"
            for offset, loss in zip(OFFSETS, payload["mean_loss"], strict=True)
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
