#!/usr/bin/env python
"""Evaluate retained canonical-LR checkpoints on the Phase-55 holdout."""

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
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import default_data_collator


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train_gpt import make_model


PHASE49_ROOT = ROOT / "model-output" / "position_bias_phase49_mature_qk_readout"
RESULT_ROOT = ROOT / "results" / "phase55_lr_robustness"
HOLDOUT_START = 6_144
HOLDOUT_BLOCKS = 1_024
RUNS = {
    "rope": PHASE49_ROOT / "phase49-rope-seed123-b32-s100000-h768d8",
    "scalar-qkpre": PHASE49_ROOT / "phase49-scalar-qkpre-seed123-b32-s100000-h768d8",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@torch.inference_mode()
def _evaluate(model, loader) -> tuple[list[float], float]:
    losses = []
    started = time.perf_counter()
    for batch in loader:
        tokens = batch["input_ids"].to("cuda", non_blocking=True)
        inputs = tokens[:, :-1]
        targets = tokens[:, 1:]
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(input_ids=inputs)
        token_losses = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]).float(),
            targets.reshape(-1),
            reduction="none",
        ).reshape_as(targets)
        losses.extend(token_losses.double().mean(dim=-1).cpu().tolist())
    torch.cuda.synchronize()
    return losses, time.perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, choices=tuple(RUNS))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Phase-55 reference evaluation requires CUDA")
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    output_path = RESULT_ROOT / f"reference_lr3e4_{args.arm}.json"
    if output_path.is_file():
        print(f"Already complete: {output_path}")
        return

    run_dir = RUNS[args.arm]
    config_path = run_dir / "training_config.json"
    weights_path = run_dir / "pytorch_model.bin"
    config = json.loads(config_path.read_text())
    state = torch.load(weights_path, map_location="cpu", weights_only=True, mmap=True)
    vocab_size = int(state["token_embedding.weight"].shape[0])
    model = make_model(SimpleNamespace(**config), vocab_size)
    model.load_state_dict(state, strict=True)
    del state
    model.requires_grad_(False).eval().to("cuda")

    validation = datasets.load_from_disk(config["tokenized_dataset_path"])["validation"]
    evaluation = validation.select(range(HOLDOUT_START, HOLDOUT_START + HOLDOUT_BLOCKS))
    loader = DataLoader(
        evaluation,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=default_data_collator,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=bool(args.num_workers),
    )
    losses, elapsed = _evaluate(model, loader)
    if len(losses) != HOLDOUT_BLOCKS or not all(math.isfinite(x) for x in losses):
        raise ValueError("Invalid Phase-55 reference losses")
    payload = {
        "scope": "phase55_lr_reference",
        "arm": args.arm,
        "learning_rate": 3e-4,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "source_run": str(run_dir.relative_to(ROOT)),
        "source_config_sha256": _sha256(config_path),
        "source_weights_sha256": _sha256(weights_path),
        "analysis_script_sha256": _sha256(Path(__file__)),
        "evaluation_start_batch": HOLDOUT_START,
        "evaluation_blocks": HOLDOUT_BLOCKS,
        "context_length": 1_024,
        "mean_loss": sum(losses) / len(losses),
        "losses": losses,
        "elapsed_seconds": elapsed,
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "gpu": torch.cuda.get_device_name(0),
        "artifact_policy": {"source_weights_read_only": True, "new_weights": False},
    }
    temporary = output_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(output_path)
    print(f"{args.arm}: NLL={payload['mean_loss']:.6f} elapsed={elapsed:.1f}s")


if __name__ == "__main__":
    main()
