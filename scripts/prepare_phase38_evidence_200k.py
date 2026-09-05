#!/usr/bin/env python
"""Generate the frozen Phase-38 evidence-strengthening configs."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path


REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from train_gpt import load_config


CONFIG_DIR = REPO_DIR / "sweep_configs" / "phase38_evidence_200k"
OUTPUT_DIR = REPO_DIR / "model-output" / "position_bias_phase38_evidence_200k"
PREFLIGHT_OUTPUT_DIR = (
    REPO_DIR / "model-output" / "position_bias_phase38_evidence_preflight"
)
DATASET = Path("/workspace/data/tokenized/openwebtext_gpt2_bs1024")

# Priority is launch order: put the long scale pair on GPUs first.
PAIRS = (
    {
        "key": "scale-h1024d12-seed123",
        "hidden_size": 1024,
        "depth": 12,
        "seed": 123,
        "qk_norm": True,
    },
    {
        "key": "rep-h768d8-seed456",
        "hidden_size": 768,
        "depth": 8,
        "seed": 456,
        "qk_norm": True,
    },
    {
        "key": "rep-h768d8-seed789",
        "hidden_size": 768,
        "depth": 8,
        "seed": 789,
        "qk_norm": True,
    },
    {
        "key": "noqknorm-h768d8-seed123",
        "hidden_size": 768,
        "depth": 8,
        "seed": 123,
        "qk_norm": False,
    },
)


def _cli(path: str) -> argparse.Namespace:
    return argparse.Namespace(
        override_json=path,
        pos_variant=None,
        attn_impl=None,
        max_train_steps=None,
        dry_run=False,
        print_model=False,
    )


def _resolve(payload: dict) -> dict:
    with tempfile.NamedTemporaryFile("w", suffix=".json") as handle:
        json.dump(payload, handle)
        handle.flush()
        return vars(load_config(_cli(handle.name)))


def _base_payload(pair: dict, *, run_name: str, output_root: Path) -> dict:
    return {
        "run_name": run_name,
        "base_output_dir": str(output_root),
        "tokenized_dataset_path": str(DATASET),
        "hidden_size": pair["hidden_size"],
        "depth": pair["depth"],
        "n_head": 8,
        "ff_mult": 4,
        "training_length": 1024,
        "model_position_extent": 1024,
        "evaluation_lengths": [1024],
        "scalar_normalization_extent": 1024,
        "per_device_train_batch_size": 8,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "gradient_checkpointing": False,
        "learning_rate": 3.0e-4,
        "weight_decay": 0.01,
        "lr_scheduler_type": "linear",
        "num_warmup_steps": 200,
        "max_train_steps": 200_000,
        "seed": pair["seed"],
        "paired_initialization_seed": pair["seed"],
        "beta1": 0.9,
        "beta2": 0.98,
        "max_grad_norm": 1.0,
        "qk": {"enabled": False},
        "qk_norm": pair["qk_norm"],
        "qk_norm_mode": "method_aware_rms",
        "post_position_qk_norm": False,
        "use_rope": True,
        "rope_theta": 10_000.0,
        "attn_impl": "sdpa",
        "compile": True,
        "compile_mode": "default",
        "compile_fullgraph": False,
        "mixed_precision": "bf16",
        "checkpointing_steps": 5_000,
        "checkpoint_keep_latest": 1,
        "checkpoint_milestones": [],
        "resume_from_checkpoint": "auto",
        "validate_every": 10_000,
        "num_validation_batches": 128,
        "validation_start_batch": 0,
        "num_final_validation_batches": 1_024,
        "final_validation_start_batch": 2_048,
        "save_evaluation_details": True,
        "save_final_model": True,
        "log_every_n_steps": 50,
        "profile_every_n_steps": 0,
        "num_workers": 4,
        "persistent_workers": True,
        "prefetch_factor": 2,
        "non_blocking": True,
        "with_tracking": False,
        "exclude_position_from_decay": False,
        "position_lr_multiplier": 1.0,
        "wandb_group": "phase38-evidence-200k",
    }


def _arm_payload(candidate: bool) -> dict:
    return {
        "qk_preprojection": {
            "enabled": candidate,
            "mode": "tied_scalar",
            "gate_init": 1.0,
            "learnable_gate": True,
        }
    }


def generate() -> list[Path]:
    if not DATASET.is_dir():
        raise FileNotFoundError(f"Canonical dataset is missing: {DATASET}")
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    preflight_dir = CONFIG_DIR / "preflight"
    preflight_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for pair_index, pair in enumerate(PAIRS):
        for arm_index, (arm, candidate) in enumerate(
            (("rope", False), ("qkpre", True))
        ):
            run_name = f"phase38-{pair['key']}-{arm}-s200000"
            payload = _base_payload(pair, run_name=run_name, output_root=OUTPUT_DIR)
            payload.update(_arm_payload(candidate))
            resolved = _resolve(payload)
            path = CONFIG_DIR / f"{pair_index:02d}{arm_index}-{run_name}.json"
            path.write_text(json.dumps(resolved, indent=2, sort_keys=True) + "\n")
            written.append(path)

            # One short operational check per unique structure. Replication
            # pairs share the already-tested h768/QKNorm structure.
            if pair["key"].startswith(("scale-", "noqknorm-")):
                preflight_name = f"phase38-preflight-{pair['key']}-{arm}-s20"
                preflight = _base_payload(
                    pair,
                    run_name=preflight_name,
                    output_root=PREFLIGHT_OUTPUT_DIR,
                )
                preflight.update(_arm_payload(candidate))
                preflight.update(
                    {
                        "max_train_steps": 20,
                        "checkpointing_steps": None,
                        "checkpoint_keep_latest": None,
                        "resume_from_checkpoint": None,
                        "validate_every": 10_000,
                        "num_validation_batches": 4,
                        "num_final_validation_batches": 4,
                        "final_validation_start_batch": 256,
                        "save_evaluation_details": False,
                        "save_final_model": False,
                        "wandb_group": "phase38-evidence-preflight",
                    }
                )
                preflight_resolved = _resolve(preflight)
                preflight_path = preflight_dir / f"{preflight_name}.json"
                preflight_path.write_text(
                    json.dumps(preflight_resolved, indent=2, sort_keys=True) + "\n"
                )
                written.append(preflight_path)
    return written


if __name__ == "__main__":
    for config_path in generate():
        print(config_path.relative_to(REPO_DIR))
