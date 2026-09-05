#!/usr/bin/env python
"""Generate the frozen Phase-37 direct-amplitude confirmation configs."""

from __future__ import annotations

import argparse
import copy
import json
import sys
import tempfile
from pathlib import Path


REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from train_gpt import load_config


CONFIG_DIR = REPO_DIR / "sweep_configs" / "phase37_direct_amplitude_200k"
OUTPUT_DIR = REPO_DIR / "model-output" / "position_bias_phase37_direct_amplitude_200k"
CANONICAL_DATASET = Path("/workspace/data/tokenized/openwebtext_gpt2_bs1024")

SCALAR = {
    "enabled": True,
    "mode": "tied_scalar",
    "gate_init": 1.0,
    "learnable_gate": True,
}
EXPONENTIAL_AMPLITUDE = {
    **SCALAR,
    "mode": "tied_smooth_amplitude",
    "smooth_rank": 4,
}
DIRECT_AMPLITUDE = {
    **SCALAR,
    "mode": "tied_smooth_direct_amplitude",
    "smooth_rank": 4,
}
ARMS = {
    "qkpre-scalar": SCALAR,
    "qkpre-exponential-amplitude-r4": EXPONENTIAL_AMPLITUDE,
    "qkpre-direct-amplitude-r4": DIRECT_AMPLITUDE,
}


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


def _base_payload(*, run_name: str) -> dict:
    return {
        "run_name": run_name,
        "base_output_dir": str(OUTPUT_DIR),
        "tokenized_dataset_path": str(CANONICAL_DATASET),
        "hidden_size": 768,
        "depth": 8,
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
        "position_lr_multiplier": 1.0,
        "frequency_lr_multiplier": 1.0,
        "weight_decay": 0.01,
        "exclude_position_from_decay": True,
        "lr_scheduler_type": "linear",
        "num_warmup_steps": 200,
        "max_train_steps": 200_000,
        "seed": 123,
        "paired_initialization_seed": 123,
        "beta1": 0.9,
        "beta2": 0.98,
        "max_grad_norm": 1.0,
        "qk": {"enabled": False},
        "qk_norm": True,
        "qk_norm_mode": "method_aware_rms",
        "post_position_qk_norm": False,
        "use_rope": True,
        "rope_theta": 10_000.0,
        "attn_impl": "sdpa",
        "compile": True,
        "compile_mode": "default",
        "compile_fullgraph": False,
        "mixed_precision": "bf16",
        "checkpointing_steps": 10_000,
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
        "intervention_optimizer_warmup_steps": [
            1,
            2,
            4,
            8,
            16,
            32,
            64,
            128,
            256,
            512,
        ],
        "intervention_optimizer_log_every": 1_000,
        "profile_every_n_steps": 0,
        "num_workers": 4,
        "persistent_workers": True,
        "prefetch_factor": 2,
        "non_blocking": True,
        "with_tracking": False,
        "wandb_group": "phase37-direct-amplitude-200k",
    }


def generate() -> list[Path]:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    expected = {CONFIG_DIR / f"arm-{arm}.json" for arm in ARMS}
    for stale_path in CONFIG_DIR.glob("arm-*.json"):
        if stale_path not in expected:
            stale_path.unlink()
    written = []
    for arm, preprojection in ARMS.items():
        run_name = f"phase37-{arm}-seed123-s200000-h768d8"
        payload = _base_payload(run_name=run_name)
        payload["qk_preprojection"] = copy.deepcopy(preprojection)
        config = _resolve(payload)
        path = CONFIG_DIR / f"arm-{arm}.json"
        path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
        written.append(path)
    return written


if __name__ == "__main__":
    for config_path in generate():
        print(config_path.relative_to(REPO_DIR))
