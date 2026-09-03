#!/usr/bin/env python
"""Generate the frozen phase-33 long-run and isolated preflight configs."""

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


CONFIG_DIR = REPO_DIR / "sweep_configs" / "phase33_static_qkpre_200k"
LONG_OUTPUT_DIR = REPO_DIR / "model-output" / "position_bias_phase33_static_qkpre_200k"
PREFLIGHT_OUTPUT_DIR = (
    REPO_DIR / "model-output" / "position_bias_phase33_static_qkpre_preflight"
)
CANONICAL_DATASET = Path(
    "/workspace/data/tokenized/openwebtext_gpt2_bs1024"
)


ARMS = {
    "rope-fixed": {"use_rope": True, "qk_preprojection": {"enabled": False}},
    "qkpre-tied-nope": {
        "use_rope": False,
        "qk_preprojection": {"enabled": True, "mode": "tied_scalar"},
    },
    "qkpre-tied-rope": {
        "use_rope": True,
        "qk_preprojection": {"enabled": True, "mode": "tied_scalar"},
    },
    "qkpre-split-scalar-rope": {
        "use_rope": True,
        "qk_preprojection": {"enabled": True, "mode": "split_scalar"},
    },
    "qkpre-pair-amplitude-rope": {
        "use_rope": True,
        "qk_preprojection": {
            "enabled": True,
            "mode": "split_pair_amplitude",
        },
    },
    "qkpre-pair-polar-rope": {
        "use_rope": True,
        "qk_preprojection": {"enabled": True, "mode": "split_pair_polar"},
    },
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


def _base_payload(*, run_name: str, output_dir: Path) -> dict:
    return {
        "run_name": run_name,
        "base_output_dir": str(output_dir),
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
        "weight_decay": 0.01,
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
        "wandb_group": "phase33-static-qkpre-200k",
    }


def generate() -> list[Path]:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    preflight_dir = CONFIG_DIR / "preflight"
    preflight_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for arm, arm_payload in ARMS.items():
        long_name = f"phase33-{arm}-seed123-s200000-h768d8"
        long_payload = _base_payload(
            run_name=long_name,
            output_dir=LONG_OUTPUT_DIR,
        )
        long_payload.update(arm_payload)
        long_config = _resolve(long_payload)
        long_path = CONFIG_DIR / f"{long_name}.json"
        long_path.write_text(json.dumps(long_config, indent=2, sort_keys=True) + "\n")
        written.append(long_path)

        preflight_name = f"phase33-preflight-{arm}-seed123-s50-h768d8"
        preflight_payload = _base_payload(
            run_name=preflight_name,
            output_dir=PREFLIGHT_OUTPUT_DIR,
        )
        preflight_payload.update(arm_payload)
        preflight_payload.update(
            {
                "max_train_steps": 50,
                "checkpointing_steps": None,
                "checkpoint_keep_latest": None,
                "checkpoint_milestones": [],
                "resume_from_checkpoint": None,
                "validate_every": 10_000,
                "num_validation_batches": 4,
                "num_final_validation_batches": 4,
                "final_validation_start_batch": 256,
                "save_evaluation_details": False,
                "save_final_model": False,
                "wandb_group": "phase33-static-qkpre-preflight",
            }
        )
        preflight_config = _resolve(preflight_payload)
        preflight_path = preflight_dir / f"{preflight_name}.json"
        preflight_path.write_text(
            json.dumps(preflight_config, indent=2, sort_keys=True) + "\n"
        )
        written.append(preflight_path)
    return written


if __name__ == "__main__":
    for config_path in generate():
        print(config_path.relative_to(REPO_DIR))
