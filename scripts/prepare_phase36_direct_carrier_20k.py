#!/usr/bin/env python
"""Generate Phase-36 direct carrier calibration and 20k screen configs."""

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


CONFIG_DIR = REPO_DIR / "sweep_configs" / "phase36_direct_carrier_20k"
LONG_OUTPUT_DIR = (
    REPO_DIR / "model-output" / "position_bias_phase36_direct_carrier_20k"
)
CALIBRATION_OUTPUT_DIR = (
    REPO_DIR / "model-output" / "position_bias_phase36_direct_carrier_calibration"
)
CANONICAL_DATASET = Path("/workspace/data/tokenized/openwebtext_gpt2_bs1024")


SCALAR = {
    "enabled": True,
    "mode": "tied_scalar",
    "gate_init": 1.0,
    "learnable_gate": True,
}
DIRECT_AMPLITUDE = {
    **SCALAR,
    "mode": "tied_smooth_direct_amplitude",
    "smooth_rank": 4,
}
GLOBAL_FREQUENCY = {
    **SCALAR,
    "frequency": {
        "mode": "learned_global_direct",
        "reference_length": 1024,
        "endpoint_phase_scale": 1.0,
        "smooth_rank": 4,
        "max_grad_norm": 1.0,
    },
}
HYBRID_FREQUENCY = {
    **SCALAR,
    "frequency": {
        "mode": "learned_hybrid_direct",
        "reference_length": 1024,
        "endpoint_phase_scale": 1.0,
        "smooth_rank": 4,
        "max_grad_norm": 1.0,
    },
}
DIRECT_AMPLITUDE_HYBRID_FREQUENCY = {
    **DIRECT_AMPLITUDE,
    "frequency": copy.deepcopy(HYBRID_FREQUENCY["frequency"]),
}


MAIN_ARMS = {
    "rope-fixed": {"qk_preprojection": {"enabled": False}},
    "qkpre-scalar": {"qk_preprojection": SCALAR},
    "qkpre-direct-amplitude-r4": {"qk_preprojection": DIRECT_AMPLITUDE},
    "qkpre-global-frequency": {"qk_preprojection": GLOBAL_FREQUENCY},
    "qkpre-hybrid-frequency-r4": {"qk_preprojection": HYBRID_FREQUENCY},
    "qkpre-direct-amplitude-hybrid-frequency-r4": {
        "qk_preprojection": DIRECT_AMPLITUDE_HYBRID_FREQUENCY
    },
}


def calibration_arms() -> dict[str, dict]:
    arms = {
        "direct-amplitude-r4-lr1": {
            "qk_preprojection": DIRECT_AMPLITUDE,
            "position_lr_multiplier": 1.0,
        },
        "direct-amplitude-hybrid-frequency-r4-lr1": {
            "qk_preprojection": DIRECT_AMPLITUDE_HYBRID_FREQUENCY,
            "frequency_lr_multiplier": 1.0,
        },
    }
    for mode, preprojection in (
        ("global-frequency", GLOBAL_FREQUENCY),
        ("hybrid-frequency-r4", HYBRID_FREQUENCY),
    ):
        for label, multiplier in (("025", 0.25), ("1", 1.0), ("4", 4.0)):
            arms[f"{mode}-lr{label}"] = {
                "qk_preprojection": preprojection,
                "frequency_lr_multiplier": multiplier,
            }
    return arms


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
        "position_lr_multiplier": 1.0,
        "frequency_lr_multiplier": 1.0,
        "weight_decay": 0.01,
        "exclude_position_from_decay": True,
        "lr_scheduler_type": "linear",
        "num_warmup_steps": 200,
        "max_train_steps": 20_000,
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
        "checkpointing_steps": None,
        "checkpoint_keep_latest": None,
        "checkpoint_milestones": [],
        "resume_from_checkpoint": None,
        "validate_every": 2_000,
        "num_validation_batches": 128,
        "validation_start_batch": 0,
        "num_final_validation_batches": 1_024,
        "final_validation_start_batch": 2_048,
        "save_evaluation_details": True,
        "save_final_model": False,
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
        "wandb_group": "phase36-direct-carrier-20k",
    }


def generate() -> list[Path]:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    calibration_dir = CONFIG_DIR / "calibration"
    calibration_dir.mkdir(parents=True, exist_ok=True)
    written = []

    for arm, arm_payload in MAIN_ARMS.items():
        run_name = f"phase36-{arm}-seed123-s20000-h768d8"
        payload = _base_payload(run_name=run_name, output_dir=LONG_OUTPUT_DIR)
        payload.update(copy.deepcopy(arm_payload))
        config = _resolve(payload)
        path = CONFIG_DIR / f"arm-{arm}.json"
        path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
        written.append(path)

    for arm, arm_payload in calibration_arms().items():
        run_name = f"phase36-calibration-{arm}-seed123-s512-h768d8"
        payload = _base_payload(
            run_name=run_name,
            output_dir=CALIBRATION_OUTPUT_DIR,
        )
        payload.update(copy.deepcopy(arm_payload))
        payload.update(
            {
                "max_train_steps": 512,
                "validate_every": 10_000,
                "num_validation_batches": 4,
                "num_final_validation_batches": 4,
                "final_validation_start_batch": 256,
                "save_evaluation_details": False,
                "intervention_optimizer_log_every": None,
                "wandb_group": "phase36-direct-carrier-calibration",
            }
        )
        config = _resolve(payload)
        path = calibration_dir / f"arm-{arm}.json"
        path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
        written.append(path)
    return written


if __name__ == "__main__":
    for config_path in generate():
        print(config_path.relative_to(REPO_DIR))
