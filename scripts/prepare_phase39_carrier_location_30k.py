#!/usr/bin/env python
"""Generate the frozen Phase-39A carrier-location comparison configs."""

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


CONFIG_DIR = REPO_DIR / "sweep_configs" / "phase39_carrier_location_30k"
OUTPUT_DIR = REPO_DIR / "model-output" / "position_bias_phase39_carrier_location_30k"
PREFLIGHT_OUTPUT_DIR = (
    REPO_DIR / "model-output" / "position_bias_phase39_carrier_location_preflight"
)
DATASET = Path("/workspace/data/tokenized/openwebtext_gpt2_bs1024")

# The order is also the preferred queue order. The common RoPE reference and
# promoted method start first, followed by the placement and AddRoPE controls.
ARMS = (
    "rope-fixed",
    "qkpre-rope",
    "input-rope",
    "addrope-fixed-rope",
    "addrope-direct-nope",
    "addrope-direct-rope",
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


def _base_payload(*, run_name: str, output_root: Path) -> dict:
    return {
        "run_name": run_name,
        "base_output_dir": str(output_root),
        "tokenized_dataset_path": str(DATASET),
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
        "max_train_steps": 30_000,
        "seed": 123,
        "paired_initialization_seed": 123,
        "beta1": 0.9,
        "beta2": 0.98,
        "max_grad_norm": 1.0,
        "input_sinusoid": {"enabled": False},
        "qk": {"enabled": False},
        "qk_preprojection": {"enabled": False},
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
        "validate_every": 5_000,
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
        "wandb_group": "phase39-carrier-location-30k",
    }


def _direct_addrope(*, learned: bool) -> dict:
    """Canonical head-space carrier, without a feature mapper.

    Direct parameters have shape ``[heads, head_dim / 2]`` in each layer.
    Separate Q/K tensors make the actual granularity
    ``[layer, branch, head, frequency_pair]``. The canonical frequencies stay
    frozen; raw amplitude and phase parameters both start at zero, yielding
    amplitude one and phase zero exactly.
    """
    return {
        "enabled": True,
        "application": "additive",
        "geometry": "amplitude_phase",
        "input": {
            "kind": "frozen_fourier",
            "basis_dim": 96,
            "theta": None,
            "scalars": [],
        },
        "mapper": {
            "kind": "identity",
            "residual": False,
        },
        "output": {
            "parameter_source": "direct",
            "amplitude_init": 1.0,
            "amplitude_parameterization": "signed",
            "learn_amplitude": learned,
            "learn_phase": learned,
            "phase_scale": 1.0,
            "additive_normalization": "none",
        },
        "conditioning": {"kind": "none"},
        "qk_coupling": (
            "shared_trunk_separate_readouts" if learned else "shared"
        ),
        "head_coupling": "per_head_independent",
    }


def _arm_payload(arm: str) -> dict:
    if arm == "rope-fixed":
        return {}
    if arm == "input-rope":
        return {
            "input_sinusoid": {
                "enabled": True,
                "gate_init": 1.0,
                "learnable_gate": True,
            }
        }
    if arm == "qkpre-rope":
        return {
            "qk_preprojection": {
                "enabled": True,
                "mode": "tied_scalar",
                "gate_init": 1.0,
                "learnable_gate": True,
            }
        }
    if arm == "addrope-fixed-rope":
        return {"qk": _direct_addrope(learned=False)}
    if arm in {"addrope-direct-nope", "addrope-direct-rope"}:
        return {
            "qk": _direct_addrope(learned=True),
            "use_rope": arm.endswith("-rope"),
        }
    raise ValueError(f"Unknown Phase-39 arm: {arm}")


def generate() -> list[Path]:
    if not DATASET.is_dir():
        raise FileNotFoundError(f"Canonical dataset is missing: {DATASET}")
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    preflight_dir = CONFIG_DIR / "preflight"
    preflight_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for arm_index, arm in enumerate(ARMS):
        run_name = f"phase39-{arm}-seed123-s30000-h768d8"
        payload = _base_payload(run_name=run_name, output_root=OUTPUT_DIR)
        payload.update(_arm_payload(arm))
        path = CONFIG_DIR / f"{arm_index:02d}-{arm}.json"
        path.write_text(json.dumps(_resolve(payload), indent=2, sort_keys=True) + "\n")
        written.append(path)

        preflight_name = f"phase39-preflight-{arm}-seed123-s20-h768d8"
        preflight = _base_payload(
            run_name=preflight_name,
            output_root=PREFLIGHT_OUTPUT_DIR,
        )
        preflight.update(_arm_payload(arm))
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
                "wandb_group": "phase39-carrier-location-preflight",
            }
        )
        preflight_path = preflight_dir / f"{arm_index:02d}-{arm}.json"
        preflight_path.write_text(
            json.dumps(_resolve(preflight), indent=2, sort_keys=True) + "\n"
        )
        written.append(preflight_path)
    return written


if __name__ == "__main__":
    for config_path in generate():
        print(config_path.relative_to(REPO_DIR))
