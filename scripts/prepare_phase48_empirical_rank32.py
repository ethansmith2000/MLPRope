#!/usr/bin/env python
"""Generate the one-shot empirical rank-32 function-step calibration."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path


REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from scripts.prepare_phase42_paper_components import _paper_base
from train_gpt import load_config


CONFIG_DIR = REPO_DIR / "sweep_configs" / "phase48_empirical_rank32"
OUTPUT_DIR = REPO_DIR / "model-output" / "position_bias_phase48_empirical_rank32"
PREFLIGHT_OUTPUT_DIR = (
    REPO_DIR / "model-output" / "position_bias_phase48_empirical_rank32_preflight"
)

# Predeclared from Phase 47's optimizer diagnostic, without using validation
# loss: sqrt(768/32) * median(r128/r32 carrier-function step at steps 1k--19k).
PHASE47_THEORETICAL_R32_MULTIPLIER = 4.898979485566356
PHASE47_POSTWARMUP_FUNCTION_RATIO = 1.2997578757740733
EMPIRICAL_R32_MULTIPLIER = (
    PHASE47_THEORETICAL_R32_MULTIPLIER * PHASE47_POSTWARMUP_FUNCTION_RATIO
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


def _arm_payload() -> dict:
    return {
        "use_rope": True,
        "qk_preprojection": {
            "enabled": True,
            "mode": "low_rank_qk_residual",
            "rank": 32,
            "gate_init": 1.0,
            "learnable_gate": True,
            "gate_sharing": "per_layer",
            "active_layers": None,
            "readout_lr_multiplier": EMPIRICAL_R32_MULTIPLIER,
            "compensate_readout_weight_decay": True,
        },
    }


def _development_base(*, run_name: str, output_root: Path) -> dict:
    payload = _paper_base(run_name=run_name, output_root=output_root)
    payload.update(
        {
            "max_train_steps": 20_000,
            "num_warmup_steps": 200,
            "checkpointing_steps": 5_000,
            "checkpoint_keep_latest": 1,
            "validate_every": 5_000,
            "wandb_group": "phase48-empirical-rank32-b32-s20000",
        }
    )
    return payload


def generate() -> list[Path]:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    preflight_dir = CONFIG_DIR / "preflight"
    preflight_dir.mkdir(parents=True, exist_ok=True)

    arm = "empirical-r32"
    run_name = f"phase48-{arm}-seed123-b32-s20000-h768d8"
    payload = _development_base(run_name=run_name, output_root=OUTPUT_DIR)
    payload.update(_arm_payload())
    path = CONFIG_DIR / f"00-{arm}.json"
    path.write_text(json.dumps(_resolve(payload), indent=2, sort_keys=True) + "\n")

    preflight_name = f"phase48-preflight-{arm}-seed123-b32-s20-h768d8"
    preflight = _development_base(
        run_name=preflight_name,
        output_root=PREFLIGHT_OUTPUT_DIR,
    )
    preflight.update(_arm_payload())
    preflight.update(
        {
            "max_train_steps": 20,
            "num_warmup_steps": 2,
            "checkpointing_steps": None,
            "checkpoint_keep_latest": None,
            "resume_from_checkpoint": None,
            "validate_every": 10_000,
            "num_validation_batches": 4,
            "num_final_validation_batches": 4,
            "final_validation_start_batch": 256,
            "save_evaluation_details": False,
            "save_final_model": False,
            "log_every_n_steps": 5,
            "wandb_group": "phase48-empirical-rank32-preflight",
        }
    )
    preflight_path = preflight_dir / f"00-{arm}.json"
    preflight_path.write_text(
        json.dumps(_resolve(preflight), indent=2, sort_keys=True) + "\n"
    )
    return [path, preflight_path]


if __name__ == "__main__":
    for config_path in generate():
        print(config_path.relative_to(REPO_DIR))
