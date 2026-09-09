#!/usr/bin/env python
"""Generate dimension-calibrated rank-32/rank-128 Q/K readout runs."""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
from pathlib import Path


REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from scripts.prepare_phase42_paper_components import _paper_base
from train_gpt import load_config


CONFIG_DIR = REPO_DIR / "sweep_configs" / "phase47_rank_calibration"
OUTPUT_DIR = REPO_DIR / "model-output" / "position_bias_phase47_rank_calibration"
PREFLIGHT_OUTPUT_DIR = (
    REPO_DIR / "model-output" / "position_bias_phase47_rank_calibration_preflight"
)
MODEL_DIM = 768
ARMS = (("calibrated-r32", 32), ("calibrated-r128", 128))


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


def _arm_payload(rank: int) -> dict:
    return {
        "use_rope": True,
        "qk_preprojection": {
            "enabled": True,
            "mode": "low_rank_qk_residual",
            "rank": rank,
            "gate_init": 1.0,
            "learnable_gate": True,
            "gate_sharing": "per_layer",
            "active_layers": None,
            "readout_lr_multiplier": math.sqrt(MODEL_DIM / rank),
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
            "wandb_group": "phase47-rank-calibration-b32-s20000",
        }
    )
    return payload


def generate() -> list[Path]:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    preflight_dir = CONFIG_DIR / "preflight"
    preflight_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for arm_index, (arm, rank) in enumerate(ARMS):
        run_name = f"phase47-{arm}-seed123-b32-s20000-h768d8"
        payload = _development_base(run_name=run_name, output_root=OUTPUT_DIR)
        payload.update(_arm_payload(rank))
        path = CONFIG_DIR / f"{arm_index:02d}-{arm}.json"
        path.write_text(json.dumps(_resolve(payload), indent=2, sort_keys=True) + "\n")
        written.append(path)

        preflight_name = f"phase47-preflight-{arm}-seed123-b32-s20-h768d8"
        preflight = _development_base(
            run_name=preflight_name,
            output_root=PREFLIGHT_OUTPUT_DIR,
        )
        preflight.update(_arm_payload(rank))
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
                "wandb_group": "phase47-rank-calibration-preflight",
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
