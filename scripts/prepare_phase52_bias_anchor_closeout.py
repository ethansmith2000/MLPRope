#!/usr/bin/env python
"""Generate the frozen Phase-52 optimization-scaffold close-out."""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.prepare_phase42_paper_components import _paper_base
from scripts.prepare_phase49_mature_qk_readout import (
    R32_READOUT_LR_MULTIPLIER,
    _resolve,
    _scalar_qkpre,
)


CONFIG_ROOT = ROOT / "sweep_configs" / "phase52_bias_anchor_closeout"
OUTPUT_ROOT = ROOT / "model-output" / "position_bias_phase52_bias_anchor_closeout"
PREFLIGHT_OUTPUT_ROOT = ROOT / "model-output" / "position_bias_phase52_bias_anchor_closeout_preflight"
FINAL_HOLDOUT_START = 5_120
ARMS = ("scalar-qk-bias", "qk-readout-r32-no-anchor")


def _arm_payload(arm: str) -> dict:
    if arm == "scalar-qk-bias":
        return {
            "use_rope": True,
            "qk_preprojection": _scalar_qkpre(),
            "qk_projection_bias": True,
        }
    if arm == "qk-readout-r32-no-anchor":
        return {
            "use_rope": True,
            "qk_preprojection": {
                "enabled": True,
                "mode": "low_rank_qk_replace",
                "rank": 32,
                "readout_lr_multiplier": R32_READOUT_LR_MULTIPLIER,
                "compensate_readout_weight_decay": True,
            },
            "qk_projection_bias": False,
        }
    raise ValueError(f"Unknown Phase-52 arm: {arm}")


def _base(*, run_name: str, output_root: Path) -> dict:
    payload = _paper_base(run_name=run_name, output_root=output_root)
    payload.update(
        {
            "seed": 123,
            "paired_initialization_seed": 123,
            "max_train_steps": 100_000,
            "num_warmup_steps": 200,
            "checkpointing_steps": None,
            "checkpoint_keep_latest": 1,
            "checkpoint_milestones": [],
            "resume_from_checkpoint": None,
            "save_final_model": False,
            "save_evaluation_details": True,
            "validate_every": 5_000,
            "num_final_validation_batches": 1_024,
            "final_validation_start_batch": FINAL_HOLDOUT_START,
            "wandb_group": "phase52-bias-anchor-closeout-b32-s100000",
        }
    )
    return payload


def generate() -> list[Path]:
    CONFIG_ROOT.mkdir(parents=True, exist_ok=True)
    preflight_root = CONFIG_ROOT / "preflight"
    preflight_root.mkdir(parents=True, exist_ok=True)
    written = []
    for index, arm in enumerate(ARMS):
        run_name = f"phase52-{arm}-seed123-b32-s100000-h768d8"
        payload = _base(run_name=run_name, output_root=OUTPUT_ROOT)
        payload.update(_arm_payload(arm))
        path = CONFIG_ROOT / f"{index:02d}-{arm}.json"
        path.write_text(json.dumps(_resolve(payload), indent=2, sort_keys=True) + "\n")
        written.append(path)

        preflight_name = f"phase52-preflight-{arm}-seed123-b32-s20-h768d8"
        preflight = _base(run_name=preflight_name, output_root=PREFLIGHT_OUTPUT_ROOT)
        preflight.update(_arm_payload(arm))
        preflight.update(
            {
                "max_train_steps": 20,
                "num_warmup_steps": 2,
                "validate_every": 10_000,
                "num_validation_batches": 4,
                "num_final_validation_batches": 4,
                "final_validation_start_batch": 512,
                "save_evaluation_details": False,
                "log_every_n_steps": 5,
                "wandb_group": "phase52-bias-anchor-closeout-preflight",
            }
        )
        preflight_path = preflight_root / f"{index:02d}-{arm}.json"
        preflight_path.write_text(
            json.dumps(_resolve(preflight), indent=2, sort_keys=True) + "\n"
        )
        written.append(preflight_path)
    return written


if __name__ == "__main__":
    for config_path in generate():
        print(config_path.relative_to(ROOT))
