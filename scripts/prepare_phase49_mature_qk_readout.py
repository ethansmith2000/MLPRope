#!/usr/bin/env python
"""Generate the frozen 100k mature Q/K readout confirmation cohort."""

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


CONFIG_DIR = REPO_DIR / "sweep_configs" / "phase49_mature_qk_readout"
OUTPUT_DIR = REPO_DIR / "model-output" / "position_bias_phase49_mature_qk_readout"
PREFLIGHT_OUTPUT_DIR = (
    REPO_DIR / "model-output" / "position_bias_phase49_mature_qk_readout_preflight"
)
FINAL_HOLDOUT_START = 4_096
R32_READOUT_LR_MULTIPLIER = 6.367487169620489
R128_READOUT_LR_MULTIPLIER = 2.449489742783178
ARMS = (
    "rope",
    "scalar-qkpre",
    "scalar-ffnmatch-r32",
    "qk-readout-r32",
    "qk-readout-r128",
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


def _scalar_qkpre() -> dict:
    return {
        "enabled": True,
        "mode": "tied_scalar",
        "gate_init": 1.0,
        "learnable_gate": True,
        "gate_sharing": "per_layer",
        "active_layers": None,
    }


def _qk_readout(*, rank: int, readout_lr_multiplier: float) -> dict:
    return {
        "enabled": True,
        "mode": "low_rank_qk_residual",
        "rank": rank,
        "gate_init": 1.0,
        "learnable_gate": True,
        "gate_sharing": "per_layer",
        "active_layers": None,
        "readout_lr_multiplier": readout_lr_multiplier,
        "compensate_readout_weight_decay": True,
    }


def _arm_payload(arm: str) -> dict:
    if arm == "rope":
        return {"use_rope": True}
    if arm == "scalar-qkpre":
        return {"use_rope": True, "qk_preprojection": _scalar_qkpre()}
    if arm == "scalar-ffnmatch-r32":
        return {
            "use_rope": True,
            "qk_preprojection": _scalar_qkpre(),
            # Four layers * 64 extra GeGLU units * (3*d + 2) = 590,336
            # parameters, only 512 above the rank-32 readout's 589,824.
            "ff_widened_hidden_dim": 3_136,
            "ff_widened_layers": [0, 2, 4, 6],
        }
    if arm == "qk-readout-r32":
        return {
            "use_rope": True,
            "qk_preprojection": _qk_readout(
                rank=32,
                readout_lr_multiplier=R32_READOUT_LR_MULTIPLIER,
            ),
        }
    if arm == "qk-readout-r128":
        return {
            "use_rope": True,
            "qk_preprojection": _qk_readout(
                rank=128,
                readout_lr_multiplier=R128_READOUT_LR_MULTIPLIER,
            ),
        }
    raise ValueError(f"Unknown Phase-49 arm: {arm}")


def _mature_base(*, run_name: str, output_root: Path) -> dict:
    payload = _paper_base(run_name=run_name, output_root=output_root)
    payload.update(
        {
            "max_train_steps": 100_000,
            "num_warmup_steps": 200,
            "checkpointing_steps": 5_000,
            "checkpoint_keep_latest": 1,
            "validate_every": 5_000,
            "final_validation_start_batch": FINAL_HOLDOUT_START,
            "wandb_group": "phase49-mature-qk-readout-b32-s100000",
        }
    )
    return payload


def generate() -> list[Path]:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    preflight_dir = CONFIG_DIR / "preflight"
    preflight_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for arm_index, arm in enumerate(ARMS):
        run_name = f"phase49-{arm}-seed123-b32-s100000-h768d8"
        payload = _mature_base(run_name=run_name, output_root=OUTPUT_DIR)
        payload.update(_arm_payload(arm))
        path = CONFIG_DIR / f"{arm_index:02d}-{arm}.json"
        path.write_text(json.dumps(_resolve(payload), indent=2, sort_keys=True) + "\n")
        written.append(path)

        preflight_name = f"phase49-preflight-{arm}-seed123-b32-s20-h768d8"
        preflight = _mature_base(
            run_name=preflight_name,
            output_root=PREFLIGHT_OUTPUT_DIR,
        )
        preflight.update(_arm_payload(arm))
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
                "final_validation_start_batch": 512,
                "save_evaluation_details": False,
                "save_final_model": False,
                "log_every_n_steps": 5,
                "wandb_group": "phase49-mature-qk-readout-preflight",
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
