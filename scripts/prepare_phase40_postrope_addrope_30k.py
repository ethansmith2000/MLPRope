#!/usr/bin/env python
"""Generate the Phase-40 post-RoPE AddRoPE ordering configs."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path


REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from scripts.prepare_phase39_carrier_location_30k import (
    _base_payload,
    _direct_addrope,
)
from train_gpt import load_config


CONFIG_DIR = REPO_DIR / "sweep_configs" / "phase40_postrope_addrope_30k"
OUTPUT_DIR = REPO_DIR / "model-output" / "position_bias_phase40_postrope_addrope_30k"
PREFLIGHT_OUTPUT_DIR = (
    REPO_DIR / "model-output" / "position_bias_phase40_postrope_addrope_preflight"
)
ARMS = (
    ("addrope-fixed-postrope", False),
    ("addrope-direct-postrope", True),
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


def _arm_payload(*, learned: bool) -> dict:
    qk = _direct_addrope(learned=learned)
    qk["placement"] = "after_rope"
    return {"qk": qk, "use_rope": True}


def generate() -> list[Path]:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    preflight_dir = CONFIG_DIR / "preflight"
    preflight_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for arm_index, (arm, learned) in enumerate(ARMS):
        run_name = f"phase40-{arm}-seed123-s30000-h768d8"
        payload = _base_payload(run_name=run_name, output_root=OUTPUT_DIR)
        payload.update(_arm_payload(learned=learned))
        payload["wandb_group"] = "phase40-postrope-addrope-30k"
        path = CONFIG_DIR / f"{arm_index:02d}-{arm}.json"
        path.write_text(json.dumps(_resolve(payload), indent=2, sort_keys=True) + "\n")
        written.append(path)

        preflight_name = f"phase40-preflight-{arm}-seed123-s20-h768d8"
        preflight = _base_payload(
            run_name=preflight_name,
            output_root=PREFLIGHT_OUTPUT_DIR,
        )
        preflight.update(_arm_payload(learned=learned))
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
                "wandb_group": "phase40-postrope-addrope-preflight",
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
