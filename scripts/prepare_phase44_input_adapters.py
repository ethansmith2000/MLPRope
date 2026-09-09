#!/usr/bin/env python
"""Generate the matched Phase-44 input-sinusoid adapter cohort."""

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


CONFIG_DIR = REPO_DIR / "sweep_configs" / "phase44_input_adapters"
OUTPUT_DIR = REPO_DIR / "model-output" / "position_bias_phase44_input_adapters"
PREFLIGHT_OUTPUT_DIR = (
    REPO_DIR / "model-output" / "position_bias_phase44_input_adapters_preflight"
)
ARMS = (
    "lowrank-linear-r32",
    "per-pair-amplitude",
    "input-scalar-control",
)
NOVEL_ARMS = ARMS[:2]


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


def _input(mode: str, *, rank: int = 32) -> dict:
    return {
        "enabled": True,
        "mode": mode,
        "rank": rank,
        "gate_init": 1.0,
        "learnable_gate": True,
    }


def _arm_payload(arm: str) -> dict:
    if arm == "lowrank-linear-r32":
        return {
            "use_rope": True,
            "input_sinusoid": _input("low_rank_linear_residual"),
        }
    if arm == "per-pair-amplitude":
        return {
            "use_rope": True,
            "input_sinusoid": _input("per_pair_amplitude"),
        }
    if arm == "input-scalar-control":
        return {
            "use_rope": True,
            "input_sinusoid": _input("tied_scalar"),
        }
    raise ValueError(f"Unknown Phase-44 arm: {arm}")


def _development_base(*, run_name: str, output_root: Path) -> dict:
    payload = _paper_base(run_name=run_name, output_root=output_root)
    payload.update(
        {
            "max_train_steps": 20_000,
            "num_warmup_steps": 200,
            "checkpointing_steps": 5_000,
            "checkpoint_keep_latest": 1,
            "validate_every": 5_000,
            "wandb_group": "phase44-input-adapters-b32-s20000",
        }
    )
    return payload


def generate() -> list[Path]:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    preflight_dir = CONFIG_DIR / "preflight"
    preflight_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for arm_index, arm in enumerate(ARMS):
        run_name = f"phase44-{arm}-seed123-b32-s20000-h768d8"
        payload = _development_base(run_name=run_name, output_root=OUTPUT_DIR)
        payload.update(_arm_payload(arm))
        path = CONFIG_DIR / f"{arm_index:02d}-{arm}.json"
        path.write_text(json.dumps(_resolve(payload), indent=2, sort_keys=True) + "\n")
        written.append(path)

        if arm not in NOVEL_ARMS:
            continue
        preflight_name = f"phase44-preflight-{arm}-seed123-b32-s20-h768d8"
        preflight = _development_base(
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
                "final_validation_start_batch": 256,
                "save_evaluation_details": False,
                "save_final_model": False,
                "log_every_n_steps": 5,
                "wandb_group": "phase44-input-adapters-preflight",
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
