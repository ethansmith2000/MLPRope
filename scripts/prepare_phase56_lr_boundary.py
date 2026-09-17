#!/usr/bin/env python
"""Generate the frozen Phase-56 one-point LR boundary configs."""

from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.prepare_phase42_paper_components import _paper_base
from scripts.prepare_phase49_mature_qk_readout import _resolve, _scalar_qkpre


CONFIG_ROOT = ROOT / "sweep_configs" / "phase56_lr_boundary"
OUTPUT_ROOT = ROOT / "model-output" / "position_bias_phase56_lr_boundary"
PREFLIGHT_OUTPUT_ROOT = ROOT / "model-output" / "position_bias_phase56_lr_boundary_preflight"
LEARNING_RATE = 1.2e-3
FINAL_SELECTION_START = 6_144
ARMS = ("rope", "scalar-qkpre")


def _arm_payload(arm: str) -> dict:
    if arm == "rope":
        return {"use_rope": True, "qk_preprojection": {"enabled": False}}
    if arm == "scalar-qkpre":
        return {"use_rope": True, "qk_preprojection": _scalar_qkpre()}
    raise ValueError(f"Unknown Phase-56 arm: {arm}")


def _base(*, run_name: str, output_root: Path) -> dict:
    payload = _paper_base(run_name=run_name, output_root=output_root)
    payload.update({
        "seed": 123,
        "paired_initialization_seed": 123,
        "learning_rate": LEARNING_RATE,
        "beta1": 0.9,
        "beta2": 0.98,
        "weight_decay": 0.01,
        "max_grad_norm": 1.0,
        "max_train_steps": 100_000,
        "num_warmup_steps": 200,
        "checkpointing_steps": 10_000,
        "checkpoint_keep_latest": 1,
        "checkpoint_milestones": [],
        "resume_from_checkpoint": "auto",
        "save_final_model": False,
        "save_evaluation_details": True,
        "validate_every": 5_000,
        "num_final_validation_batches": 1_024,
        "final_validation_start_batch": FINAL_SELECTION_START,
        "wandb_group": "phase56-lr-boundary-b32-s100000",
    })
    return payload


def generate() -> list[Path]:
    CONFIG_ROOT.mkdir(parents=True, exist_ok=True)
    preflight_root = CONFIG_ROOT / "preflight"
    preflight_root.mkdir(parents=True, exist_ok=True)
    written = []
    for index, arm in enumerate(ARMS):
        run_name = f"phase56-{arm}-lr1p2e3-seed123-b32-s100000-h768d8"
        payload = _base(run_name=run_name, output_root=OUTPUT_ROOT)
        payload.update(_arm_payload(arm))
        path = CONFIG_ROOT / f"{index:02d}-{arm}-lr1p2e3.json"
        path.write_text(json.dumps(_resolve(payload), indent=2, sort_keys=True) + "\n")
        written.append(path)

        preflight_name = f"phase56-preflight-{arm}-lr1p2e3-seed123-b32-s100-h768d8"
        preflight = _base(run_name=preflight_name, output_root=PREFLIGHT_OUTPUT_ROOT)
        preflight.update(_arm_payload(arm))
        preflight.update({
            "max_train_steps": 100,
            "num_warmup_steps": 10,
            "checkpointing_steps": None,
            "checkpoint_keep_latest": None,
            "resume_from_checkpoint": None,
            "validate_every": 10_000,
            "num_validation_batches": 4,
            "num_final_validation_batches": 4,
            "final_validation_start_batch": 512,
            "save_evaluation_details": False,
            "save_final_model": False,
            "log_every_n_steps": 10,
            "wandb_group": "phase56-lr-boundary-preflight",
        })
        preflight_path = preflight_root / f"{index:02d}-{arm}-lr1p2e3.json"
        preflight_path.write_text(
            json.dumps(_resolve(preflight), indent=2, sort_keys=True) + "\n"
        )
        written.append(preflight_path)
    return written


if __name__ == "__main__":
    for generated_path in generate():
        print(generated_path.relative_to(ROOT))
