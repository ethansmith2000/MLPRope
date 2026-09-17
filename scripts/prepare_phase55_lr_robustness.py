#!/usr/bin/env python
"""Generate the frozen symmetric outer-LR paper controls and preflights."""

from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.prepare_phase42_paper_components import _paper_base
from scripts.prepare_phase49_mature_qk_readout import _resolve, _scalar_qkpre


CONFIG_ROOT = ROOT / "sweep_configs" / "phase55_lr_robustness"
OUTPUT_ROOT = ROOT / "model-output" / "position_bias_phase55_lr_robustness"
PREFLIGHT_OUTPUT_ROOT = ROOT / "model-output" / "position_bias_phase55_lr_robustness_preflight"
FINAL_HOLDOUT_START = 6_144
LEARNING_RATES = (1.5e-4, 6.0e-4)
ARMS = ("rope", "scalar-qkpre")


def _arm_payload(arm: str) -> dict:
    if arm == "rope":
        return {"use_rope": True, "qk_preprojection": {"enabled": False}}
    if arm == "scalar-qkpre":
        return {"use_rope": True, "qk_preprojection": _scalar_qkpre()}
    raise ValueError(f"Unknown Phase-55 arm: {arm}")


def _lr_tag(learning_rate: float) -> str:
    return {1.5e-4: "lr1p5e4", 6.0e-4: "lr6e4"}[learning_rate]


def _base(*, run_name: str, output_root: Path, learning_rate: float) -> dict:
    payload = _paper_base(run_name=run_name, output_root=output_root)
    payload.update({
        "seed": 123,
        "paired_initialization_seed": 123,
        "learning_rate": learning_rate,
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
        "final_validation_start_batch": FINAL_HOLDOUT_START,
        "wandb_group": "phase55-lr-robustness-b32-s100000",
    })
    return payload


def generate() -> list[Path]:
    CONFIG_ROOT.mkdir(parents=True, exist_ok=True)
    preflight_root = CONFIG_ROOT / "preflight"
    preflight_root.mkdir(parents=True, exist_ok=True)
    written = []
    index = 0
    for learning_rate in LEARNING_RATES:
        for arm in ARMS:
            tag = _lr_tag(learning_rate)
            run_name = f"phase55-{arm}-{tag}-seed123-b32-s100000-h768d8"
            payload = _base(
                run_name=run_name,
                output_root=OUTPUT_ROOT,
                learning_rate=learning_rate,
            )
            payload.update(_arm_payload(arm))
            path = CONFIG_ROOT / f"{index:02d}-{arm}-{tag}.json"
            path.write_text(json.dumps(_resolve(payload), indent=2, sort_keys=True) + "\n")
            written.append(path)

            preflight_name = f"phase55-preflight-{arm}-{tag}-seed123-b32-s20-h768d8"
            preflight = _base(
                run_name=preflight_name,
                output_root=PREFLIGHT_OUTPUT_ROOT,
                learning_rate=learning_rate,
            )
            preflight.update(_arm_payload(arm))
            preflight.update({
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
                "wandb_group": "phase55-lr-robustness-preflight",
            })
            preflight_path = preflight_root / f"{index:02d}-{arm}-{tag}.json"
            preflight_path.write_text(
                json.dumps(_resolve(preflight), indent=2, sort_keys=True) + "\n"
            )
            written.append(preflight_path)
            index += 1
    return written


if __name__ == "__main__":
    for path in generate():
        print(path.relative_to(ROOT))
