#!/usr/bin/env python
"""Generate the frozen Phase-58 scalar-gate initialization scout configs."""

from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.prepare_phase42_paper_components import _paper_base
from scripts.prepare_phase49_mature_qk_readout import _resolve


CONFIG_ROOT = ROOT / "sweep_configs" / "phase58_gate_initialization"
OUTPUT_ROOT = ROOT / "model-output" / "position_bias_phase58_gate_initialization"
PREFLIGHT_OUTPUT_ROOT = (
    ROOT / "model-output" / "position_bias_phase58_gate_initialization_preflight"
)
LEARNING_RATE = 1.2e-3
TRAINING_STEPS = 20_000
FINAL_START = 8_192
ARMS = (
    "alpha1-direct",
    "alpha01-direct",
    "alpha01-scaled",
    "alpha01-fixed",
)


def _qkpre(*, gate_init: float, gate_output_scale: float, learnable: bool) -> dict:
    return {
        "enabled": True,
        "mode": "tied_scalar",
        "gate_init": gate_init,
        "gate_output_scale": gate_output_scale,
        "learnable_gate": learnable,
        "gate_sharing": "per_layer",
        "active_layers": None,
    }


def _arm_payload(arm: str) -> dict:
    if arm == "alpha1-direct":
        qkpre = _qkpre(gate_init=1.0, gate_output_scale=1.0, learnable=True)
    elif arm == "alpha01-direct":
        qkpre = _qkpre(gate_init=0.1, gate_output_scale=1.0, learnable=True)
    elif arm == "alpha01-scaled":
        qkpre = _qkpre(gate_init=1.0, gate_output_scale=0.1, learnable=True)
    elif arm == "alpha01-fixed":
        qkpre = _qkpre(gate_init=0.1, gate_output_scale=1.0, learnable=False)
    else:
        raise ValueError(f"Unknown Phase-58 arm: {arm}")
    return {
        "use_rope": True,
        "qk_projection_bias": False,
        "qk": {"enabled": False},
        "input_sinusoid": {"enabled": False},
        "qk_preprojection": qkpre,
    }


def _base(*, run_name: str, output_root: Path) -> dict:
    payload = _paper_base(run_name=run_name, output_root=output_root)
    payload.update(
        {
            "seed": 123,
            "paired_initialization_seed": 123,
            "learning_rate": LEARNING_RATE,
            "beta1": 0.9,
            "beta2": 0.98,
            "weight_decay": 0.01,
            "max_grad_norm": 1.0,
            "max_train_steps": TRAINING_STEPS,
            "num_warmup_steps": 200,
            "checkpointing_steps": None,
            "checkpoint_keep_latest": None,
            "checkpoint_milestones": [],
            "resume_from_checkpoint": None,
            "save_final_model": False,
            "save_evaluation_details": True,
            "validate_every": 2_000,
            "num_final_validation_batches": 1_024,
            "final_validation_start_batch": FINAL_START,
            "intervention_optimizer_log_every": 200,
            "wandb_group": "phase58-gate-initialization-b32-s20000",
        }
    )
    return payload


def generate() -> list[Path]:
    CONFIG_ROOT.mkdir(parents=True, exist_ok=True)
    preflight_root = CONFIG_ROOT / "preflight"
    preflight_root.mkdir(parents=True, exist_ok=True)
    written = []
    for index, arm in enumerate(ARMS):
        run_name = f"phase58-{arm}-lr1p2e3-seed123-b32-s20000-h768d8"
        payload = _base(run_name=run_name, output_root=OUTPUT_ROOT)
        payload.update(_arm_payload(arm))
        path = CONFIG_ROOT / f"{index:02d}-{arm}.json"
        path.write_text(json.dumps(_resolve(payload), indent=2, sort_keys=True) + "\n")
        written.append(path)

        preflight_name = (
            f"phase58-preflight-{arm}-lr1p2e3-seed123-b32-s100-h768d8"
        )
        preflight = _base(run_name=preflight_name, output_root=PREFLIGHT_OUTPUT_ROOT)
        preflight.update(_arm_payload(arm))
        preflight.update(
            {
                "max_train_steps": 100,
                "num_warmup_steps": 10,
                "validate_every": 10_000,
                "num_validation_batches": 4,
                "num_final_validation_batches": 4,
                "final_validation_start_batch": 1_024,
                "save_evaluation_details": False,
                "intervention_optimizer_log_every": 10,
                "log_every_n_steps": 10,
                "wandb_group": "phase58-gate-initialization-preflight",
            }
        )
        preflight_path = preflight_root / f"{index:02d}-{arm}.json"
        preflight_path.write_text(
            json.dumps(_resolve(preflight), indent=2, sort_keys=True) + "\n"
        )
        written.append(preflight_path)
    return written


if __name__ == "__main__":
    for generated_path in generate():
        print(generated_path.relative_to(ROOT))
