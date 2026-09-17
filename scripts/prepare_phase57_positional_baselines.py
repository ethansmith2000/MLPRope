#!/usr/bin/env python
"""Generate the frozen Phase-57 recognized positional-baseline configs."""

from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.prepare_phase42_paper_components import _paper_base
from scripts.prepare_phase49_mature_qk_readout import _resolve, _scalar_qkpre


CONFIG_ROOT = ROOT / "sweep_configs" / "phase57_positional_baselines"
OUTPUT_ROOT = ROOT / "model-output" / "position_bias_phase57_positional_baselines"
PREFLIGHT_OUTPUT_ROOT = (
    ROOT / "model-output" / "position_bias_phase57_positional_baselines_preflight"
)
LEARNING_RATE = 1.2e-3
FINAL_CONFIRMATION_START = 7_168
ARMS = (
    "rope",
    "scalar-qkpre",
    "nope",
    "fixed-input-sinusoid",
    "learned-absolute",
    "partial-rope-25",
    "alibi",
)


def _disabled_position_payload() -> dict:
    return {
        "use_rope": False,
        "rope_fraction": 1.0,
        "use_alibi": False,
        "use_learned_absolute_position": False,
        "qk_projection_bias": False,
        "qk": {"enabled": False},
        "qk_preprojection": {"enabled": False},
        "input_sinusoid": {"enabled": False},
        "attn_impl": "sdpa",
    }


def _arm_payload(arm: str) -> dict:
    payload = _disabled_position_payload()
    if arm == "rope":
        payload["use_rope"] = True
    elif arm == "scalar-qkpre":
        payload["use_rope"] = True
        payload["qk_preprojection"] = _scalar_qkpre()
    elif arm == "nope":
        pass
    elif arm == "fixed-input-sinusoid":
        payload["input_sinusoid"] = {
            "enabled": True,
            "mode": "tied_scalar",
            "gate_init": 1.0,
            "learnable_gate": False,
        }
    elif arm == "learned-absolute":
        payload["use_learned_absolute_position"] = True
    elif arm == "partial-rope-25":
        payload["use_rope"] = True
        payload["rope_fraction"] = 0.25
    elif arm == "alibi":
        payload["use_alibi"] = True
        payload["attn_impl"] = "flex"
    else:
        raise ValueError(f"Unknown Phase-57 arm: {arm}")
    return payload


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
            "final_validation_start_batch": FINAL_CONFIRMATION_START,
            "wandb_group": "phase57-positional-baselines-b32-s100000",
        }
    )
    return payload


def generate() -> list[Path]:
    CONFIG_ROOT.mkdir(parents=True, exist_ok=True)
    preflight_root = CONFIG_ROOT / "preflight"
    preflight_root.mkdir(parents=True, exist_ok=True)
    written = []
    for index, arm in enumerate(ARMS):
        run_name = f"phase57-{arm}-lr1p2e3-seed123-b32-s100000-h768d8"
        payload = _base(run_name=run_name, output_root=OUTPUT_ROOT)
        payload.update(_arm_payload(arm))
        path = CONFIG_ROOT / f"{index:02d}-{arm}.json"
        path.write_text(json.dumps(_resolve(payload), indent=2, sort_keys=True) + "\n")
        written.append(path)

        preflight_name = f"phase57-preflight-{arm}-lr1p2e3-seed123-b32-s100-h768d8"
        preflight = _base(run_name=preflight_name, output_root=PREFLIGHT_OUTPUT_ROOT)
        preflight.update(_arm_payload(arm))
        preflight.update(
            {
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
                "wandb_group": "phase57-positional-baselines-preflight",
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

