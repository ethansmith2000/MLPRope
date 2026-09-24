#!/usr/bin/env python
"""Generate the frozen Phase-61 modern larger-scale transfer configs."""

from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.prepare_phase42_paper_components import _paper_base
from scripts.prepare_phase49_mature_qk_readout import _resolve


CONFIG_ROOT = ROOT / "sweep_configs" / "phase61_modern_lscale"
OUTPUT_ROOT = ROOT / "model-output" / "position_bias_phase61_modern_lscale"
PREFLIGHT_OUTPUT_ROOT = (
    ROOT / "model-output" / "position_bias_phase61_modern_lscale_preflight"
)
FINAL_START = 11_264
ARMS = ("rope", "scalar-qkpre")


def _scalar_qkpre() -> dict:
    return {
        "enabled": True,
        "mode": "tied_scalar",
        "gate_init": 1.0,
        "gate_output_scale": 1.0,
        "learnable_gate": True,
        "gate_sharing": "per_layer",
        "active_layers": None,
    }


def _arm_payload(arm: str) -> dict:
    payload = {
        "use_rope": True,
        "rope_fraction": 1.0,
        "use_alibi": False,
        "use_learned_absolute_position": False,
        "qk_projection_bias": False,
        "qk": {"enabled": False},
        "qk_preprojection": {"enabled": False},
        "input_sinusoid": {"enabled": False},
        "attn_impl": "sdpa",
    }
    if arm == "scalar-qkpre":
        payload["qk_preprojection"] = _scalar_qkpre()
    elif arm != "rope":
        raise ValueError(f"Unknown Phase-61 arm: {arm}")
    return payload


def _base(*, run_name: str, output_root: Path) -> dict:
    payload = _paper_base(run_name=run_name, output_root=output_root)
    payload.update(
        {
            "backbone_variant": "modern",
            "hidden_size": 1_024,
            "depth": 12,
            "n_head": 8,
            "ff_hidden_dim": None,
            "ff_widened_hidden_dim": None,
            "ff_widened_layers": [],
            "qk_norm": True,
            "qk_norm_mode": "method_aware_rms",
            "post_position_qk_norm": False,
            "seed": 123,
            "paired_initialization_seed": 123,
            "learning_rate": 1.2e-3,
            "beta1": 0.9,
            "beta2": 0.98,
            "weight_decay": 0.01,
            "max_grad_norm": 1.0,
            "per_device_train_batch_size": 32,
            "gradient_accumulation_steps": 1,
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
            "final_validation_start_batch": FINAL_START,
            "wandb_group": "phase61-modern-lscale-b32-s100000",
        }
    )
    return payload


def generate() -> list[Path]:
    CONFIG_ROOT.mkdir(parents=True, exist_ok=True)
    preflight_root = CONFIG_ROOT / "preflight"
    preflight_root.mkdir(parents=True, exist_ok=True)
    written = []
    for index, arm in enumerate(ARMS):
        run_name = (
            f"phase61-modern-lscale-{arm}-lr1p2e3-seed123-"
            "b32-s100000-h1024d12"
        )
        payload = _base(run_name=run_name, output_root=OUTPUT_ROOT)
        payload.update(_arm_payload(arm))
        path = CONFIG_ROOT / f"{index:02d}-{arm}.json"
        path.write_text(json.dumps(_resolve(payload), indent=2, sort_keys=True) + "\n")
        written.append(path)

        preflight_name = (
            f"phase61-preflight-modern-lscale-{arm}-lr1p2e3-seed123-"
            "b32-s100-h1024d12"
        )
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
                "final_validation_start_batch": 1_024,
                "save_evaluation_details": False,
                "save_final_model": False,
                "log_every_n_steps": 10,
                "intervention_optimizer_log_every": 10,
                "wandb_group": "phase61-modern-lscale-preflight",
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
