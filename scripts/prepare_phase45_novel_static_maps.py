#!/usr/bin/env python
"""Generate the Phase-45 breadth-first static sinusoid-map screen."""

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


CONFIG_DIR = REPO_DIR / "sweep_configs" / "phase45_novel_static_maps"
OUTPUT_DIR = REPO_DIR / "model-output" / "position_bias_phase45_novel_static_maps"
PREFLIGHT_OUTPUT_DIR = (
    REPO_DIR / "model-output" / "position_bias_phase45_novel_static_maps_preflight"
)
ARMS = (
    "input-dense-linear",
    "input-mlp-h768",
    "qk-dense-premap",
    "qk-dense-shared",
    "qk-dense-separate",
    "qk-mlp-r128",
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


def _input(mode: str, **updates) -> dict:
    config = {
        "enabled": True,
        "mode": mode,
        "gate_init": 1.0,
        "learnable_gate": True,
    }
    config.update(updates)
    return config


def _qk(mode: str, **updates) -> dict:
    config = {
        "enabled": True,
        "mode": mode,
        "gate_init": 1.0,
        "learnable_gate": True,
        "gate_sharing": "per_layer",
        "active_layers": None,
    }
    config.update(updates)
    return config


def _arm_payload(arm: str) -> dict:
    if arm == "input-dense-linear":
        return {
            "use_rope": True,
            "input_sinusoid": _input("dense_linear_residual"),
        }
    if arm == "input-mlp-h768":
        return {
            "use_rope": True,
            "input_sinusoid": _input("residual_mlp", hidden_dim=768),
        }
    qk_modes = {
        "qk-dense-premap": "dense_premap_residual",
        "qk-dense-shared": "dense_qk_shared_residual",
        "qk-dense-separate": "dense_qk_residual",
        "qk-mlp-r128": "low_rank_qk_mlp_residual",
    }
    if arm in qk_modes:
        updates = {"rank": 128} if arm == "qk-mlp-r128" else {}
        return {
            "use_rope": True,
            "qk_preprojection": _qk(qk_modes[arm], **updates),
        }
    raise ValueError(f"Unknown Phase-45 arm: {arm}")


def _development_base(*, run_name: str, output_root: Path) -> dict:
    payload = _paper_base(run_name=run_name, output_root=output_root)
    payload.update(
        {
            "max_train_steps": 20_000,
            "num_warmup_steps": 200,
            "checkpointing_steps": 5_000,
            "checkpoint_keep_latest": 1,
            "validate_every": 5_000,
            "wandb_group": "phase45-novel-static-maps-b32-s20000",
        }
    )
    return payload


def generate() -> list[Path]:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    preflight_dir = CONFIG_DIR / "preflight"
    preflight_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for arm_index, arm in enumerate(ARMS):
        run_name = f"phase45-{arm}-seed123-b32-s20000-h768d8"
        payload = _development_base(run_name=run_name, output_root=OUTPUT_DIR)
        payload.update(_arm_payload(arm))
        path = CONFIG_DIR / f"{arm_index:02d}-{arm}.json"
        path.write_text(json.dumps(_resolve(payload), indent=2, sort_keys=True) + "\n")
        written.append(path)

        preflight_name = f"phase45-preflight-{arm}-seed123-b32-s20-h768d8"
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
                "wandb_group": "phase45-novel-static-maps-preflight",
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
