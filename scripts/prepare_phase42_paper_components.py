#!/usr/bin/env python
"""Generate the matched batch-32 paper component and comparison configs."""

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


CONFIG_DIR = REPO_DIR / "sweep_configs" / "phase42_paper_components"
OUTPUT_DIR = REPO_DIR / "model-output" / "position_bias_phase42_paper_components"
PREFLIGHT_OUTPUT_DIR = (
    REPO_DIR / "model-output" / "position_bias_phase42_paper_components_preflight"
)
ARMS = (
    "nope",
    "rope",
    "qkpre-nope",
    "qkpre-rope",
    "qkpre-fixed-rope",
    "qkpre-global-rope",
    "qkpre-first-rope",
    "input-rope",
    "addrope-fixed-nope",
    "addrope-fixed-postrope",
)
NOVEL_PREFLIGHT_ARMS = (
    "qkpre-fixed-rope",
    "qkpre-global-rope",
    "qkpre-first-rope",
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


def _paper_base(*, run_name: str, output_root: Path) -> dict:
    payload = _base_payload(run_name=run_name, output_root=output_root)
    payload.update(
        {
            "per_device_train_batch_size": 32,
            "max_train_steps": 100_000,
            "num_warmup_steps": 200,
            "checkpointing_steps": 5_000,
            "checkpoint_keep_latest": 1,
            "validate_every": 5_000,
            "wandb_group": "phase42-paper-components-b32",
        }
    )
    return payload


def _qkpre(
    *,
    learnable: bool = True,
    gate_sharing: str = "per_layer",
    active_layers: list[int] | None = None,
) -> dict:
    return {
        "enabled": True,
        "mode": "tied_scalar",
        "gate_init": 1.0,
        "learnable_gate": learnable,
        "gate_sharing": gate_sharing,
        "active_layers": active_layers,
    }


def _arm_payload(arm: str) -> dict:
    if arm == "nope":
        return {"use_rope": False}
    if arm == "rope":
        return {}
    if arm == "qkpre-nope":
        return {"use_rope": False, "qk_preprojection": _qkpre()}
    if arm == "qkpre-rope":
        return {"qk_preprojection": _qkpre()}
    if arm == "qkpre-fixed-rope":
        return {"qk_preprojection": _qkpre(learnable=False)}
    if arm == "qkpre-global-rope":
        return {"qk_preprojection": _qkpre(gate_sharing="global")}
    if arm == "qkpre-first-rope":
        return {"qk_preprojection": _qkpre(active_layers=[0])}
    if arm == "input-rope":
        return {
            "input_sinusoid": {
                "enabled": True,
                "gate_init": 1.0,
                "learnable_gate": True,
            }
        }
    if arm == "addrope-fixed-nope":
        return {"qk": _direct_addrope(learned=False), "use_rope": False}
    if arm == "addrope-fixed-postrope":
        qk = _direct_addrope(learned=False)
        qk["placement"] = "after_rope"
        return {"qk": qk, "use_rope": True}
    raise ValueError(f"Unknown Phase-42 arm: {arm}")


def generate() -> list[Path]:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    preflight_dir = CONFIG_DIR / "preflight"
    preflight_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for arm_index, arm in enumerate(ARMS):
        run_name = f"phase42-{arm}-seed123-b32-s100000-h768d8"
        payload = _paper_base(run_name=run_name, output_root=OUTPUT_DIR)
        payload.update(_arm_payload(arm))
        path = CONFIG_DIR / f"{arm_index:02d}-{arm}.json"
        path.write_text(json.dumps(_resolve(payload), indent=2, sort_keys=True) + "\n")
        written.append(path)

        if arm not in NOVEL_PREFLIGHT_ARMS:
            continue
        preflight_name = f"phase42-preflight-{arm}-seed123-b32-s20-h768d8"
        preflight = _paper_base(
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
                "wandb_group": "phase42-paper-components-preflight",
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
