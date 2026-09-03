#!/usr/bin/env python
"""Generate the locked Phase-31 causal-EMA dynamic-position screen."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
BASE_CONFIG = (
    ROOT
    / "sweep_configs"
    / "phase30_qkpre_addrope_factorial_15k"
    / "phase30-rope-fixed-seed123-s15000-h768d8.json"
)
POSITION_QK_SOURCE = (
    ROOT
    / "sweep_configs"
    / "phase19_confirmation"
    / "phase19-position-only-seed123-s30000-h1024d12.json"
)
CONTENT_QK_SOURCE = (
    ROOT
    / "sweep_configs"
    / "phase19_confirmation"
    / "phase19-content-position-seed123-s30000-h1024d12.json"
)
CONFIG_DIR = ROOT / "sweep_configs" / "phase31_dynamic_ema_screen"
PREFLIGHT_CONFIG_DIR = CONFIG_DIR / "preflight"
OUTPUT_ROOT = ROOT / "model-output" / "position_bias_phase31_dynamic_ema"
PREFLIGHT_OUTPUT_ROOT = ROOT / "model-output" / "position_bias_phase31_preflight"
SEED = 123
STEPS = 15_000
ARMS = (
    "rope-fixed",
    "clock-pointwise",
    "clock-ema",
    "addrope-position",
    "addrope-content-pointwise",
    "addrope-content-ema",
)
PREFLIGHT_ARMS = (
    "clock-ema",
    "addrope-content-pointwise",
    "addrope-content-ema",
)


def _disabled_mechanisms() -> dict:
    return {
        "qk": {"enabled": False},
        "qk_preprojection": {"enabled": False},
        "rotary_clock": {"enabled": False},
        "position_gain": {"enabled": False},
    }


def _clock(temporal: str) -> dict:
    config = {
        "enabled": True,
        "source": "normalized_residual",
        "head_coupling": "per_head",
        "mapper": "low_rank_silu",
        "rank": 32,
        "temporal": temporal,
        "kernel_size": 1,
        # At length 1,024 this hard-limits worst-case cumulative displacement
        # to about 10.2 positions, versus 255.8 in the exploratory pilot.
        "speed_bound": 0.01,
    }
    if temporal == "ema":
        config["ema_decay_init"] = 0.9
    return config


def _carrier_qk(*, input_mode: str, temporal: str) -> dict:
    source_path = (
        POSITION_QK_SOURCE if input_mode == "position" else CONTENT_QK_SOURCE
    )
    qk = copy.deepcopy(json.loads(source_path.read_text())["qk"])
    conditioning = qk["conditioning"]
    conditioning["input_mode"] = input_mode
    conditioning["temporal"] = temporal
    conditioning["ema_decay_init"] = 0.9
    return qk


def _arm_overrides(arm: str) -> dict:
    overrides = _disabled_mechanisms()
    overrides["use_rope"] = True
    if arm == "rope-fixed":
        pass
    elif arm == "clock-pointwise":
        overrides["rotary_clock"] = _clock("pointwise")
    elif arm == "clock-ema":
        overrides["rotary_clock"] = _clock("ema")
    elif arm == "addrope-position":
        overrides["qk"] = _carrier_qk(
            input_mode="position",
            temporal="pointwise",
        )
    elif arm == "addrope-content-pointwise":
        overrides["qk"] = _carrier_qk(
            input_mode="content_position",
            temporal="pointwise",
        )
    elif arm == "addrope-content-ema":
        overrides["qk"] = _carrier_qk(
            input_mode="content_position",
            temporal="ema",
        )
    else:
        raise ValueError(f"Unknown Phase-31 arm: {arm}")
    return overrides


def _write_locked(path: Path, payload: dict, *, force: bool) -> None:
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists() and path.read_text() != rendered and not force:
        raise RuntimeError(
            f"Refusing to change locked Phase-31 config {path}; pass --force "
            "only for an intentional protocol revision."
        )
    path.write_text(rendered)


def build(*, force: bool = False) -> list[Path]:
    base = json.loads(BASE_CONFIG.read_text())
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    PREFLIGHT_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    PREFLIGHT_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    written = []
    for arm in ARMS:
        cfg = copy.deepcopy(base)
        run_name = f"phase31-{arm}-seed{SEED}-s{STEPS}-h768d8"
        cfg.update(
            {
                "run_name": run_name,
                "base_output_dir": str(OUTPUT_ROOT),
                "seed": SEED,
                "paired_initialization_seed": SEED,
                "max_train_steps": STEPS,
                "model_position_extent": 1_024,
                "scalar_normalization_extent": 1_024,
                "evaluation_lengths": [1_024],
                "validate_every": 5_000,
                "num_validation_batches": 25,
                "validation_start_batch": 0,
                "num_final_validation_batches": 256,
                "final_validation_start_batch": 2_048,
                "save_evaluation_details": True,
                "save_final_model": True,
                "checkpointing_steps": 5_000,
                "resume_from_checkpoint": "auto",
                "with_tracking": False,
                "profile_every_n_steps": 0,
                "log_every_n_steps": 50,
            }
        )
        cfg.update(_arm_overrides(arm))
        cfg.pop("output_dir", None)
        path = CONFIG_DIR / f"{run_name}.json"
        _write_locked(path, cfg, force=force)
        written.append(path)

        if arm in PREFLIGHT_ARMS:
            preflight = copy.deepcopy(cfg)
            preflight_name = f"phase31-preflight-{arm}-seed{SEED}-s20-h768d8"
            preflight.update(
                {
                    "run_name": preflight_name,
                    "base_output_dir": str(PREFLIGHT_OUTPUT_ROOT),
                    "max_train_steps": 20,
                    "validate_every": 20,
                    "num_validation_batches": 2,
                    "num_final_validation_batches": 2,
                    "save_evaluation_details": False,
                    "save_final_model": False,
                    "checkpointing_steps": None,
                    "resume_from_checkpoint": False,
                }
            )
            _write_locked(
                PREFLIGHT_CONFIG_DIR / f"{preflight_name}.json",
                preflight,
                force=force,
            )
    return written


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    paths = build(force=parser.parse_args().force)
    print(
        f"locked {len(paths)} Phase-31 screen configs and "
        f"{len(PREFLIGHT_ARMS)} preflights under {CONFIG_DIR}"
    )


if __name__ == "__main__":
    main()
