#!/usr/bin/env python
"""Generate the locked Phase-32 AddRoPE EMA coefficient-axis screen."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
BASE_CONFIG = (
    ROOT
    / "sweep_configs"
    / "phase31_dynamic_ema_screen"
    / "phase31-addrope-content-pointwise-seed123-s15000-h768d8.json"
)
CONFIG_DIR = ROOT / "sweep_configs" / "phase32_dynamic_ema_coupling"
PREFLIGHT_CONFIG_DIR = CONFIG_DIR / "preflight"
OUTPUT_ROOT = ROOT / "model-output" / "position_bias_phase32_ema_coupling"
PREFLIGHT_OUTPUT_ROOT = (
    ROOT / "model-output" / "position_bias_phase32_ema_coupling_preflight"
)
SEED = 123
STEPS = 15_000
ARMS = (
    "addrope-content-pointwise",
    "addrope-content-ema-scalar",
    "addrope-content-ema-per-head",
    "addrope-content-ema-per-dim",
)
PREFLIGHT_ARMS = ARMS[1:]


def _arm_overrides(arm: str) -> dict:
    if arm == "addrope-content-pointwise":
        temporal = "pointwise"
        coupling = "per_dim"
    else:
        temporal = "ema"
        coupling = arm.removeprefix("addrope-content-ema-").replace("-", "_")
    return {
        "temporal": temporal,
        "ema_decay_init": 0.9,
        "ema_decay_coupling": coupling,
    }


def _write_locked(path: Path, payload: dict, *, force: bool) -> None:
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists() and path.read_text() != rendered and not force:
        raise RuntimeError(
            f"Refusing to change locked Phase-32 config {path}; pass --force "
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
        run_name = f"phase32-{arm}-seed{SEED}-s{STEPS}-h768d8"
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
        cfg["qk"]["conditioning"].update(_arm_overrides(arm))
        cfg.pop("output_dir", None)
        path = CONFIG_DIR / f"{run_name}.json"
        _write_locked(path, cfg, force=force)
        written.append(path)

        if arm in PREFLIGHT_ARMS:
            preflight = copy.deepcopy(cfg)
            preflight_name = f"phase32-preflight-{arm}-seed{SEED}-s20-h768d8"
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
        f"locked {len(paths)} Phase-32 screen configs and "
        f"{len(PREFLIGHT_ARMS)} preflights under {CONFIG_DIR}"
    )


if __name__ == "__main__":
    main()
