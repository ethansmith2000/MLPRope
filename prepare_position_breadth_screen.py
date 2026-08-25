#!/usr/bin/env python
"""Generate the locked single-seed phase-26 position breadth screen.

The purpose of this stage is mechanism triage, not a multi-seed claim.  Every
arm uses the seed-123 phase-24 control as its training/evaluation protocol and
stable named initialization.  Only the position intervention changes.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE_DIR = ROOT / "sweep_configs" / "phase24_rope_embed_basis"
CONFIG_DIR = ROOT / "sweep_configs" / "phase26_position_breadth"
PREFLIGHT_CONFIG_DIR = CONFIG_DIR / "preflight"
OUTPUT_ROOT = ROOT / "model-output" / "position_bias_phase26_breadth"
PREFLIGHT_OUTPUT_ROOT = ROOT / "model-output" / "position_bias_phase26_preflight"
SEED = 123
STEPS = 5_000
ARMS = (
    "rope-fixed",
    "nope",
    "addrope-a10",
    "posgain-q",
    "posgain-k",
    "posgain-qk",
    "qkpre-nope",
    "qkpre-rope",
    "clock-pointwise",
    "clock-causalconv",
)
PREFLIGHT_ARMS = (
    "posgain-qk",
    "qkpre-rope",
    "clock-pointwise",
    "clock-causalconv",
)


def _disabled_mechanisms() -> dict:
    return {
        "qk": {"enabled": False},
        "qk_preprojection": {"enabled": False},
        "rotary_clock": {"enabled": False},
        "position_gain": {"enabled": False},
    }


def _addrope_a10() -> dict:
    source = SOURCE_DIR / (
        f"phase24-basis16-a10-seed{SEED}-s{STEPS}-h768d8.json"
    )
    return copy.deepcopy(json.loads(source.read_text())["qk"])


def _position_gain(target: str) -> dict:
    return {
        "enabled": True,
        "target": target,
        "head_coupling": "per_head",
        "basis_dim": 16,
        "theta": None,
        "scalars": ["normalized_position", "log_position"],
        "mapper": "linear",
        "hidden_dim": 96,
        "log_gain_bound": 1.0,
    }


def _qk_preprojection() -> dict:
    return {
        "enabled": True,
        "basis_dim": 768,
        "theta": None,
        "gate_init": 1.0,
        "learnable_gate": True,
    }


def _rotary_clock(temporal: str) -> dict:
    return {
        "enabled": True,
        "source": "normalized_residual",
        "head_coupling": "per_head",
        "mapper": "low_rank_silu",
        "rank": 32,
        "temporal": temporal,
        "kernel_size": 4 if temporal == "causal_conv" else 1,
        "speed_bound": 0.25,
    }


def _arm_overrides(arm: str) -> dict:
    overrides = _disabled_mechanisms()
    overrides["use_rope"] = arm not in {"nope", "qkpre-nope"}
    if arm == "addrope-a10":
        overrides["qk"] = _addrope_a10()
    elif arm == "posgain-q":
        overrides["position_gain"] = _position_gain("q")
    elif arm == "posgain-k":
        overrides["position_gain"] = _position_gain("k")
    elif arm == "posgain-qk":
        overrides["position_gain"] = _position_gain("both")
    elif arm in {"qkpre-nope", "qkpre-rope"}:
        overrides["qk_preprojection"] = _qk_preprojection()
    elif arm == "clock-pointwise":
        overrides["rotary_clock"] = _rotary_clock("pointwise")
    elif arm == "clock-causalconv":
        overrides["rotary_clock"] = _rotary_clock("causal_conv")
    elif arm not in {"rope-fixed", "nope"}:
        raise ValueError(f"Unknown phase-26 arm: {arm}")
    return overrides


def _write_locked(path: Path, payload: dict, *, force: bool) -> None:
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists() and path.read_text() != rendered and not force:
        raise RuntimeError(
            f"Refusing to change locked phase-26 config {path}; pass --force "
            "only for an intentional protocol revision."
        )
    path.write_text(rendered)


def build(*, force: bool = False) -> list[Path]:
    source = SOURCE_DIR / (
        f"phase24-rope-fixed-seed{SEED}-s{STEPS}-h768d8.json"
    )
    base = json.loads(source.read_text())
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    PREFLIGHT_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    PREFLIGHT_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    written = []
    for arm in ARMS:
        cfg = copy.deepcopy(base)
        run_name = f"phase26-{arm}-seed{SEED}-s{STEPS}-h768d8"
        cfg.update(
            {
                "run_name": run_name,
                "base_output_dir": str(OUTPUT_ROOT),
                "max_train_steps": STEPS,
                "model_position_extent": 1_024,
                "scalar_normalization_extent": 1_024,
                "evaluation_lengths": [1_024],
                "validate_every": STEPS,
                "num_validation_batches": 25,
                "validation_start_batch": 0,
                "num_final_validation_batches": 256,
                "final_validation_start_batch": 2_048,
                "save_evaluation_details": True,
                "save_final_model": True,
                "checkpointing_steps": 2_500,
                "resume_from_checkpoint": "auto",
                "with_tracking": False,
                "profile_every_n_steps": 0,
                "log_every_n_steps": 50,
                "paired_initialization_seed": SEED,
                "seed": SEED,
            }
        )
        cfg.update(_arm_overrides(arm))
        cfg.pop("output_dir", None)
        path = CONFIG_DIR / f"{run_name}.json"
        _write_locked(path, cfg, force=force)
        written.append(path)

        if arm in PREFLIGHT_ARMS:
            preflight = copy.deepcopy(cfg)
            preflight_name = f"phase26-preflight-{arm}-seed{SEED}-s20-h768d8"
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
        f"locked {len(paths)} phase-26 screen configs and "
        f"{len(PREFLIGHT_ARMS)} preflights under {CONFIG_DIR}"
    )


if __name__ == "__main__":
    main()
