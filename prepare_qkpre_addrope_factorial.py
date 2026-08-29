#!/usr/bin/env python
"""Generate the locked Phase-29 qkpre x AddRoPE factorial configs."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
BASE_CONFIG = (
    ROOT
    / "sweep_configs"
    / "phase28_qkpre_rope_30k"
    / "phase28-rope-fixed-seed123-s30000-h768d8.json"
)
QKPRE_CONFIG = (
    ROOT
    / "sweep_configs"
    / "phase28_qkpre_rope_30k"
    / "phase28-qkpre-rope-seed123-s30000-h768d8.json"
)
ADDROPE_CONFIG = (
    ROOT
    / "sweep_configs"
    / "phase25_rope_embed_basis_30k"
    / "phase25-basis16-a10-seed123-s30000-h768d8.json"
)
CONFIG_DIR = ROOT / "sweep_configs" / "phase29_qkpre_addrope_factorial"
OUTPUT_ROOT = ROOT / "model-output" / "position_bias_phase29_qkpre_addrope_factorial"
SEED = 123
STEPS = 5_000
ARMS = {
    "rope-fixed": (False, False),
    "qkpre-rope": (True, False),
    "addrope-a10": (False, True),
    "qkpre-addrope-a10": (True, True),
}


def _write_locked(path: Path, payload: dict, *, force: bool) -> None:
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists() and path.read_text() != rendered and not force:
        raise RuntimeError(
            f"Refusing to change locked Phase-29 config {path}; pass --force "
            "only for an intentional protocol revision."
        )
    path.write_text(rendered)


def build(*, force: bool = False) -> list[Path]:
    base = json.loads(BASE_CONFIG.read_text())
    qkpre = json.loads(QKPRE_CONFIG.read_text())["qk_preprojection"]
    addrope = json.loads(ADDROPE_CONFIG.read_text())["qk"]
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    written = []
    for arm, (qkpre_enabled, addrope_enabled) in ARMS.items():
        cfg = copy.deepcopy(base)
        run_name = f"phase29-{arm}-seed{SEED}-s{STEPS}-h768d8"
        cfg.update(
            {
                "run_name": run_name,
                "base_output_dir": str(OUTPUT_ROOT),
                "seed": SEED,
                "paired_initialization_seed": SEED,
                "max_train_steps": STEPS,
                "model_position_extent": 1_024,
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
                "use_rope": True,
                "qk_preprojection": (
                    copy.deepcopy(qkpre)
                    if qkpre_enabled
                    else {"enabled": False}
                ),
                "qk": (
                    copy.deepcopy(addrope)
                    if addrope_enabled
                    else {"enabled": False}
                ),
            }
        )
        cfg.pop("output_dir", None)
        path = CONFIG_DIR / f"{run_name}.json"
        _write_locked(path, cfg, force=force)
        written.append(path)
    return written


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    paths = build(force=parser.parse_args().force)
    print(f"locked {len(paths)} Phase-29 factorial configs under {CONFIG_DIR}")


if __name__ == "__main__":
    main()
