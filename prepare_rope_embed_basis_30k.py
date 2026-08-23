#!/usr/bin/env python
"""Generate the locked phase-25 additive-carrier 30k promotion configs.

Each configuration is derived from its exact phase-24 arm/seed predecessor.
The intervention is unchanged; phase 25 extends training to 30k steps, uses a
single primary context, and enlarges the disjoint final holdout.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE_DIR = ROOT / "sweep_configs" / "phase24_rope_embed_basis"
CONFIG_DIR = ROOT / "sweep_configs" / "phase25_rope_embed_basis_30k"
OUTPUT_ROOT = ROOT / "model-output" / "position_bias_phase25_rope_embed_basis_30k"
SEEDS = (123, 456, 789)
ARMS = ("rope-fixed", "basis16-a03", "basis16-a10")


def _write_locked(path: Path, payload: dict, *, force: bool) -> None:
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists() and path.read_text() != rendered and not force:
        raise RuntimeError(
            f"Refusing to change locked phase-25 config {path}; "
            "pass --force only for an intentional protocol revision."
        )
    path.write_text(rendered)


def build(*, force: bool = False) -> list[Path]:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for seed in SEEDS:
        for arm in ARMS:
            source_name = f"phase24-{arm}-seed{seed}-s5000-h768d8.json"
            cfg = json.loads((SOURCE_DIR / source_name).read_text())
            run_name = f"phase25-{arm}-seed{seed}-s30000-h768d8"
            cfg.update(
                {
                    "run_name": run_name,
                    "base_output_dir": str(OUTPUT_ROOT),
                    "max_train_steps": 30_000,
                    "evaluation_lengths": [1024],
                    "validate_every": 5_000,
                    "num_validation_batches": 25,
                    "validation_start_batch": 0,
                    "num_final_validation_batches": 1_024,
                    "final_validation_start_batch": 2_048,
                    "save_evaluation_details": True,
                    "save_final_model": True,
                    "checkpointing_steps": 10_000,
                    "resume_from_checkpoint": "auto",
                    "with_tracking": False,
                    "profile_every_n_steps": 0,
                    "log_every_n_steps": 50,
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
    print(f"locked {len(paths)} phase-25 configs under {CONFIG_DIR}")


if __name__ == "__main__":
    main()
