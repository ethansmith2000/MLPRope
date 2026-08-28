#!/usr/bin/env python
"""Generate locked Phase-28 qk-preprojection 30k promotion configs.

Each seed/arm starts from its exact successful 5k predecessor.  The position
mechanism remains unchanged; only the long-horizon training and evaluation
protocol plus run/output identity are updated.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PHASE26_CONFIG_ROOT = ROOT / "sweep_configs" / "phase26_position_breadth"
PHASE27_CONFIG_ROOT = ROOT / "sweep_configs" / "phase27_position_survivor_replication"
CONFIG_DIR = ROOT / "sweep_configs" / "phase28_qkpre_rope_30k"
OUTPUT_ROOT = ROOT / "model-output" / "position_bias_phase28_qkpre_rope_30k"
SEEDS = (123, 456, 789)
ARMS = ("rope-fixed", "qkpre-rope")


def predecessor(arm: str, seed: int) -> Path:
    if seed == 123:
        return (
            PHASE26_CONFIG_ROOT
            / f"phase26-{arm}-seed123-s5000-h768d8.json"
        )
    return (
        PHASE27_CONFIG_ROOT
        / f"phase27-{arm}-seed{seed}-s5000-h768d8.json"
    )


def _write_locked(path: Path, payload: dict, *, force: bool) -> None:
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists() and path.read_text() != rendered and not force:
        raise RuntimeError(
            f"Refusing to change locked Phase-28 config {path}; pass --force "
            "only for an intentional protocol revision."
        )
    path.write_text(rendered)


def build(*, force: bool = False) -> list[Path]:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    written = []
    for seed in SEEDS:
        for arm in ARMS:
            cfg = json.loads(predecessor(arm, seed).read_text())
            run_name = f"phase28-{arm}-seed{seed}-s30000-h768d8"
            cfg.update(
                {
                    "run_name": run_name,
                    "base_output_dir": str(OUTPUT_ROOT),
                    "max_train_steps": 30_000,
                    "evaluation_lengths": [1_024],
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
    print(f"locked {len(paths)} Phase-28 configs under {CONFIG_DIR}")


if __name__ == "__main__":
    main()
