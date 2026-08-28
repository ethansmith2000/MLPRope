#!/usr/bin/env python
"""Generate locked seed-456/789 configs for Phase-26 survivors.

Each arm is copied from its exact seed-123 Phase-26 config.  Only the seed,
paired-initialization seed, run name, and output root change.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE_DIR = ROOT / "sweep_configs" / "phase26_position_breadth"
CONFIG_DIR = ROOT / "sweep_configs" / "phase27_position_survivor_replication"
OUTPUT_ROOT = ROOT / "model-output" / "position_bias_phase27_survivor_replication"
SEEDS = (456, 789)
ARMS = ("rope-fixed", "qkpre-rope", "posgain-qk")
STEPS = 5_000


def _write_locked(path: Path, payload: dict, *, force: bool) -> None:
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists() and path.read_text() != rendered and not force:
        raise RuntimeError(
            f"Refusing to change locked phase-27 config {path}; pass --force "
            "only for an intentional protocol revision."
        )
    path.write_text(rendered)


def build(*, force: bool = False) -> list[Path]:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    written = []
    for seed in SEEDS:
        for arm in ARMS:
            source_name = f"phase26-{arm}-seed123-s{STEPS}-h768d8.json"
            cfg = copy.deepcopy(json.loads((SOURCE_DIR / source_name).read_text()))
            run_name = f"phase27-{arm}-seed{seed}-s{STEPS}-h768d8"
            cfg.update(
                {
                    "run_name": run_name,
                    "base_output_dir": str(OUTPUT_ROOT),
                    "seed": seed,
                    "paired_initialization_seed": seed,
                    "resume_from_checkpoint": "auto",
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
    print(f"locked {len(paths)} phase-27 configs under {CONFIG_DIR}")


if __name__ == "__main__":
    main()
