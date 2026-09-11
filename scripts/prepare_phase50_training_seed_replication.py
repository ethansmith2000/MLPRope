#!/usr/bin/env python
"""Generate the two-seed replication of the mature rank-32 result."""

from __future__ import annotations

import json
from pathlib import Path
import sys

REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from scripts.prepare_phase42_paper_components import _paper_base
from scripts.prepare_phase49_mature_qk_readout import (
    FINAL_HOLDOUT_START,
    _arm_payload,
    _resolve,
)


CONFIG_DIR = REPO_DIR / "sweep_configs" / "phase50_training_seed_replication"
OUTPUT_DIR = (
    REPO_DIR / "model-output" / "position_bias_phase50_training_seed_replication"
)
SEEDS = (456, 789)
# Keep the primary comparison adjacent so it occupies the first two slots for
# each seed. The launcher enforces a hard two-job concurrency ceiling.
ARMS = (
    "qk-readout-r32",
    "scalar-qkpre",
    "scalar-ffnmatch-r32",
    "rope",
)


def generate() -> list[Path]:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    written = []
    index = 0
    for seed in SEEDS:
        for arm in ARMS:
            run_name = f"phase50-{arm}-seed{seed}-b32-s100000-h768d8"
            payload = _paper_base(run_name=run_name, output_root=OUTPUT_DIR)
            payload.update(
                {
                    "seed": seed,
                    "paired_initialization_seed": seed,
                    "max_train_steps": 100_000,
                    "num_warmup_steps": 200,
                    "checkpointing_steps": 5_000,
                    "checkpoint_keep_latest": 1,
                    "validate_every": 5_000,
                    "final_validation_start_batch": FINAL_HOLDOUT_START,
                    "wandb_group": "phase50-training-seed-replication-b32-s100000",
                }
            )
            payload.update(_arm_payload(arm))
            path = CONFIG_DIR / f"{index:02d}-seed{seed}-{arm}.json"
            path.write_text(json.dumps(_resolve(payload), indent=2, sort_keys=True) + "\n")
            written.append(path)
            index += 1
    return written


if __name__ == "__main__":
    for config_path in generate():
        print(config_path.relative_to(REPO_DIR))
