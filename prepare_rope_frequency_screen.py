#!/usr/bin/env python
"""Generate locked phase-20 learned-RoPE frequency configurations."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CONFIG_DIR = ROOT / "sweep_configs" / "phase20_rope_frequency"
OUTPUT_ROOT = ROOT / "model-output" / "position_bias_phase20_rope_frequency"
BASE_CONFIG = (
    ROOT
    / "sweep_configs"
    / "phase17_gate_30k"
    / "phase17-standard-rope-seed123-s30000-h768d8.json"
)


def _write_locked(path: Path, payload: dict, *, force: bool) -> None:
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists() and path.read_text() != rendered and not force:
        raise RuntimeError(
            f"Refusing to change locked frequency config {path}; "
            "pass --force only for an intentional protocol revision."
        )
    path.write_text(rendered)


def build_configs(*, force: bool = False) -> list[Path]:
    base = json.loads(BASE_CONFIG.read_text())
    modes = {
        "fixed": "fixed",
        "layer-shared": "layer_shared",
        "layer-head": "layer_head",
    }
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for seed in (123, 456, 789):
        for arm, mode in modes.items():
            config = copy.deepcopy(base)
            run_name = f"phase20-{arm}-seed{seed}-s5000-h768d8"
            config.update(
                {
                    "seed": seed,
                    "paired_initialization_seed": seed,
                    "run_name": run_name,
                    "base_output_dir": str(OUTPUT_ROOT),
                    "rope_frequency_mode": mode,
                    "qk": {"enabled": False},
                    "logit_bias": {"enabled": False},
                    "residual_stream": {"enabled": False},
                    "attention_write": {"enabled": False},
                    "attn_impl": "sdpa",
                    "qk_norm_mode": "method_aware_rms",
                    "training_length": 1024,
                    "model_position_extent": 4096,
                    "evaluation_lengths": [1024, 2048, 4096],
                    "scalar_normalization_extent": 1024,
                    "per_device_eval_batch_size": 1,
                    "max_train_steps": 5_000,
                    "validate_every": 5_000,
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
                }
            )
            path = CONFIG_DIR / f"{run_name}.json"
            _write_locked(path, config, force=force)
            paths.append(path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    paths = build_configs(force=args.force)
    print(f"locked {len(paths)} frequency configs under {CONFIG_DIR}")


if __name__ == "__main__":
    main()
