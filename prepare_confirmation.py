#!/usr/bin/env python
"""Generate the locked phase-19 paired confirmation configurations."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONFIG_DIR = ROOT / "sweep_configs" / "phase19_confirmation"
OUTPUT_ROOT = ROOT / "model-output" / "position_bias_phase19_confirmation"
PHASE18_DIR = ROOT / "sweep_configs" / "phase18_scale_h1024"


def _read(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def _write_locked(path: Path, payload: dict, *, force: bool) -> None:
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists():
        existing = path.read_text()
        if existing == rendered:
            return
        if not force:
            raise RuntimeError(
                f"Refusing to change locked confirmation config {path}; "
                "pass --force only for an intentional protocol revision."
            )
    path.write_text(rendered)


def build_configs(*, force: bool = False) -> list[Path]:
    standard = _read(
        PHASE18_DIR / "phase18-standard-rope-seed123-s30000-h1024d12.json"
    )
    position_only = _read(
        PHASE18_DIR / "phase18-position-only-seed123-s30000-h1024d12.json"
    )
    content_position = _read(
        PHASE18_DIR / "phase18-free-control-seed123-s30000-h1024d12.json"
    )
    mapped = _read(
        ROOT
        / "sweep_configs"
        / "phase9_hyper_30k"
        / "phase9-30k-mapped-addrope-a03-seed123-s30000-h768d8.json"
    )

    arm_qk = {
        "standard-rope": copy.deepcopy(standard["qk"]),
        "mapped-addrope-a03": copy.deepcopy(mapped["qk"]),
        "position-only": copy.deepcopy(position_only["qk"]),
        "content-position": copy.deepcopy(content_position["qk"]),
        "rope-matched-ffn": copy.deepcopy(standard["qk"]),
    }
    seeds = (123, 456, 789)
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    written = []
    for seed in seeds:
        for arm, qk in arm_qk.items():
            config = copy.deepcopy(standard)
            run_name = f"phase19-{arm}-seed{seed}-s30000-h1024d12"
            config.update(
                {
                    "seed": seed,
                    "paired_initialization_seed": seed,
                    "run_name": run_name,
                    "base_output_dir": str(OUTPUT_ROOT),
                    "qk": copy.deepcopy(qk),
                    "max_train_steps": 30_000,
                    "validate_every": 5_000,
                    "num_validation_batches": 25,
                    "validation_start_batch": 0,
                    "num_final_validation_batches": 1_024,
                    "final_validation_start_batch": 2_048,
                    "save_evaluation_details": True,
                    "save_final_model": True,
                    "checkpointing_steps": None,
                    "resume_from_checkpoint": None,
                    "with_tracking": False,
                    "profile_every_n_steps": 0,
                    "log_every_n_steps": 50,
                    "ff_hidden_dim": None,
                    "ff_widened_hidden_dim": None,
                    "ff_widened_layers": [],
                }
            )
            if arm == "rope-matched-ffn":
                # Nine 64-unit layer expansions add 1,770,624 useful FFN
                # parameters, within 3.3% of position-only's 1,714,176.
                config["ff_widened_hidden_dim"] = 4_160
                config["ff_widened_layers"] = [0, 1, 3, 4, 5, 7, 8, 9, 11]
            path = CONFIG_DIR / f"{run_name}.json"
            _write_locked(path, config, force=force)
            written.append(path)
    return written


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    paths = build_configs(force=args.force)
    print(f"locked {len(paths)} confirmation configs under {CONFIG_DIR}")


if __name__ == "__main__":
    main()
