#!/usr/bin/env python
"""Generate the phase-24 RoPE-embed-basis screen.

Question: the mapped additive carrier's position input was inherited from a
5k single-seed *efficiency* screen (`basis_dim=16` plus two scalars), and its
amplitude anchor (`0.3`) was designated a control rather than the winner in the
phase-3 sweep. Both settings rode unexamined into the phase-19 confirmation,
where the mapped arm tied the headline method.

`frozen_fourier` at native width is exactly the cached RoPE schedule, so this
screen replaces the bespoke basis with the model's own RoPE embeddings and
removes the redundant mapper composition.

Arms (one axis moves at a time, in this order):

  1. rope-fixed          standard RoPE reference
  2. basis16-a03         the phase-19 mapped arm, unchanged (continuity anchor)
  3. basis16-a10         + amplitude anchor 1.0            [anchor axis]
  4. ropeembed-a10       + full RoPE-embed basis, identity mapper [input axis]

Arm 4 is strictly smaller than arm 2 (3.17M vs 3.40M positional parameters at
h1024) while raising the achievable rank of the amplitude/phase map from 18 to
the full 64, because the input is no longer decimated to 8 frequencies.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONFIG_DIR = ROOT / "sweep_configs" / "phase24_rope_embed_basis"
OUTPUT_ROOT = ROOT / "model-output" / "position_bias_phase24_rope_embed_basis"
BASE = ROOT / "sweep_configs" / "phase20_rope_frequency" / "phase20-fixed-seed123-s5000-h768d8.json"

SEEDS = (123, 456, 789)


def mapped_qk(*, basis_dim, scalars, mapper_kind, amplitude_init):
    return {
        "enabled": True,
        "application": "additive",
        "geometry": "amplitude_phase",
        "input": {
            "kind": "frozen_fourier",
            "basis_dim": basis_dim,
            "theta": None,
            "scalars": list(scalars),
        },
        "mapper": {
            "kind": mapper_kind,
            "residual": False,
            "rank": 32,
            "hidden_dim": 128,
        },
        "qk_coupling": "shared_trunk_separate_readouts",
        "head_coupling": "per_head_independent",
        "conditioning": {"kind": "none"},
        "output": {
            "parameter_source": "mapped",
            "amplitude_init": amplitude_init,
            "amplitude_parameterization": "signed",
            "learn_amplitude": True,
            "learn_phase": True,
        },
    }


LEGACY_SCALARS = ["normalized_position", "log_position"]

ARMS = {
    "rope-fixed": {"enabled": False},
    "basis16-a03": mapped_qk(
        basis_dim=16, scalars=LEGACY_SCALARS, mapper_kind="linear", amplitude_init=0.3
    ),
    "basis16-a10": mapped_qk(
        basis_dim=16, scalars=LEGACY_SCALARS, mapper_kind="linear", amplitude_init=1.0
    ),
    # basis_dim=None resolves to head_dim: the model's own RoPE schedule, one
    # (cos, sin) pair per rotary pair. Identity mapper keeps a single linear
    # stage (the readouts) instead of two back-to-back affine maps.
    "ropeembed-a10": mapped_qk(
        basis_dim=None, scalars=[], mapper_kind="identity", amplitude_init=1.0
    ),
}


def build(force: bool = False) -> list[Path]:
    base = json.loads(BASE.read_text())
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    written = []
    for seed in SEEDS:
        for arm, qk in ARMS.items():
            cfg = copy.deepcopy(base)
            run_name = f"phase24-{arm}-seed{seed}-s5000-h768d8"
            cfg.update(
                {
                    "seed": seed,
                    "paired_initialization_seed": seed,
                    "run_name": run_name,
                    "base_output_dir": str(OUTPUT_ROOT),
                    "qk": copy.deepcopy(qk),
                    "rope_frequency": {"mode": "fixed"},
                    "with_tracking": False,
                    "checkpointing_steps": 2500,
                    "resume_from_checkpoint": "auto",
                }
            )
            cfg.pop("output_dir", None)
            path = CONFIG_DIR / f"{run_name}.json"
            rendered = json.dumps(cfg, indent=2, sort_keys=True) + "\n"
            if path.exists() and path.read_text() != rendered and not force:
                raise RuntimeError(f"refusing to change {path}; pass --force")
            path.write_text(rendered)
            written.append(path)
    return written


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    paths = build(parser.parse_args().force)
    print(f"wrote {len(paths)} configs to {CONFIG_DIR}")
