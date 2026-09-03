#!/usr/bin/env python
"""Generate the locked Phase-30 15k qkpre x AddRoPE factorial configs."""

from __future__ import annotations

import argparse

import prepare_qkpre_addrope_factorial as factorial


factorial.PHASE = 30
factorial.STEPS = 15_000
factorial.VALIDATE_EVERY = 5_000
factorial.CHECKPOINTING_STEPS = 5_000
factorial.CONFIG_DIR = (
    factorial.ROOT / "sweep_configs" / "phase30_qkpre_addrope_factorial_15k"
)
factorial.OUTPUT_ROOT = (
    factorial.ROOT
    / "model-output"
    / "position_bias_phase30_qkpre_addrope_factorial_15k"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    paths = factorial.build(force=parser.parse_args().force)
    print(
        f"locked {len(paths)} Phase-30 factorial configs under "
        f"{factorial.CONFIG_DIR}"
    )


if __name__ == "__main__":
    main()
