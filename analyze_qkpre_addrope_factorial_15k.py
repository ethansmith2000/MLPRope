#!/usr/bin/env python
"""Analyze the paired Phase-30 15k qkpre x AddRoPE factorial."""

from __future__ import annotations

import analyze_qkpre_addrope_factorial as factorial


factorial.PHASE = 30
factorial.STEPS = 15_000
factorial.CONFIG_ROOT = (
    factorial.ROOT / "sweep_configs" / "phase30_qkpre_addrope_factorial_15k"
)
factorial.OUTPUT_ROOT = (
    factorial.ROOT
    / "model-output"
    / "position_bias_phase30_qkpre_addrope_factorial_15k"
)
factorial.RESULT_ROOT = (
    factorial.ROOT / "results" / "phase30_qkpre_addrope_factorial_15k"
)


if __name__ == "__main__":
    factorial.main()
