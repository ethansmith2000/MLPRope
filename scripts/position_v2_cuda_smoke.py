#!/usr/bin/env python
"""Claimed-GPU smoke for position v2: eager + compiled forward/backward steps."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from position import normalize_position_config_v2
from transformer import Transformer


def _qk(
    *,
    application: str,
    qk_coupling: str,
    mapper_kind: str = "mlp",
    head_coupling: str = "per_head_independent",
) -> dict:
    residual = mapper_kind in {"low_rank", "bottleneck_mlp", "mlp"}
    return normalize_position_config_v2(
        "qk",
        {
            "enabled": True,
            "application": application,
            "geometry": "free" if application == "additive" else "phase",
            "input": {
                "kind": "frozen_fourier",
                "basis_dim": None,
                "theta": None,
                "scalars": [],
            },
            "mapper": {
                "kind": mapper_kind,
                "residual": residual,
                "rank": 4,
                "hidden_dim": 12,
            },
            "qk_coupling": qk_coupling,
            "head_coupling": head_coupling,
        },
        model_dim=64,
        heads=4,
        rope_theta=10_000.0,
    )


def _logit() -> dict:
    return normalize_position_config_v2(
        "logit_bias",
        {
            "enabled": True,
            "application": "logit_bias",
            "geometry": "scalar_curve",
            "input": {
                "kind": "frozen_fourier",
                "basis_dim": None,
                "theta": None,
                "scalars": [],
            },
            "mapper": {
                "kind": "linear",
                "residual": False,
                "rank": 4,
                "hidden_dim": 12,
            },
            "head_coupling": "per_head_independent",
        },
        model_dim=64,
        heads=4,
        rope_theta=10_000.0,
    )


CASES = [
    ("rope_baseline", {"enabled": False}, {"enabled": False}, "sdpa"),
    ("additive_shared", _qk(application="additive", qk_coupling="shared"), {"enabled": False}, "sdpa"),
    ("phase_shared", _qk(application="rotary", qk_coupling="shared"), {"enabled": False}, "sdpa"),
    (
        "additive_separate_readout",
        _qk(application="additive", qk_coupling="shared_trunk_separate_readouts"),
        {"enabled": False},
        "sdpa",
    ),
    (
        "phase_separate_readout",
        _qk(application="rotary", qk_coupling="shared_trunk_separate_readouts"),
        {"enabled": False},
        "sdpa",
    ),
    (
        "additive_separate",
        _qk(application="additive", qk_coupling="separate"),
        {"enabled": False},
        "sdpa",
    ),
    (
        "phase_separate",
        _qk(application="rotary", qk_coupling="separate"),
        {"enabled": False},
        "sdpa",
    ),
    ("logit_flex", {"enabled": False}, _logit(), "flex"),
    (
        "combined_flex",
        _qk(application="rotary", qk_coupling="shared", mapper_kind="linear"),
        _logit(),
        "flex",
    ),
]


def run_case(name: str, qk: dict, logit: dict, attn_impl: str, *, compile_model: bool) -> None:
    device = torch.device("cuda")
    model = Transformer(
        dim=64,
        depth=2,
        heads=4,
        ff_mult=2,
        vocab_size=128,
        max_seq_len=64,
        qk_config=qk,
        logit_bias_config=logit,
        attn_impl=attn_impl,
        rel_extent=64,
    ).to(device=device, dtype=torch.bfloat16)
    model.train()
    if attn_impl == "flex":
        # Training uses a causal shift: queries see length-1 vs full block.
        model.prepare_flex_masks(query_length=31, device=device)
    if compile_model:
        model = torch.compile(model, mode="default", fullgraph=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    input_ids = torch.randint(0, 128, (2, 32), device=device)
    targets = torch.randint(0, 128, (2, 32), device=device)
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = model(input_ids=input_ids, targets=targets)
        if not torch.isfinite(loss).item():
            raise RuntimeError(f"{name}: non-finite loss under compile={compile_model}")
        loss.backward()
        for parameter in model.parameters():
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                raise RuntimeError(
                    f"{name}: non-finite grad under compile={compile_model}"
                )
        optimizer.step()
    print(f"OK {name} compile={compile_model} loss={float(loss.detach().float()):.4f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eager-only", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for position_v2_cuda_smoke")

    modes = [False] if args.eager_only else [False, True]
    for compile_model in modes:
        for name, qk, logit, attn_impl in CASES:
            run_case(name, qk, logit, attn_impl, compile_model=compile_model)
    print("ALL_SMOKE_OK")


if __name__ == "__main__":
    main()
