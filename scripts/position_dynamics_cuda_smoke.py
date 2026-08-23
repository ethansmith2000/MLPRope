#!/usr/bin/env python
"""Claimed-GPU smoke for Q/K preprojection and rotary-clock training paths."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from transformer import Transformer


COMMON = {
    "dim": 64,
    "depth": 2,
    "heads": 4,
    "ff_mult": 2,
    "vocab_size": 256,
    "max_seq_len": 128,
    "attn_impl": "sdpa",
    "qk_config": {"enabled": False},
    "logit_bias_config": {"enabled": False},
    "qk_norm_mode": "method_aware_rms",
    "paired_initialization_seed": 20260822,
}


CASES = {
    "qk_preprojection": {
        "use_rope": False,
        "qk_preprojection_config": {
            "enabled": True,
            "gate_init": 1.0,
            "learnable_gate": True,
        },
    },
    "rotary_clock_pointwise": {
        "rotary_clock_config": {
            "enabled": True,
            "mapper": "low_rank_silu",
            "rank": 16,
            "temporal": "pointwise",
            "speed_bound": 0.2,
        },
    },
    "rotary_clock_causal_conv": {
        "rotary_clock_config": {
            "enabled": True,
            "mapper": "low_rank_silu",
            "rank": 16,
            "temporal": "causal_conv",
            "kernel_size": 4,
            "speed_bound": 0.2,
        },
    },
}


def train_step(model: Transformer, input_ids: torch.Tensor) -> float:
    targets = input_ids.roll(-1, dims=1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = model(input_ids, targets)
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    return float(loss.detach())


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; invoke this script through gpu-claim")
    device = torch.device("cuda")
    input_ids = torch.randint(0, 256, (2, 64), device=device)

    baseline = Transformer(**COMMON).to(device).eval()
    clock_anchor = Transformer(
        **COMMON,
        **CASES["rotary_clock_causal_conv"],
    ).to(device).eval()
    clock_anchor.load_state_dict(baseline.state_dict(), strict=False)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        torch.testing.assert_close(
            clock_anchor(input_ids),
            baseline(input_ids),
            atol=0,
            rtol=0,
        )
    del baseline, clock_anchor

    for name, overrides in CASES.items():
        model = Transformer(**COMMON, **overrides).to(device).train()
        compiled = torch.compile(model, mode="default", fullgraph=False)
        first = train_step(compiled, input_ids)
        second = train_step(compiled, input_ids)
        if not torch.isfinite(torch.tensor([first, second])).all():
            raise RuntimeError(f"{name} produced a non-finite loss")

        if name == "qk_preprojection":
            gradient = model.blocks[0].attn.qk_preprojection.gate.grad
        else:
            gradient = model.blocks[0].attn.rotary_clock.controller.output.weight.grad
        if gradient is None or not torch.isfinite(gradient).all():
            raise RuntimeError(f"{name} did not produce a finite controller gradient")
        print(f"{name}: loss {first:.6f} -> {second:.6f}; compiled backward ok")
        del compiled, model
        torch.compiler.reset()


if __name__ == "__main__":
    main()
