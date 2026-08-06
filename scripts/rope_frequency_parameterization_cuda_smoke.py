#!/usr/bin/env python
"""Claimed-GPU eager/compiled smoke for static RoPE parameterizations."""

from __future__ import annotations

import sys
from pathlib import Path

import torch


REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from transformer import Transformer


PARAMETERIZATIONS = (
    "exp",
    "exp_full_ste",
    "softplus",
    "additive",
    "bounded_log",
)


def build_model(parameterization: str) -> Transformer:
    return Transformer(
        dim=64,
        depth=2,
        heads=4,
        ff_mult=2,
        vocab_size=128,
        max_seq_len=64,
        qk_config={"enabled": False},
        logit_bias_config={"enabled": False},
        qk_norm_mode="method_aware_rms",
        paired_initialization_seed=123,
        rope_frequency_config={
            "mode": "static",
            "head_coupling": "shared",
            "parameterization": parameterization,
            "log_bound": 1.0,
        },
    ).cuda()


def step(model: Transformer, tokens: torch.Tensor) -> float:
    model.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = model(tokens, tokens)
    loss.backward()
    for block in model.blocks:
        gradient = block.attn.rope_log_frequency_delta.grad
        if gradient is None or not torch.isfinite(gradient).all():
            raise RuntimeError("missing or nonfinite frequency gradient")
        if gradient.abs().sum().item() == 0:
            raise RuntimeError("zero frequency gradient")
    return float(loss.detach())


def build_content_model(mapper: str) -> Transformer:
    return Transformer(
        dim=64,
        depth=2,
        heads=4,
        ff_mult=2,
        vocab_size=128,
        max_seq_len=64,
        qk_config={"enabled": False},
        logit_bias_config={"enabled": False},
        qk_norm_mode="method_aware_rms",
        paired_initialization_seed=123,
        rope_frequency_config={
            "mode": "content",
            "head_coupling": "per_head",
            "parameterization": "horizon_bounded",
            "source": "normalized_residual",
            "mapper": mapper,
            "rank": 8,
            "qk_coupling": "shared",
            "phase_bound": 1.0,
            "reference_length": 64,
        },
    ).cuda()


def content_step(model: Transformer, tokens: torch.Tensor) -> float:
    model.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = model(tokens, tokens)
    loss.backward()
    for block in model.blocks:
        controller = block.attn.rope_frequency_controller
        gradient = controller.output.weight.grad
        if gradient is None or not torch.isfinite(gradient).all():
            raise RuntimeError("missing or nonfinite controller gradient")
        if gradient.abs().sum().item() == 0:
            raise RuntimeError("zero controller gradient")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            phase = controller.phase_delta(
                torch.randn(2, 64, 64, device="cuda")
            )
        if phase.dtype != torch.float32 or phase.abs().max().item() > 1.0:
            raise RuntimeError("dynamic phase dtype/bound failure")
    return float(loss.detach())


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    tokens = torch.randint(0, 128, (2, 64), device="cuda")
    for parameterization in PARAMETERIZATIONS:
        model = build_model(parameterization)
        loss = step(model, tokens)
        print(f"eager {parameterization} loss={loss:.6f}")
        del model

    for parameterization in ("exp_full_ste", "additive"):
        eager = build_model(parameterization)
        compiled = torch.compile(eager, mode="default", fullgraph=False)
        first = step(compiled, tokens)
        second = step(compiled, tokens)
        print(
            f"compiled {parameterization} first={first:.6f} "
            f"second={second:.6f}"
        )
        del compiled, eager

    for mapper in ("linear", "low_rank_linear", "low_rank_silu"):
        model = build_content_model(mapper)
        loss = content_step(model, tokens)
        print(f"content eager {mapper} loss={loss:.6f}")
        del model

    eager = build_content_model("low_rank_silu")
    compiled = torch.compile(eager, mode="default", fullgraph=False)
    first = content_step(compiled, tokens)
    second = content_step(compiled, tokens)
    print(f"content compiled low_rank_silu first={first:.6f} second={second:.6f}")
    del compiled, eager
    torch.cuda.synchronize()
    print("rope frequency parameterization CUDA smoke passed")


if __name__ == "__main__":
    main()
