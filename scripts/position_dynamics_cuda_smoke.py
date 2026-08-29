#!/usr/bin/env python
"""Claimed-GPU smoke for the active dynamic-position training paths."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from position import (
    RotaryClockController,
    build_rope_frequencies,
    normalize_rotary_clock_config,
)
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
    "position_gain_qk": {
        "position_gain_config": {
            "enabled": True,
            "target": "both",
            "basis_dim": 16,
            "scalars": ["normalized_position", "log_position"],
            "mapper": "linear",
            "log_gain_bound": 1.0,
        },
    },
    "qk_preprojection": {
        "use_rope": False,
        "qk_preprojection_config": {
            "enabled": True,
            "gate_init": 1.0,
            "learnable_gate": True,
        },
    },
    "addrope": {
        "qk_config": {
            "enabled": True,
            "feature_map": "mlp",
            "sharing": "per_head",
            "apply": "add",
            "rank": 16,
            "mlp_hidden": 32,
        },
    },
    "qk_preprojection_addrope": {
        "qk_config": {
            "enabled": True,
            "feature_map": "mlp",
            "sharing": "per_head",
            "apply": "add",
            "rank": 16,
            "mlp_hidden": 32,
        },
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


class LongCausalClockScan(torch.nn.Module):
    """Regression fixture for the length-1024 compiled SplitScan path."""

    def __init__(self) -> None:
        super().__init__()
        config = normalize_rotary_clock_config(
            {
                "enabled": True,
                "head_coupling": "per_head",
                "mapper": "low_rank_silu",
                "rank": 16,
                "temporal": "causal_conv",
                "kernel_size": 4,
                "speed_bound": 0.2,
            }
        )
        self.clock = RotaryClockController(
            model_dim=64,
            heads=8,
            pair_dim=4,
            inverse_frequency=build_rope_frequencies(8, 10_000.0),
            config=config,
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.clock.phase_delta(values)


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

    baseline = Transformer(**COMMON).to(device).eval()
    gain_anchor = Transformer(
        **COMMON,
        **CASES["position_gain_qk"],
    ).to(device).eval()
    gain_anchor.load_state_dict(baseline.state_dict(), strict=False)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        torch.testing.assert_close(
            gain_anchor(input_ids),
            baseline(input_ids),
            atol=0,
            rtol=0,
        )
    del baseline, gain_anchor

    long_values = torch.randn(8, 1_024, 64, device=device, requires_grad=True)
    long_clock = LongCausalClockScan().to(device).train()
    compiled_long_clock = torch.compile(long_clock, mode="default", fullgraph=False)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        long_phase = compiled_long_clock(long_values)
        long_loss = long_phase.float().square().mean()
    long_loss.backward()
    long_gradient = long_clock.clock.controller.output.weight.grad
    if long_gradient is None or not torch.isfinite(long_gradient).all():
        raise RuntimeError("length-1024 causal clock scan produced a bad gradient")
    print("rotary_clock_causal_conv_length1024: compiled forward/backward ok")
    del compiled_long_clock, long_clock, long_values, long_phase, long_loss
    torch.compiler.reset()

    for name, overrides in CASES.items():
        model = Transformer(**COMMON, **overrides).to(device).train()
        compiled = torch.compile(model, mode="default", fullgraph=False)
        first = train_step(compiled, input_ids)
        second = train_step(compiled, input_ids)
        if not torch.isfinite(torch.tensor([first, second])).all():
            raise RuntimeError(f"{name} produced a non-finite loss")

        if name == "position_gain_qk":
            gradient = model.blocks[0].attn.position_gain.q_readout.weight.grad
        elif name == "qk_preprojection":
            gradient = model.blocks[0].attn.qk_preprojection.gate.grad
        elif name == "addrope":
            gradients = [
                parameter.grad
                for parameter in model.blocks[0].attn.qk_position.parameters()
                if parameter.requires_grad
            ]
            gradient = torch.cat(
                [item.reshape(-1) for item in gradients if item is not None]
            )
        elif name == "qk_preprojection_addrope":
            gradients = [model.blocks[0].attn.qk_preprojection.gate.grad]
            gradients.extend(
                parameter.grad
                for parameter in model.blocks[0].attn.qk_position.parameters()
                if parameter.requires_grad
            )
            gradient = torch.cat(
                [item.reshape(-1) for item in gradients if item is not None]
            )
        else:
            gradient = model.blocks[0].attn.rotary_clock.controller.output.weight.grad
        if gradient is None or not torch.isfinite(gradient).all():
            raise RuntimeError(f"{name} did not produce a finite controller gradient")
        print(f"{name}: loss {first:.6f} -> {second:.6f}; compiled backward ok")
        del compiled, model
        torch.compiler.reset()


if __name__ == "__main__":
    main()
