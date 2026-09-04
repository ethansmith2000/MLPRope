#!/usr/bin/env python
"""Claimed-GPU smoke for the consolidated position mechanisms."""

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


def additive_qk(
    *,
    geometry: str = "amplitude_phase",
    conditioning: str = "none",
    input_mode: str = "content",
    learn_amplitude: bool = True,
    learn_phase: bool = True,
) -> dict:
    """Build a small canonical additive Q/K configuration."""
    return normalize_position_config_v2(
        "qk",
        {
            "enabled": True,
            "application": "additive",
            "geometry": geometry,
            "input": {
                "kind": "frozen_fourier",
                "basis_dim": 8,
                "theta": None,
                "scalars": ["normalized_position", "log_position"],
            },
            "mapper": {
                "kind": "linear",
                "residual": False,
                "rank": 4,
                "hidden_dim": 16,
            },
            "output": {
                "parameter_source": "direct"
                if geometry == "amplitude_phase"
                else "mapped",
                "amplitude_init": 1.0,
                "amplitude_parameterization": "signed",
                "learn_amplitude": learn_amplitude,
                "learn_phase": learn_phase,
            },
            "conditioning": {
                "kind": conditioning,
                "source": "dedicated",
                "target": "both",
                "coupling": "shared_trunk_separate_readouts",
                "static_complement": False,
                "input_mode": input_mode,
                "input_normalization": "modality_rms"
                if conditioning == "carrier_hypernetwork"
                else "none",
                "learnable_input_gains": conditioning == "carrier_hypernetwork",
                "network": "silu_mlp",
                "components": "amplitude_phase",
                "head_coupling": "per_head_independent",
                "phase_bound": 0.25,
                "hidden_dim": 16,
            },
            "qk_coupling": "shared_trunk_separate_readouts",
            "head_coupling": "per_head_independent",
        },
        model_dim=64,
        heads=4,
        rope_theta=10_000.0,
    )


def preprojection(mode: str, *, smooth_rank: int = 4) -> dict:
    return {
        "enabled": True,
        "mode": mode,
        "basis_dim": 64,
        "gate_init": 1.0,
        "learnable_gate": True,
        "smooth_rank": smooth_rank,
    }


CASES = {
    "fixed_rope": {},
    "no_explicit_position": {"use_rope": False},
    "addrope_static_qk_adapter": {
        "qk_config": additive_qk(),
        "qk_norm_mode": "method_aware_rms",
    },
    "addrope_static_qk_adapter_no_rope": {
        "use_rope": False,
        "qk_config": additive_qk(),
        "qk_norm_mode": "method_aware_rms",
    },
    "addrope_dynamic_pointwise": {
        "qk_config": additive_qk(
            conditioning="carrier_hypernetwork",
            input_mode="content_position",
            learn_amplitude=False,
            learn_phase=False,
        ),
        "qk_norm_mode": "method_aware_rms",
    },
    "preprojection_rope": {
        "qk_preprojection_config": preprojection("tied_scalar"),
    },
    "preprojection_no_rope": {
        "use_rope": False,
        "qk_preprojection_config": preprojection("tied_scalar"),
    },
    "preprojection_tied_smooth_amplitude_rope": {
        "qk_preprojection_config": preprojection("tied_smooth_amplitude"),
    },
    "preprojection_tied_smooth_amplitude_no_rope": {
        "use_rope": False,
        "qk_preprojection_config": preprojection("tied_smooth_amplitude"),
    },
    "preprojection_addrope": {
        "qk_config": additive_qk(),
        "qk_preprojection_config": preprojection("tied_scalar"),
        "qk_norm_mode": "method_aware_rms",
    },
}


def run_case(name: str, overrides: dict, *, compile_model: bool) -> None:
    device = torch.device("cuda")
    kwargs = {
        "dim": 64,
        "depth": 2,
        "heads": 4,
        "ff_mult": 2,
        "vocab_size": 128,
        "max_seq_len": 64,
        "qk_config": {"enabled": False},
        "logit_bias_config": {"enabled": False},
        "attn_impl": "sdpa",
    }
    kwargs.update(overrides)
    model = Transformer(**kwargs).to(device=device, dtype=torch.bfloat16).train()
    if compile_model:
        # Each case intentionally changes module structure. Reset Dynamo so the
        # suite tests every structure instead of hitting the per-code-object
        # recompile limit and silently falling back on later cases.
        torch.compiler.reset()
        model = torch.compile(model, mode="default", fullgraph=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    input_ids = torch.randint(0, 128, (2, 32), device=device)
    targets = torch.randint(0, 128, (2, 32), device=device)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = model(input_ids=input_ids, targets=targets)
        if not torch.isfinite(loss).item():
            raise RuntimeError(f"{name}: non-finite loss")
        loss.backward()
        for parameter in model.parameters():
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                raise RuntimeError(f"{name}: non-finite gradient")
        optimizer.step()
    print(
        f"OK {name} compile={compile_model} "
        f"loss={float(loss.detach().float()):.4f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eager-only", action="store_true")
    parser.add_argument("--case", action="append", default=[])
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for position_v2_cuda_smoke")

    selected = set(args.case) if args.case else set(CASES)
    unknown = selected.difference(CASES)
    if unknown:
        raise SystemExit(f"Unknown cases: {sorted(unknown)}")
    modes = (False,) if args.eager_only else (False, True)
    for compile_model in modes:
        for name, overrides in CASES.items():
            if name in selected:
                run_case(name, overrides, compile_model=compile_model)
    print("ALL_SMOKE_OK")


if __name__ == "__main__":
    main()
