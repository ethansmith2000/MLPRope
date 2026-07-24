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
    geometry: str | None = None,
    basis_kind: str = "frozen_fourier",
    conditioning: str = "none",
    scalars: list[str] | None = None,
) -> dict:
    residual = mapper_kind in {"low_rank", "bottleneck_mlp", "mlp"}
    return normalize_position_config_v2(
        "qk",
        {
            "enabled": True,
            "application": application,
            "geometry": geometry
            or ("free" if application == "additive" else "phase"),
            "input": {
                "kind": basis_kind,
                "basis_dim": None,
                "theta": None,
                "scalars": list(scalars or []),
            },
            "mapper": {
                "kind": mapper_kind,
                "residual": residual,
                "rank": 4,
                "hidden_dim": 12,
            },
            "qk_coupling": qk_coupling,
            "head_coupling": head_coupling,
            "conditioning": {
                "kind": conditioning,
                "hidden_dim": 12,
            },
        },
        model_dim=64,
        heads=4,
        rope_theta=10_000.0,
    )


def _logit(conditioning: str = "none") -> dict:
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
            "conditioning": {
                "kind": conditioning,
                "num_profiles": 4,
                "router_hidden_dim": 8,
                "num_frequencies": 4,
                "gate_init": 0.0,
            },
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

NEW_CASES = [
    (
        "canonical_addrope",
        _qk(
            application="additive",
            geometry="amplitude_phase",
            qk_coupling="shared_trunk_separate_readouts",
        ),
        {"enabled": False},
        "sdpa",
        None,
        None,
    ),
    (
        "canonical_addrope_content",
        _qk(
            application="additive",
            geometry="amplitude_phase",
            qk_coupling="shared_trunk_separate_readouts",
            conditioning="local_residual",
        ),
        {"enabled": False},
        "sdpa",
        None,
        None,
    ),
    (
        "projected_learned_frequency",
        _qk(
            application="rotary",
            geometry="projected_phase",
            basis_kind="learned_frequency_fourier",
            qk_coupling="shared_trunk_separate_readouts",
        ),
        {"enabled": False},
        "sdpa",
        None,
        None,
    ),
    (
        "scaled_rotary",
        _qk(
            application="rotary",
            geometry="scaled_phase",
            basis_kind="learned_temperature_fourier",
            qk_coupling="shared",
        ),
        {"enabled": False},
        "sdpa",
        None,
        None,
    ),
    (
        "content_local_qk",
        _qk(
            application="additive",
            qk_coupling="shared_trunk_separate_readouts",
            conditioning="local_residual",
        ),
        {"enabled": False},
        "sdpa",
        None,
        None,
    ),
    (
        "scalar_input_qk",
        _qk(
            application="additive",
            qk_coupling="shared_trunk_separate_readouts",
            mapper_kind="linear",
            scalars=["normalized_position", "log_position"],
        ),
        {"enabled": False},
        "sdpa",
        None,
        None,
    ),
    (
        "residual_stream",
        {"enabled": False},
        {"enabled": False},
        "sdpa",
        {
            "enabled": True,
            "placement": "both",
            "source": "position_basis",
            "gate_init": 0.0,
        },
        None,
    ),
    (
        "key_position_write",
        {"enabled": False},
        {"enabled": False},
        "sdpa",
        None,
        {
            "enabled": True,
            "mode": "key_position",
            "gate_init": 0.0,
        },
    ),
    (
        "relative_offset_write",
        {"enabled": False},
        {"enabled": False},
        "sdpa",
        None,
        {
            "enabled": True,
            "mode": "relative_offset",
            "gate_init": 0.0,
        },
    ),
    (
        "inkling_table",
        {"enabled": False},
        _logit("inkling_table"),
        "flex",
        None,
        None,
    ),
    (
        "inkling_cosnet_combined",
        _qk(
            application="additive",
            qk_coupling="shared_trunk_separate_readouts",
            mapper_kind="linear",
        ),
        _logit("inkling_cosnet"),
        "flex",
        None,
        None,
    ),
    (
        "all_channels_flex",
        _qk(
            application="additive",
            qk_coupling="shared_trunk_separate_readouts",
            mapper_kind="linear",
            conditioning="content_gate",
        ),
        _logit("inkling_table"),
        "flex",
        {
            "enabled": True,
            "placement": "per_layer",
            "source": "learned_absolute",
            "gate_init": 0.0,
            "layer_shared": True,
        },
        {
            "enabled": True,
            "mode": "relative_offset",
            "gate_init": 0.0,
        },
    ),
]


def run_case(
    name: str,
    qk: dict,
    logit: dict,
    attn_impl: str,
    residual: dict | None = None,
    write: dict | None = None,
    *,
    compile_model: bool,
) -> None:
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
        residual_stream_config=residual,
        attention_write_config=write,
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
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        help="Run only the named smoke case (repeatable).",
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for position_v2_cuda_smoke")

    modes = [False] if args.eager_only else [False, True]
    for compile_model in modes:
        for name, qk, logit, attn_impl in CASES:
            if args.case and name not in args.case:
                continue
            run_case(name, qk, logit, attn_impl, compile_model=compile_model)
        for name, qk, logit, attn_impl, residual, write in NEW_CASES:
            if args.case and name not in args.case:
                continue
            run_case(
                name,
                qk,
                logit,
                attn_impl,
                residual,
                write,
                compile_model=compile_model,
            )
    print("ALL_SMOKE_OK")


if __name__ == "__main__":
    main()
