#!/usr/bin/env python
"""Paper mechanism measurements on retained Phase-49/50 checkpoints.

This is inference-only. It writes compact summaries plus compressed per-block
arrays and never modifies the source run or saves model weights.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import datasets
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import default_data_collator


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train_gpt import make_model


RESULT_ROOT = ROOT / "results" / "phase53_paper_mechanism"
PHASE49_ROOT = ROOT / "model-output" / "position_bias_phase49_mature_qk_readout"
PHASE50_ROOT = ROOT / "model-output" / "position_bias_phase50_training_seed_replication"
ARMS = ("rope", "scalar-qkpre", "qk-readout-r32")
SEEDS = (123, 456, 789)
HOLDOUT_START = 4_096
LOSS_BLOCKS = 1_024
ATTENTION_BLOCKS = 64
ATTENTION_OFFSET = 8
ATTENTION_STRIDE = 16
POSITION_BIN_EDGES = (1, 16, 32, 64, 128, 256, 512, 1_024)
ATTENTION_METRIC_NAMES = (
    "entropy_normalized",
    "attended_distance_tokens",
    "attended_distance_fraction",
    "first_token_mass",
    "distance_mass_0",
    "distance_mass_1_3",
    "distance_mass_4_15",
    "distance_mass_16_63",
    "distance_mass_64_255",
    "distance_mass_256_plus",
    "query_corr_entropy",
    "query_corr_distance_fraction",
    "query_corr_first_token_mass",
    "logit_cc_centered_rms",
    "logit_cp_centered_rms",
    "logit_pc_centered_rms",
    "logit_pp_centered_rms",
    "logit_position_total_centered_rms",
    "logit_total_centered_rms",
    "logit_position_total_cosine",
    "attention_kl_full_vs_content_only",
)


def _run_dir(seed: int, arm: str) -> Path:
    if seed == 123:
        return PHASE49_ROOT / f"phase49-{arm}-seed123-b32-s100000-h768d8"
    return PHASE50_ROOT / f"phase50-{arm}-seed{seed}-b32-s100000-h768d8"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _correlation_with_query_position(value: torch.Tensor, start: int = 16) -> torch.Tensor:
    """Pearson correlation over queries, independently for each head."""
    start = min(start, max(1, value.shape[-1] // 4))
    value = value[..., start:]
    query = torch.arange(
        start,
        start + value.shape[-1],
        device=value.device,
        dtype=value.dtype,
    )
    query = query - query.mean()
    query = query / query.square().mean().sqrt().clamp_min(1e-12)
    centered = value - value.mean(dim=-1, keepdim=True)
    denominator = centered.square().mean(dim=-1).sqrt().clamp_min(1e-12)
    return (centered * query).mean(dim=-1) / denominator


def _normalize_components(
    norm: torch.nn.RMSNorm,
    content: torch.Tensor,
    position: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    total = content + position
    inverse_rms = torch.rsqrt(
        total.float().square().mean(dim=-1, keepdim=True) + float(norm.eps)
    )
    weight = norm.weight.detach().float().view(1, 1, 1, -1)
    content_normalized = content.float() * inverse_rms * weight
    position_normalized = position.float() * inverse_rms * weight
    reconstructed = content_normalized + position_normalized
    expected = norm(total.float()).float()
    torch.testing.assert_close(reconstructed, expected, rtol=2e-5, atol=2e-5)
    return content_normalized, position_normalized


def _qk_components(attention, x: torch.Tensor):
    """Return exact content/position Q/K pieces after QKNorm and RoPE."""
    if attention.qk_position is not None:
        raise ValueError("Phase-53 supports the frozen pre-Q/K carrier family only")
    if attention.qk_norm_mode != "method_aware_rms":
        raise ValueError("Phase-53 decomposition requires method-aware RMS QKNorm")
    if attention.post_position_qk_norm:
        raise ValueError("Unexpected post-position Q/K normalization")
    if not attention.multiplicative_rope:
        raise ValueError("Phase-53 expects standard RoPE")
    if attention.to_q.bias is not None or attention.to_k.bias is not None:
        raise ValueError("Phase-53 retained checkpoints must have bias-free Q/K")

    x = x.detach().float()
    q_content = attention._split_heads(attention.to_q(x))
    k_content = attention._split_heads(attention.to_k(x))
    q_position = torch.zeros_like(q_content)
    k_position = torch.zeros_like(k_content)
    if attention.qk_preprojection is not None:
        positional = attention.qk_preprojection(x.shape[1], dtype=x.dtype)
        if positional.q_input is not None:
            q_position = q_position + attention._split_heads(
                attention.to_q(positional.q_input[None])
            )
        if positional.k_input is not None:
            k_position = k_position + attention._split_heads(
                attention.to_k(positional.k_input[None])
            )
        if positional.q_projected is not None:
            q_position = q_position + attention._split_heads(
                positional.q_projected[None]
            )
        if positional.k_projected is not None:
            k_position = k_position + attention._split_heads(
                positional.k_projected[None]
            )

    q_full = attention.q_norm(q_content + q_position)
    k_full = attention.k_norm(k_content + k_position)
    q_content, q_position = _normalize_components(
        attention.q_norm, q_content, q_position
    )
    k_content, k_position = _normalize_components(
        attention.k_norm, k_content, k_position
    )
    q_full, k_full = attention._apply_rope(q_full, k_full)
    q_content, k_content = attention._apply_rope(q_content, k_content)
    q_position, k_position = attention._apply_rope(q_position, k_position)
    torch.testing.assert_close(
        q_content + q_position, q_full, rtol=3e-5, atol=3e-5
    )
    torch.testing.assert_close(
        k_content + k_position, k_full, rtol=3e-5, atol=3e-5
    )
    return q_content, q_position, k_content, k_position, q_full, k_full


def _query_centered_stats(
    component: torch.Tensor,
    total_centered: torch.Tensor,
    visible: torch.Tensor,
    visible_count: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    masked = component.masked_fill(~visible, 0.0)
    row_mean = masked.sum(dim=-1) / visible_count
    centered = (component - row_mean[..., None]).masked_fill(~visible, 0.0)
    row_rms = (centered.square().sum(dim=-1) / visible_count).sqrt()
    total_rms = (total_centered.square().sum(dim=-1) / visible_count).sqrt()
    dot = (centered * total_centered).sum(dim=-1) / visible_count
    cosine = dot / (row_rms * total_rms).clamp_min(1e-12)
    cosine = torch.where(row_rms > 1e-12, cosine, torch.zeros_like(cosine))
    return row_rms[..., 1:].mean(dim=-1), cosine[..., 1:].mean(dim=-1)


@torch.no_grad()
def _attention_metrics(attention, x: torch.Tensor) -> tuple[np.ndarray, float]:
    qc, qp, kc, kp, q_full, k_full = _qk_components(attention, x)
    scale = attention.head_dim ** -0.5
    cc = torch.matmul(qc, kc.transpose(-1, -2)) * scale
    cp = torch.matmul(qc, kp.transpose(-1, -2)) * scale
    pc = torch.matmul(qp, kc.transpose(-1, -2)) * scale
    pp = torch.matmul(qp, kp.transpose(-1, -2)) * scale
    component_sum = cc + cp + pc + pp
    total = torch.matmul(q_full, k_full.transpose(-1, -2)) * scale
    length = total.shape[-1]
    visible_2d = torch.ones(
        length, length, dtype=torch.bool, device=total.device
    ).tril_()
    visible = visible_2d.view(1, 1, length, length)
    visible_count = torch.arange(
        1, length + 1, dtype=total.dtype, device=total.device
    ).view(1, 1, length)
    masked_total = total.masked_fill(~visible, float("-inf"))
    probability = torch.softmax(masked_total, dim=-1)

    entropy = -(probability * probability.clamp_min(1e-30).log()).sum(dim=-1)
    entropy_normalized = entropy[..., 1:] / visible_count[..., 1:].log()
    query = torch.arange(length, dtype=total.dtype, device=total.device)
    key = torch.arange(length, dtype=total.dtype, device=total.device)
    distance = (query[:, None] - key[None, :]).clamp_min(0)
    expected_distance = (probability * distance).sum(dim=-1)
    distance_fraction = expected_distance[..., 1:] / query[1:]
    first_token_mass = probability[..., 0]

    metric_values: dict[str, torch.Tensor] = {
        "entropy_normalized": entropy_normalized.mean(dim=-1).squeeze(0),
        "attended_distance_tokens": expected_distance[..., 1:].mean(dim=-1).squeeze(0),
        "attended_distance_fraction": distance_fraction.mean(dim=-1).squeeze(0),
        "first_token_mass": first_token_mass[..., 1:].mean(dim=-1).squeeze(0),
    }
    distance_bins = (
        ("distance_mass_0", distance == 0),
        ("distance_mass_1_3", (distance >= 1) & (distance <= 3)),
        ("distance_mass_4_15", (distance >= 4) & (distance <= 15)),
        ("distance_mass_16_63", (distance >= 16) & (distance <= 63)),
        ("distance_mass_64_255", (distance >= 64) & (distance <= 255)),
        ("distance_mass_256_plus", distance >= 256),
    )
    for name, mask in distance_bins:
        mass = (probability * mask).sum(dim=-1)
        metric_values[name] = mass[..., 1:].mean(dim=-1).squeeze(0)

    metric_values["query_corr_entropy"] = _correlation_with_query_position(
        entropy.squeeze(0)
    )
    distance_fraction_all = expected_distance / query.clamp_min(1)
    metric_values["query_corr_distance_fraction"] = _correlation_with_query_position(
        distance_fraction_all.squeeze(0)
    )
    metric_values["query_corr_first_token_mass"] = _correlation_with_query_position(
        first_token_mass.squeeze(0)
    )

    total_masked = total.masked_fill(~visible, 0.0)
    total_row_mean = total_masked.sum(dim=-1) / visible_count
    total_centered = (total - total_row_mean[..., None]).masked_fill(~visible, 0.0)
    component_map = {
        "logit_cc_centered_rms": cc,
        "logit_cp_centered_rms": cp,
        "logit_pc_centered_rms": pc,
        "logit_pp_centered_rms": pp,
        "logit_position_total_centered_rms": cp + pc + pp,
        "logit_total_centered_rms": total,
    }
    position_cosine = None
    for name, component in component_map.items():
        rms, cosine = _query_centered_stats(
            component, total_centered, visible, visible_count
        )
        metric_values[name] = rms.squeeze(0)
        if name == "logit_position_total_centered_rms":
            position_cosine = cosine.squeeze(0)
    metric_values["logit_position_total_cosine"] = position_cosine

    content_probability = torch.softmax(
        cc.masked_fill(~visible, float("-inf")), dim=-1
    )
    kl = (
        probability
        * (
            probability.clamp_min(1e-30).log()
            - content_probability.clamp_min(1e-30).log()
        )
    ).sum(dim=-1)
    metric_values["attention_kl_full_vs_content_only"] = (
        kl[..., 1:].mean(dim=-1).squeeze(0)
    )

    matrix = torch.stack(
        [metric_values[name] for name in ATTENTION_METRIC_NAMES], dim=-1
    )
    reconstruction_error = float(
        (total - component_sum).abs().max().item()
    )
    return matrix.float().cpu().numpy(), reconstruction_error


def _position_bin_slices(length: int) -> list[slice]:
    if length != POSITION_BIN_EDGES[-1] - 1:
        raise ValueError(f"Expected {POSITION_BIN_EDGES[-1] - 1} targets, got {length}")
    return [
        slice(start - 1, stop - 1)
        for start, stop in zip(POSITION_BIN_EDGES[:-1], POSITION_BIN_EDGES[1:])
    ]


@torch.inference_mode()
def evaluate(model, loader, *, device: torch.device) -> dict:
    selected_offsets = {
        ATTENTION_OFFSET + ATTENTION_STRIDE * index
        for index in range(ATTENTION_BLOCKS)
    }
    if max(selected_offsets) >= LOSS_BLOCKS:
        raise ValueError("Attention sampling exceeds loss-evaluation window")
    captured: list[torch.Tensor] = []
    capture_enabled = False

    def capture(_module, inputs):
        if capture_enabled:
            captured.append(inputs[0].detach())

    hooks = [block.attn.register_forward_pre_hook(capture) for block in model.blocks]
    position_loss_sum = None
    per_block_mean_loss = []
    per_block_position_bin_loss = []
    attention_samples = []
    attention_offsets = []
    maximum_reconstruction_error = 0.0
    started = time.perf_counter()
    try:
        for offset, batch in enumerate(loader):
            capture_enabled = offset in selected_offsets
            captured.clear()
            tokens = batch["input_ids"].to(device, non_blocking=True)
            inputs = tokens[:, :-1]
            targets = tokens[:, 1:]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(input_ids=inputs)
            token_loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]).float(),
                targets.reshape(-1),
                reduction="none",
            ).reshape_as(targets)
            loss_row = token_loss[0].double().cpu()
            if position_loss_sum is None:
                position_loss_sum = torch.zeros_like(loss_row)
            position_loss_sum += loss_row
            per_block_mean_loss.append(float(loss_row.mean().item()))
            per_block_position_bin_loss.append(
                [float(loss_row[part].mean().item()) for part in _position_bin_slices(loss_row.numel())]
            )
            if capture_enabled:
                if len(captured) != len(model.blocks):
                    raise RuntimeError(
                        f"Captured {len(captured)} layers, expected {len(model.blocks)}"
                    )
                layer_metrics = []
                for block, hidden in zip(model.blocks, captured, strict=True):
                    metrics, error = _attention_metrics(block.attn, hidden)
                    layer_metrics.append(metrics)
                    maximum_reconstruction_error = max(
                        maximum_reconstruction_error, error
                    )
                attention_samples.append(np.stack(layer_metrics))
                attention_offsets.append(offset)
            del tokens, inputs, targets, logits, token_loss
            if offset + 1 >= LOSS_BLOCKS:
                break
    finally:
        for hook in hooks:
            hook.remove()
    torch.cuda.synchronize(device)
    if len(per_block_mean_loss) != LOSS_BLOCKS:
        raise ValueError(
            f"Expected {LOSS_BLOCKS} loss blocks, got {len(per_block_mean_loss)}"
        )
    if attention_offsets != sorted(selected_offsets):
        raise ValueError("Attention sample offsets do not match the frozen design")
    arrays = {
        "per_block_mean_loss": np.asarray(per_block_mean_loss, dtype=np.float64),
        "position_loss_sum": position_loss_sum.numpy(),
        "per_block_position_bin_loss": np.asarray(
            per_block_position_bin_loss, dtype=np.float64
        ),
        "attention_metrics": np.asarray(attention_samples, dtype=np.float32),
        "attention_offsets": np.asarray(attention_offsets, dtype=np.int64),
    }
    for name, value in arrays.items():
        if not np.isfinite(value).all():
            raise ValueError(f"Non-finite values in {name}")
    return {
        "arrays": arrays,
        "elapsed_seconds": time.perf_counter() - started,
        "maximum_logit_reconstruction_error": maximum_reconstruction_error,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True, choices=SEEDS)
    parser.add_argument("--arm", required=True, choices=ARMS)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Phase-53 mechanism analysis requires a visible CUDA GPU")

    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    stem = f"seed{args.seed}_{args.arm}"
    output_json = RESULT_ROOT / f"{stem}.json"
    output_npz = RESULT_ROOT / f"{stem}.npz"
    if output_json.is_file() and output_npz.is_file() and not args.overwrite:
        print(f"Already complete: {stem}")
        return

    run_dir = _run_dir(args.seed, args.arm)
    config_path = run_dir / "training_config.json"
    weights_path = run_dir / "pytorch_model.bin"
    if not (run_dir / "COMPLETED").is_file():
        raise RuntimeError(f"Source run is incomplete: {run_dir}")
    config = json.loads(config_path.read_text())
    state = torch.load(weights_path, map_location="cpu", weights_only=True, mmap=True)
    vocab_size = int(state["token_embedding.weight"].shape[0])
    model = make_model(SimpleNamespace(**config), vocab_size)
    model.load_state_dict(state, strict=True)
    del state
    model.requires_grad_(False).eval()
    device = torch.device("cuda", 0)
    model.to(device)

    validation = datasets.load_from_disk(config["tokenized_dataset_path"])["validation"]
    stop = HOLDOUT_START + LOSS_BLOCKS
    if stop > len(validation):
        raise ValueError(f"Evaluation window exceeds validation data: {stop} > {len(validation)}")
    evaluation = validation.select(range(HOLDOUT_START, stop))
    loader = DataLoader(
        evaluation,
        batch_size=1,
        shuffle=False,
        collate_fn=default_data_collator,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=bool(args.num_workers),
    )
    result = evaluate(model, loader, device=device)
    arrays = result.pop("arrays")
    temporary_npz = output_npz.with_suffix(".npz.tmp")
    with temporary_npz.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary_npz.replace(output_npz)

    attention_mean = arrays["attention_metrics"].mean(axis=(0, 1, 2))
    position_bin_mean = arrays["per_block_position_bin_loss"].mean(axis=0)
    payload = {
        "scope": "phase53_paper_mechanism",
        "arm": args.arm,
        "seed": args.seed,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "source_run": str(run_dir.relative_to(ROOT)),
        "source_config_sha256": _sha256(config_path),
        "source_weights_sha256": _sha256(weights_path),
        "analysis_script_sha256": _sha256(Path(__file__)),
        "array_file": output_npz.name,
        "array_file_sha256": _sha256(output_npz),
        "holdout": {"start_batch": HOLDOUT_START, "blocks": LOSS_BLOCKS},
        "attention_sample": {
            "blocks": ATTENTION_BLOCKS,
            "offset": ATTENTION_OFFSET,
            "stride": ATTENTION_STRIDE,
            "absolute_batches": [HOLDOUT_START + int(x) for x in arrays["attention_offsets"]],
        },
        "position_bins": [
            {"target_start": start, "target_stop_exclusive": stop}
            for start, stop in zip(POSITION_BIN_EDGES[:-1], POSITION_BIN_EDGES[1:])
        ],
        "mean_loss": float(arrays["per_block_mean_loss"].mean()),
        "position_bin_mean_loss": [float(x) for x in position_bin_mean],
        "attention_metric_names": list(ATTENTION_METRIC_NAMES),
        "attention_global_means": {
            name: float(value)
            for name, value in zip(ATTENTION_METRIC_NAMES, attention_mean, strict=True)
        },
        **result,
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "gpu": torch.cuda.get_device_name(device),
        "artifact_policy": {
            "source_weights_read_only": True,
            "new_weights_or_checkpoints": False,
            "compressed_per_block_arrays": True,
        },
    }
    temporary_json = output_json.with_suffix(".json.tmp")
    temporary_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary_json.replace(output_json)
    print(
        f"{stem}: NLL={payload['mean_loss']:.6f} "
        f"attention_blocks={ATTENTION_BLOCKS} elapsed={payload['elapsed_seconds']:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
