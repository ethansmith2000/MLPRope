#!/usr/bin/env python3
"""
Compute transformer parameter counts for the SNRAdam paper.

Mirrors the architecture in transformer.py:
  - Token embedding (no tied weights)
  - RMSNorm + Linear input projection
  - N x TransformerBlock:
      - Attention: Q, K, V (no bias), O (with bias), QK RMSNorm
      - GeGLU FFN: proj_in, proj_gate, proj_out (all with bias), hidden = ceil(dim*8/3) aligned to 64
      - 2 x RMSNorm (pre-attn, pre-ff)
  - RMSNorm + Linear output projection (with bias)

Usage:
    python count_params.py
    python count_params.py --dim 2048 --depth 24 --heads 16
"""

import argparse
import math


def geglu_hidden(dim, align=64):
    h = math.ceil(dim * 8 / 3)
    return math.ceil(h / align) * align


def count_params(dim, depth, heads, vocab_size=50257, use_rope=True):
    """Return (total_params, breakdown_dict)."""
    ff_hidden = geglu_hidden(dim)
    head_dim = dim // heads

    breakdown = {}

    # Token embedding
    breakdown["token_embedding"] = vocab_size * dim

    # Position embedding (only if no RoPE)
    if not use_rope:
        max_seq = 1024
        breakdown["position_embedding"] = max_seq * dim

    # Input projection: RMSNorm(dim) + Linear(dim, dim, bias=True)
    breakdown["in_proj_rmsnorm"] = dim
    breakdown["in_proj_linear"] = dim * dim + dim  # weight + bias

    # Per-block costs
    # Attention: Q, K, V are Linear(dim, dim, bias=False), O is Linear(dim, dim, bias=True)
    attn_qkv = 3 * (dim * dim)         # no bias
    attn_o = dim * dim + dim            # with bias
    attn_qk_norm = 2 * head_dim        # 2 x RMSNorm(head_dim)
    attn_total = attn_qkv + attn_o + attn_qk_norm

    # GeGLU FFN: proj_in(dim, ff_hidden), proj_gate(dim, ff_hidden), proj_out(ff_hidden, dim)
    # All have bias (nn.Linear default)
    ff_proj_in = dim * ff_hidden + ff_hidden
    ff_proj_gate = dim * ff_hidden + ff_hidden
    ff_proj_out = ff_hidden * dim + dim
    ff_total = ff_proj_in + ff_proj_gate + ff_proj_out

    # 2 x RMSNorm(dim) per block
    norms_per_block = 2 * dim

    block_total = attn_total + ff_total + norms_per_block
    breakdown["blocks"] = depth * block_total
    breakdown["blocks_detail"] = {
        "per_block": block_total,
        "attn": attn_total,
        "ff": ff_total,
        "norms": norms_per_block,
        "ff_hidden": ff_hidden,
    }

    # Output projection: RMSNorm(dim) + Linear(dim, vocab_size, bias=True)
    breakdown["out_proj_rmsnorm"] = dim
    breakdown["out_proj_linear"] = dim * vocab_size + vocab_size

    total = sum(v for k, v in breakdown.items() if k != "blocks_detail")
    return total, breakdown


def format_params(n):
    if n >= 1e9:
        return f"{n/1e9:.2f}B"
    if n >= 1e6:
        return f"{n/1e6:.1f}M"
    if n >= 1e3:
        return f"{n/1e3:.1f}K"
    return str(n)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=None)
    parser.add_argument("--depth", type=int, default=None)
    parser.add_argument("--heads", type=int, default=None)
    parser.add_argument("--vocab", type=int, default=50257)
    args = parser.parse_args()

    configs = [
        (1024, 12, 8),
        (2048, 12, 16),
        (2048, 24, 16),
        (2048, 36, 16),
        (3072, 12, 24),
        (3072, 24, 24),
        (3072, 36, 24),
    ]

    if args.dim is not None:
        configs = [(args.dim, args.depth or 12, args.heads or (args.dim // 128))]

    print(f"{'Config':<20} {'Params':>12} {'Blocks':>12} {'Embed+Head':>14} {'FF Hidden':>10}")
    print("-" * 70)

    for dim, depth, heads in configs:
        total, bd = count_params(dim, depth, heads, args.vocab)
        embed_head = bd["token_embedding"] + bd["out_proj_linear"] + bd["out_proj_rmsnorm"]
        detail = bd["blocks_detail"]
        print(
            f"{dim}x{depth}x{heads:<10} "
            f"{format_params(total):>12} "
            f"{format_params(bd['blocks']):>12} "
            f"{format_params(embed_head):>14} "
            f"{detail['ff_hidden']:>10}"
        )

    # Print detailed breakdown for the first config
    dim, depth, heads = configs[0]
    total, bd = count_params(dim, depth, heads, args.vocab)
    detail = bd["blocks_detail"]
    print(f"\nDetailed breakdown for {dim}x{depth}x{heads}:")
    print(f"  Token embedding:    {format_params(bd['token_embedding']):>10}")
    if "position_embedding" in bd:
        print(f"  Position embedding: {format_params(bd['position_embedding']):>10}")
    print(f"  Input projection:   {format_params(bd['in_proj_rmsnorm'] + bd['in_proj_linear']):>10}")
    print(f"  Blocks (x{depth}):      {format_params(bd['blocks']):>10}")
    print(f"    Per block:        {format_params(detail['per_block']):>10}")
    print(f"      Attention:      {format_params(detail['attn']):>10}")
    print(f"      GeGLU FFN:     {format_params(detail['ff']):>10}  (hidden={detail['ff_hidden']})")
    print(f"      Norms:          {format_params(detail['norms']):>10}")
    print(f"  Output projection:  {format_params(bd['out_proj_rmsnorm'] + bd['out_proj_linear']):>10}")
    print(f"  Total:              {format_params(total):>10}")


if __name__ == "__main__":
    main()
