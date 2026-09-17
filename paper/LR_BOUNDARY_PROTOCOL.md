# Phase 56 learning-rate boundary protocol

Status: frozen before training on 2026-09-14.

## Question

Both RoPE and scalar pre-Q/K improved monotonically across the Phase-55 peak
AdamW learning-rate grid, whose upper endpoint was `6e-4`. Does one additional
geometric boundary point (`1.2e-3`) improve the standard RoPE reference without
destabilizing the scalar method?

This is a bounded recipe-selection check, not an optimizer search. There will
be no recursive expansion beyond `1.2e-3` and no simultaneous change to Adam
betas, warmup, weight decay, clipping, batch size, or schedule.

## Frozen matrix

Train exactly two new seed-123 runs from initialization at the canonical
h768/d8, batch-32, context-1024, 100k-update OpenWebText recipe:

| Peak LR | RoPE | Scalar pre-Q/K + RoPE |
|---:|---|---|
| `6.0e-4` | retained Phase 55 | retained Phase 55 |
| `1.2e-3` | new | new |

Both new arms retain AdamW betas `(0.9, 0.98)`, weight decay `0.01`, a linear
schedule, 200 warmup updates, and gradient clipping at `1.0`. The scalar gate
is initialized at one, directly optimized, and receives the same optimizer
policy used in all preceding paper runs.

## Selection and evaluation

- Choose the future common learning rate using the RoPE **development** NLL at
  100k, comparing `6e-4` with `1.2e-3`.
- `1.2e-3` is eligible only if both new runs complete with finite metrics. If
  the scalar arm is unstable, retain `6e-4` as the highest jointly stable
  common recipe and report the failure.
- Reuse validation blocks `[6144, 7167]` only to extend the paired Phase-55
  robustness table. Because Phase 55 has now informed recipe choice, this
  window is selection evidence and is no longer described as untouched.
- Reserve `[7168, 8191]` for a later confirmatory paper evaluation. Phase 56
  must not inspect it.
- Report the `1.2e-3` scalar-minus-RoPE delta with IID and contiguous block-32
  intervals, plus throughput, memory, and finite-training checks.
- Stop after this point regardless of which endpoint wins. The selected value
  is the best of a resource-bounded candidate set, not a claimed optimum.

## Artifact policy

Each run may keep one rolling recovery checkpoint at 10k intervals solely for
interruption recovery. After successful final evaluation and analysis, remove
the exact recovery directories and record the cleanup. Save no final weights;
retain compact configs, provenance, metrics, optimizer diagnostics, per-block
losses, and the aggregate report.
