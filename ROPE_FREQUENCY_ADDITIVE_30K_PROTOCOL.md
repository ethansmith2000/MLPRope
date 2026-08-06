# Phase-22 additive-frequency 30k confirmation

_Locked before launch on 2026-08-04._

## Question

Does the phase-21 gain from a free additive base-RoPE frequency survive the
longer 30k training horizon across three paired seeds?

Phase 21 found `omega = omega0 + u` improved 1024-context loss over fixed RoPE
by `0.011992` on average, with gains of `0.012825`, `0.010596`, and `0.012554`
for seeds 123/456/789. It was the only static parameterization to cross the
pre-registered `0.01` promotion threshold.

## Arms

1. Fixed standard RoPE.
2. Layer-shared additive frequency: one zero-initialized `[1,F]` delta per
   layer, shared by heads and Q/K, with `omega = omega0 + u`.

Both are strict rotations, use fused SDPA, and have identical model/data
initialization within each seed. The additive arm has 384 learned parameters.

## Locked protocol

- Model: h768/d8, 8 heads, context 1024.
- Horizon: 30,000 optimizer steps, batch 8.
- Schedule: 200-step warmup to LR `3e-4`, then linear decay to zero.
- Optimizer: AdamW, betas `0.9/0.98`, weight decay `0.01`.
- Seeds and paired initialization seeds: `123`, `456`, `789`.
- Development checks: 25 batches every 5,000 steps.
- Final primary holdout: 256 batches beginning at batch 2,048, context 1024.
- bf16 training, fp32 positions/frequencies/angles/trigonometry.
- Final weights saved; no intermediate optimizer checkpoint.
- No 2048/4096 evaluation and no length-extrapolation claim.

## Confirmation rule

The additive result is confirmed only if its final 1024 loss:

- improves by at least `0.01` on average;
- has the same favorable sign in all three seeds;
- retains finite spectra without pervasive collapse;
- has no material throughput regression.

A smaller consistent result is evidence that the 5k effect decays with tokens,
not grounds for immediately expanding a token-conditioned sweep.

## Resource allocation

Every job runs through `gpu-claim`. Two sequential workers impose a maximum of
two concurrent MLPRope jobs, while `GPU_SELECTOR=any` allows the shared queue to
choose whichever two devices the higher-priority suite is not using.

