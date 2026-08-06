# Phase-21 static RoPE frequency parameterization screen

_Locked before launch on 2026-08-04._

## Question

Was phase 20's small learned-frequency effect limited by optimizing additive
steps in log-frequency space, particularly the base-frequency scaling of the
gradient?

This screen changes only the forward/backward parameterization of the direct
layer-shared frequency table. It does not test token-conditioned frequencies,
per-head spectra, learned static functions, or length extrapolation.

## Reused controls

The completed phase-20 runs provide three paired seeds for:

- fixed RoPE;
- ordinary exponential, `omega = omega0 * exp(u)`.

Their training inputs, paired initialization, optimizer, and primary 1024-token
holdout match this screen. Phase 20 used a larger cache extent for incidental
long-context evaluation; that does not add parameters or change its 1024-token
training path.

## New arms

Every arm learns one `[1,F]` table per layer, shared by all heads and Q/K, with
`u=0` giving exact standard RoPE.

1. `exp-full-ste`: exponential forward, but `d omega / d u = 1`.
2. `softplus`: `omega = omega0 * softplus(u + log(e-1))`.
3. `additive`: `omega = omega0 + u`, allowing zero crossings and reversal.
4. `bounded-log`: `omega = omega0 * exp(tanh(u))`, limiting multipliers to
   `[exp(-1), exp(1)]`.

The ordinary exponential and bounded-log arms have the same initial Jacobian.
The full-STE and additive arms both have an identity initial Jacobian but
different forward geometry. Together these contrasts distinguish gradient
scaling from the positive/log-space forward constraint.

## Locked training protocol

- Model: h768/d8, 8 heads, training context 1024.
- Horizon: 5,000 optimizer steps, batch 8.
- Schedule: 200-step warmup to LR `3e-4`, then linear decay to zero.
- Optimizer: AdamW, betas `0.9/0.98`, weight decay `0.01`.
- Seeds: `123`, `456`, `789`, each used as both data seed and paired
  initialization seed.
- Development evaluation: 25 batches at step 5,000.
- Final primary holdout: 256 batches starting at batch 2,048, context 1024.
- Fused SDPA, method-aware Q/K RMSNorm, bf16 training with fp32 frequency/angle
  arithmetic and fp32 trigonometry.
- Final model weights are saved for spectrum inspection. Intermediate optimizer
  checkpoints are omitted to limit disk usage; the suite itself runs under
  supervisor and individual failed cells can be rerun.

## Decision rule

This remains a screen. A parameterization is eligible for structured static or
dynamic-controller work only if:

- its mean improvement over fixed RoPE is at least `0.01` at context 1024;
- all three seed signs agree;
- frequency and gradient diagnostics remain finite and interpretable;
- any throughput difference is measured separately.

Differences near phase 20's `0.002` effect are recorded as null. Results at
2048/4096 are neither collected nor used.

## Resource allocation

All jobs use `gpu-claim`. This batch is restricted to GPUs 6 and 7 with two
sequential workers, so no more than two MLPRope jobs run concurrently. GPUs 0–5
remain available to the higher-priority project. This is a temporary allocation
for phase 21, not a permanent MLPRope policy.

