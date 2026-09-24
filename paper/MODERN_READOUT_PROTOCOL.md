# Phase 60 protocol: dedicated carrier readouts on the modern backbone

Status: completed without promotion on 2026-09-22. The protocol below was
frozen before GPU preflight or outcome inspection.

Execution note: all three registered 100-step preflights completed successfully
with finite metrics and optimizer telemetry. Their provisional rank-128/rank-32
carrier-function-step ratio over steps 10--90 was 1.208, inside the registered
range; the final decision used steps 5k--95k. No protocol fields were changed
after preflight inspection.

## Outcome

Scalar, rank 32, and rank 128 reached final NLL `3.131003`, `3.128782`, and
`3.126594`. Rank 32 minus scalar was `-0.002222`, with block-32 interval
`[-0.003978,-0.000465]`. It beat scalar at all five late development
checkpoints, but did not reach the registered `-0.003` materiality threshold.
Rank 128 minus rank 32 was `-0.002188`, with interval
`[-0.003882,-0.000520]`. The mature function-step ratio was matched at `1.048`,
but rank 128 also missed the materiality threshold and its five late deltas
changed sign. Therefore neither extension is promoted and readout tuning stops
on this final window.

The negative endpoint intervals establish paired stream-level precision, not
training-seed robustness. Mechanistically, both projected branches became the
dominant positional path: their scalar-gate means fell to `0.00424` and
`0.00324`, versus `0.04071` in scalar, while their normalized Q/K directions
departed much more strongly from content-only Q/K. Thus dedicated readout does
restore positional leverage, but here that leverage yields too little loss
improvement for its `0.59M`/`2.36M` parameters and approximately `3.4--3.6%`
throughput cost.

## Question

Phase 59 showed that the scalar pre-Q/K carrier transfers to the bundled modern
decoder, but its advantage over RoPE was smaller than in the controlled
backbone. Diagnostics showed a proximal difference: the scalar carrier changed
the normalized Q/K directions less in the modern model even though its learned
gate was not smaller.

This cohort tests whether a dedicated projected-space readout of the same fixed
sinusoid restores useful carrier leverage when it no longer has to share
`W_q`/`W_k` with content, and whether rank 128 is materially better than the
efficient rank-32 readout in this architecture.

## Frozen arms

All arms retain the exact Phase-59 scalar anchor: one directly optimized,
unconstrained FP32 scalar per layer, initialized at 1.0 and tied between Q and
K. Standard RoPE follows the method-aware Q/K RMS normalization.

| Arm | Additional projected-space carrier | Readout LR multiplier |
|---|---|---:|
| `scalar` | none | -- |
| `rank32` | shared bias-free `768 -> 32` sinusoidal trunk, with separate zero-initialized bias-free `32 -> 768` Q/K outputs | 6.367487169620489 |
| `rank128` | shared bias-free `768 -> 128` sinusoidal trunk, with separate zero-initialized bias-free `128 -> 768` Q/K outputs | 2.449489742783178 |

The projected Q/K outputs are added after `W_q` and `W_k`, before the existing
per-head Q/K RMS normalization and standard RoPE. Because both output matrices
start at exactly zero, all three arms implement the same scalar-carrier
function at initialization. Only the zero-initialized readout outputs receive
the rank-specific LR multiplier; the sinusoidal trunk and scalar anchor retain
the base positional LR. AdamW decay is inversely compensated on the multiplied
readouts so the effective decay product remains fixed.

The multipliers are inherited without retuning from the controlled-backbone
function-step calibration. Rank 32 was empirically calibrated; rank 128 uses
the earlier dimension-aware calibration. This cohort measures their realized
function-step ratio on the modern backbone rather than adapting either value
after seeing losses.

## Backbone and training

The complete Phase-59 modern bundle and training recipe remain fixed:

- width 768, depth 8, eight-head MHA, context 1,024;
- pre-RMSNorm with `eps=1e-6`;
- bias-free attention projections and bias-free SwiGLU at width 2,048;
- no learned input projection;
- final RMSNorm and tied token/output weights;
- full-head standard RoPE, `theta=10000`, and method-aware Q/K RMSNorm;
- canonical OpenWebText GPT-2 cache;
- sequence batch 32, 100,000 optimizer steps, seed 123;
- AdamW at peak LR `1.2e-3`, betas `(0.9, 0.98)`, weight decay `0.01`,
  200-step warmup, linear decay, and global clipping at 1.0;
- development evaluation every 5,000 steps;
- final paired evaluation on the previously uninspected 1,024 validation blocks
  `[10240, 11263]`.

The scalar reference is retrained because Phase 59 intentionally retained no
weights. This gives every Phase-60 arm the same fresh final stream and avoids
using the already-inspected Phase-59 endpoint to select between readout ranks.
A RoPE arm is not repeated: Phase 59 already registered the modern scalar
transfer, while this cohort asks only about the extension conditional on that
success.

## Registered decisions

Report mean final NLL, paired IID and contiguous-block-32 bootstrap intervals,
all development deltas, parameter counts, throughput, memory, scalar-gate and
carrier-energy summaries, optimizer finiteness, and the rank-128/rank-32
carrier-function-step ratio.

Rank 32 transfers successfully only if:

1. `rank32 - scalar <= -0.003` NLL;
2. the upper endpoint of its block-32 interval is below zero;
3. it beats scalar at every development checkpoint from 80k through 100k;
4. all training and optimizer metrics are finite.

Rank 128 is necessary rather than merely wider only if:

1. the median rank-128/rank-32 carrier-function-step ratio from 5k through 95k
   lies in `[0.8, 1.25]`;
2. `rank128 - rank32 <= -0.003` NLL;
3. the upper endpoint of its block-32 interval is below zero;
4. it beats rank 32 at every development checkpoint from 80k through 100k;
5. all metrics are finite.

If the function-step match fails, the rank comparison is optimization-
confounded and cannot promote rank 128. If rank 32 fails, do not tune rank,
readout LR, or carrier shape on the final window. The existing controlled-
backbone FFN-capacity control is retained; a new modern capacity control is not
part of this conditional transfer cohort.

## Execution and artifacts

Each arm receives a 100-step compiled-bf16 preflight. Main runs keep one
rolling recovery checkpoint at 10k intervals and remove it only after exact
completion and final-detail verification. No final weights are saved. Compact
configs, provenance, metrics, optimizer traces, per-block losses, reports, and
cleanup manifests are retained. GPU work uses `gpu-claim`, owner `mlprope`,
with a hard concurrency ceiling of two.
