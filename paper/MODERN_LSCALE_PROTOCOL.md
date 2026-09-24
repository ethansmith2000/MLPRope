# Phase 61 protocol: modern larger-scale transfer

Status: completed with positive but sub-material transfer on 2026-09-24. The
protocol below was frozen before GPU preflight or outcome inspection.

Execution note: both registered 100-step compiled-bf16 preflights passed on
2026-09-23 at the preferred per-device batch 32, with finite logs and roughly
23--24 GiB live GPU memory. No accumulation fallback or protocol field was
changed. The matched main pair then launched through `gpu-claim` under the
two-GPU project ceiling.

## Outcome

Standard RoPE reached final NLL `3.018072`; scalar pre-Q/K + RoPE reached
`3.014530`. Scalar minus RoPE was `-0.003542`, with IID interval
`[-0.005051,-0.001995]` and contiguous-block-32 interval
`[-0.005382,-0.001669]`. Scalar led at every registered late checkpoint from
80k through 100k, by `-0.00313` to `-0.00386`, and all metrics were finite.
The positive-transfer gate therefore passed, while the `-0.010` materiality
gate did not.

The final mean scalar gate was `0.04650`, close to the modern M-scale value
`0.04252`. Its mean projected position/content RMS ratios were larger than at
M scale (`Q: 0.191` versus `0.156`; `K: 0.214` versus `0.165`), and normalized
Q/K departed farther from their content-only directions. The smaller loss gain
therefore is not explained by gate collapse or failure to inject positional
energy. The conservative interpretation is positive direction transfer with
diminishing marginal value at this scale, not material scale invariance.

## Question

Does the scalar pre-Q/K sinusoidal carrier retain a meaningful advantage over
standard RoPE when the bundled modern decoder is scaled from h768/d8 to
h1024/d12, without changing the corpus, context, effective sequence batch,
optimizer, or training-token budget?

This is a scale-transfer test of the primary method. It is not a new method
search. Phase 60 did not promote dedicated rank-32 or rank-128 readouts on the
modern backbone, so neither is included.

## Frozen arms

1. standard full-head RoPE;
2. scalar pre-Q/K carrier + the same standard RoPE.

The scalar arm adds the fixed model-width sinusoid independently to the inputs
of `W_q` and `W_k` at every attention layer. It uses one directly optimized,
unconstrained FP32 scalar per layer, tied between Q and K and initialized at
`1.0`. The carrier does not enter V or the persistent residual stream. The two
arms use the same paired initialization seed for every shared tensor.

## Frozen model and training recipe

- modern decoder bundle: pre-RMSNorm (`eps=1e-6`), bias-free attention,
  bias-free SwiGLU, identity input path, final RMSNorm, tied token/output
  weights, and embedding initialization standard deviation `0.02`;
- width 1,024, depth 12, eight attention heads, context 1,024;
- full-head RoPE with `theta=10000` and method-aware Q/K RMS normalization;
- canonical OpenWebText GPT-2 cache;
- effective sequence batch 32, 100,000 optimizer steps, seed 123;
- AdamW, peak LR `1.2e-3`, betas `(0.9,0.98)`, weight decay `0.01`, global
  clipping at `1.0`, 200-step warmup, and linear decay;
- development evaluation every 5,000 steps;
- final paired evaluation on 1,024 previously uninspected validation blocks
  `[11264,12287]`.

The 3.277B-token budget is held equal to the Phase-59 M-scale transfer. At the
expected roughly 200M parameters it supplies about 16 tokens per parameter,
while cleanly isolating the model-scale change. If batch 32 does not fit one
32 GiB GPU, the only permitted memory adaptation is microbatch 16 with two
gradient-accumulation steps in both arms; the effective batch and optimizer
step count remain unchanged. That adaptation must be recorded before main
training.

## Registered readout

Report final NLL for both arms, scalar-minus-RoPE mean paired delta, IID and
contiguous-block-32 bootstrap intervals, every development delta, parameter
counts, throughput, peak memory, final carrier gates, and metric/optimizer
finiteness.

Positive scale transfer requires:

1. the block-32 interval upper endpoint is below zero;
2. scalar beats RoPE at every development checkpoint from 80k through 100k;
3. all training and intervention-optimizer metrics are finite.

Material scale transfer additionally requires scalar minus RoPE `<= -0.010`
NLL. A positive but sub-material result supports direction transfer but not a
strong scale claim. A failed positive-transfer gate stops scale expansion and
does not license method-specific tuning on this final window.

Paired block intervals measure uncertainty over this endpoint stream, not
training-seed variability. This one-seed experiment cannot establish seed
robustness or attribute an effect to an individual member of the modern
backbone bundle.

## Execution and artifacts

Each arm first receives a 100-step compiled-bf16 preflight. Main runs keep one
rolling recovery checkpoint at 10k intervals and remove it only after exact
completion and final-detail verification. No final weights are saved. Compact
configs, provenance, metrics, optimizer traces, per-block losses, reports, and
cleanup manifests are retained. GPU work uses `gpu-claim`, owner `mlprope`,
with a hard concurrency ceiling of two. Expected transient recovery storage is
approximately 5--7 GiB across the pair; retained compact evidence should remain
below 100 MiB after verified cleanup.
