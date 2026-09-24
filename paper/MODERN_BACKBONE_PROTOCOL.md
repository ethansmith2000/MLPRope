# Phase 59 protocol: modern-backbone transfer

Status: completed on 2026-09-21; registered transfer gate passed.

Execution note: both registered 100-step preflights completed successfully on
2026-09-21 with finite metrics and finite scalar intervention telemetry. The
two 100k runs then launched concurrently under supervisor through `gpu-claim`;
no protocol fields were changed after preflight inspection.

Outcome: scalar pre-Q/K + RoPE reached `3.141150` versus `3.153352` for
standard RoPE on the registered final window. The delta was `-0.012201`, with
contiguous-block-32 95% interval `[-0.013890,-0.010483]`. All metrics were
finite, so all three continuation conditions passed. See
`results/phase59_modern_backbone/REPORT.md` for the compact readout.

## Question

Does the primary scalar pre-Q/K sinusoidal carrier retain a material advantage
over standard RoPE after replacing the repository's unusual controlled decoder
with a more contemporary architecture bundle?

This is a transfer test, not an attribution study. The backbone changes as a
declared bundle; the carrier formula, training data, context, optimizer, and
paired initialization remain matched between the two arms. A negative result
does not automatically authorize architecture decomposition, carrier tuning,
or optimizer search.

## Frozen modern backbone

Both arms use:

- decoder-only causal Transformer, width 768, depth 8, eight-head MHA;
- pre-RMSNorm residual blocks with explicit `eps=1e-6`;
- full-head standard RoPE with `theta=10000`;
- the existing method-aware per-head Q/K RMSNorm with `eps=1e-6`;
- bias-free Q, K, V, attention-output, and MLP projections;
- bias-free SwiGLU with hidden width 2,048, the 64-aligned `8d/3` width;
- no learned input projection: token embeddings enter the residual stream
  directly;
- final RMSNorm followed by a bias-free projection through the same parameter
  used by the token embedding;
- tied token/output table initialized from `Normal(0, 0.02)`;
- existing name-stable Xavier initialization for all linear weights;
- no dropout, fused causal SDPA, bf16, and `torch.compile` default mode.

The implementation is selected by one resolved `backbone_variant="modern"`
field. The default `controlled` path must remain bit-exact with the historical
module graph and initialization. This transfer deliberately does not add GQA,
MQA, dropout, residual scaling, a new optimizer, or a new initialization rule
for linear weights.

## Frozen arms

| Arm | Position mechanism | Trainable positional parameters |
|---|---|---:|
| `R-modern` | standard RoPE | 0 |
| `C+R-modern` | one direct, unconstrained FP32 scalar per layer multiplying the fixed model-width sinusoid before both Q/K projections, initialized at 1.0; standard RoPE follows | 8 |

All AddRoPE channels, projected Q/K readouts, input sinusoids, learned
frequencies/phases/amplitudes, dynamic mappers, ordinary Q/K biases, and
learned absolute positions are disabled.

## Matched training and evaluation

- canonical OpenWebText GPT-2 token cache;
- context 1,024, sequence batch 32, one GPU per run;
- 100,000 optimizer steps (3.276B nominal target tokens);
- training and paired-initialization seed 123;
- AdamW, peak LR `1.2e-3`, betas `(0.9, 0.98)`, weight decay `0.01`,
  200-step warmup, linear decay, and global clipping at `1.0`;
- 128-block development evaluation from validation block 0 every 5,000 steps;
- one final 1,024-block evaluation on previously uninspected validation blocks
  `[9216, 10239]`.

The optimizer recipe is transferred unchanged from Phase 57. It is common to
both arms and is not claimed to be the optimum for the new backbone. We do not
select a method-specific LR or use a short-horizon LR sweep.

## Registered readout and continuation rule

Report final mean token NLL, paired per-block scalar-minus-RoPE differences,
IID and contiguous-block-32 bootstrap 95% intervals, all 5k development
deltas, parameter counts, throughput, elapsed time, peak CUDA memory, effective
gate trajectories, and finite-metric checks.

The primary transfer gate passes only if:

1. `C+R-modern - R-modern <= -0.010` NLL;
2. the upper endpoint of its contiguous-block-32 interval is below zero;
3. all training and intervention-optimizer metrics are finite.

If it passes, a calibrated rank-32 Q/K-readout arm may be run next on this same
backbone and protocol. It is not launched automatically. If it fails clearly,
the paper narrows the architecture claim; carrier, gate, frequency, and
optimizer variants remain closed.

## Preflight and artifact policy

Both arms receive a 100-step GPU preflight exercising the compiled bf16 path.
Main runs keep one rolling recovery checkpoint at 10k intervals. Its declared
purpose is interruption recovery for the multi-hour run; it is removed only
after the exact completion marker and final evaluation-detail file are
verified. No final weights are saved. Configs, source/data provenance, metrics,
optimizer telemetry, evaluation losses, reports, and cleanup manifests are
retained. All GPU work uses `gpu-claim` with owner `mlprope` and hard
concurrency two.
