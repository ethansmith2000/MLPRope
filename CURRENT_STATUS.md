# MLPRope current status

_Authoritative as of 2026-09-06. Older mechanisms and protocols are preserved
in git history; compact experimental evidence remains under `results/`._

## Bottom line

Two attention-local sinusoidal mechanisms remain scientifically interesting:

1. **AddRoPE:** an additive Fourier carrier on projected Q/K;
2. **pre-Q/K sinusoid:** add one tied, gated sinusoid to the inputs of the Q
   and K projections, then apply standard RoPE.

The clearest current result is the second method. At h768/d8 and 200k steps,
pre-Q/K + RoPE beat fixed RoPE in all three paired seeds, with mean delta
`-0.055334`. It also transferred to h1024/d12 (`-0.040581`) and survived
removing QKNorm (`-0.049523`); all three predeclared Phase-38 gates passed.

Phase 39 separated carrier location at 30k. A sinusoid written once at model
input helped only `-0.013170`, whereas repeated pre-Q/K access helped
`-0.073805`. A fixed native AddRoPE+RoPE carrier helped `-0.019362`; learning
separate direct Q/K amplitudes and phases improved that hybrid to `-0.027412`.
Standalone direct AddRoPE was stronger at `-0.053221`, so standard RoPE and
this native carrier interfered rather than composed in this screen.

The carrier should remain simple. Separate Q/K gains, per-pair amplitude,
phase, smooth spectral amplitude, and globally shared learned frequencies did
not improve the scalar anchor at mature horizon. Content-dependent RoPE,
cumulative clocks, and EMA/linear-RNN controllers are also closed.

## Strongest completed evidence

| Result | Protocol | Finding |
| --- | --- | ---: |
| AddRoPE amplitude 1.0 vs fixed RoPE | 30k, 3 paired seeds | `-0.076867` mean |
| AddRoPE amplitude 1.0 vs 0.3 | 30k, 3 paired seeds | `-0.014895` mean |
| pre-Q/K + RoPE vs fixed RoPE | 30k, 3 paired seeds | `-0.065235` mean |
| pre-Q/K + RoPE vs fixed RoPE | 200k, 3 paired seeds | `-0.055334` mean |
| pre-Q/K + RoPE, h1024/d12 vs matched RoPE | 200k, 1 paired seed | `-0.040581` |
| pre-Q/K + RoPE without QKNorm vs matched RoPE | 200k, 1 paired seed | `-0.049523` |
| pre-Q/K + RoPE vs pre-Q/K + NoPE | 200k, 1 paired seed | `-0.030773` |
| pre-Q/K + RoPE vs input sinusoid + RoPE | 30k, 1 paired seed | `-0.060635` |
| direct AddRoPE + NoPE vs fixed RoPE | 30k, 1 paired seed | `-0.053221` |
| split/pair amplitude/pair phase ladder | 200k, 1 paired seed | all within about `0.001` |
| shared log-frequency carrier vs fixed | 200k, 1 paired seed | `+0.000861`, null |
| horizon-frequency carrier vs fixed | 200k, 1 paired seed | `+0.001341`, worse |
| direct smooth amplitude vs scalar | 200k, 1 paired seed | `+0.000111`, null |
| exponential smooth amplitude vs scalar | 200k, 1 paired seed | `-0.000363`, null |
| pointwise content AddRoPE vs position-only | 30k, 3 paired seeds | `-0.010812` mean |
| AddRoPE scalar EMA vs pointwise | 15k, 1 paired seed | `-0.010626` step-matched; about `-0.0015` iso-wall-clock |

At 15k, AddRoPE and pre-Q/K were strongly sub-additive: their combination was
`+0.004934` worse than AddRoPE alone. This is evidence of overlapping function,
but the comparison is too early to support a mature exclusivity claim.

## Baseline architecture

The main h768/d8 model has approximately 153.4M parameters and uses:

- decoder-only causal self-attention with fused PyTorch SDPA;
- eight 96-dimensional heads;
- pre-norm residual blocks with LayerNorm;
- separate bias-free Q/K/V projections and a biased output projection;
- GeGLU feed-forwards at four times model width;
- per-head Q/K normalization;
- standard fixed split-half RoPE at context 1024;
- tied paired initialization and fixed data order for comparisons;
- AdamW with linear scheduling and bf16 autocast.

The promoted carrier uses method-aware Q/K RMSNorm: content and position are
combined before `W_q/W_k`, then each projected head is normalized once.
Phase 38 established that the benefit is not dependent on QKNorm, although
normalization changes absolute loss and remains part of the primary recipe.

## Why the closed refinements are genuinely closed

Phase 37 directly paired scalar, exponential smooth amplitude, and signed
direct smooth amplitude for 200k steps. The primary disjoint 1,024-example
holdout was null, even though both shape maps moved substantially and had
finite gradients, Adam states, updates, and carrier-function movement. This
rules out an obvious inactive-path explanation for their failure.

Phase 34 similarly showed that horizon-normalized frequency coordinates remove
the dangerous raw `p` multiplier from the endpoint derivative, but still do
not improve modeling loss. Faster direct frequency coordinates eventually
violated spectral ordering. The negative frequency result is therefore not
well explained by the one optimization pathology we originally identified.

## Evidence limitation

With batch 8 and sequence length 1024, each step consumes 8,192 tokens. The
h768/d8 model sees:

| Steps | Tokens | Tokens / parameter |
| ---: | ---: | ---: |
| 30k | 245.8M | 1.60 |
| 100k | 819.2M | 5.34 |
| 200k | 1.638B | 10.68 |

The mature three-seed result is reproducible and the h1024 test establishes
one scale transfer, but the runs are still below a conventional compute-optimal
token budget. There is no second-corpus or modality transfer result yet.
Context is now fixed at 1024 rather than treated as a paper axis. Those
remaining transfer tests are more valuable than another carrier-shape sweep.

## Active implementation

The runtime keeps:

- standard fixed RoPE and NoPE;
- the tied-scalar pre-Q/K carrier, initialized at gate 1.0;
- three isolated rank-32 development adapters: a residual carrier pre-map, a
  dedicated low-rank Q/K replacement path, and a dedicated low-rank Q/K
  residual path; all use one shared positional bottleneck and zero-initialized
  outputs;
- paper-ablation controls for a fixed gate, one gate shared globally across
  layers, and an explicit subset of carrier-active layers;
- static AddRoPE, explicit before/after-RoPE carrier placement, and the
  pointwise content-conditioned AddRoPE reference;
- generic positional LR control and optimizer/function-step diagnostics;
- paired evaluation, provenance, resumable checkpoints, and fused SDPA.

It no longer implements learned carrier frequency, pre-Q/K smooth amplitude,
dynamic RoPE, clocks, EMA, residual position writes, or attention-output
writes. Enabled archived configurations fail explicitly; disabled archived
blocks canonicalize to an inert active form. The new separate Q/K pathway is a
static, position-only low-rank readout and does not restore the removed dynamic
machinery.

## Next evidence program

The paper evidence cohort fixes context 1024, sequence batch 32, and 100k
updates (3.277B nominal tokens). A measured RTX 5090 benchmark found batch 32
at 215k target tokens/s and 14.3 GiB allocated; batch 64 gained only 3.2%
throughput while allocating 26.6 GiB. The ten matched seed-123 component and
comparison runs were launched through `gpu-claim` on 2026-09-06.

The initial interactive launcher session disappeared on 2026-09-07 while the
seven first-wave jobs were near step 58k. Every one had a complete, marked
step-55k checkpoint. The full Phase-42-to-Phase-43 chain now runs under
supervisor as `mlprope-phase42-43`; ten Phase-42 `gpu-claim` waiters are active
and will resume/start as GPUs become available. This prevents another client
session loss from terminating the suite.

A separate Phase-43 development screen is staged behind completion of that
cohort. It compares rank-32 pre-map, Q/K replacement, and Q/K residual pathways
for 20k batch-32 steps against matched 20k RoPE and scalar-carrier parents. The
paper matrix remains frozen; a successful Phase-43 arm must be promoted and
rerun rather than retroactively inserted into it.

The next experiments test the method rather than search its local shape space:

1. **component necessity:** the RoPE/carrier factorial, fixed/global/layerwise
   gates, and one-block versus repeated injection;
2. **mechanism:** position-stratified loss plus attention entropy, attended
   distance, position correlation, and carrier-logit attribution from trained
   checkpoints;
3. **generalization:** another corpus and a modernized decoder backbone;
4. **optional broader claim:** separable 2D pre-Q/K carriers in a ViT, followed
   by a spatial DiT only if image-classification transfer succeeds.

The first three should use identical data order within each pair and disjoint
1,024-example final holdouts. No refinement arm is admitted unless a distinct,
predeclared hypothesis emerges.

## Repository and storage state

- Compact phase reports and analysis JSON remain in `results/`; complete
  historical configs remain in `sweep_configs/`.
- Historical source and deleted protocols remain recoverable from git.
- On 2026-09-05, 14 redundant completed endpoint checkpoints plus one smoke
  checkpoint were deleted after verification, reclaiming about 24 GiB. Final
  model weights, evaluations, metrics, configs, and provenance remain.
- `/workspace` is not a persistent Vast volume. Irreplaceable weights must be
  copied off-box before instance recycle or destruction.
