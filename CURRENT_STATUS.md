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
token budget. There is no second-corpus, modality, or longer-context training
transfer result yet. Those are now more valuable than another carrier-shape
sweep.

## Active implementation

The runtime keeps:

- standard fixed RoPE and NoPE;
- the tied-scalar pre-Q/K carrier, initialized at gate 1.0;
- static AddRoPE and the pointwise content-conditioned AddRoPE reference;
- generic positional LR control and optimizer/function-step diagnostics;
- paired evaluation, provenance, resumable checkpoints, and fused SDPA.

It no longer implements learned carrier frequency, pre-Q/K smooth amplitude,
separate Q/K preprojection transforms, dynamic RoPE, clocks, EMA, residual
position writes, or attention-output writes. Enabled archived configurations
fail explicitly; disabled archived blocks canonicalize to an inert active form.

## Next evidence program

The next experiments should test the method, not search its local shape space:

1. **mechanism:** length-stratified loss plus attention entropy, attended
   distance, position correlation, and carrier-logit attribution from trained
   checkpoints;
2. **factorial closure:** the missing matched NoPE cell needed to quantify the
   mature RoPE-by-carrier interaction;
3. **generalization:** another corpus and context-length training transfer;
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
