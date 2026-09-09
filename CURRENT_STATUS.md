# MLPRope current status

_Authoritative as of 2026-09-08. Older mechanisms and protocols are preserved
in git history; compact experimental evidence remains under `results/`._

## Bottom line

Two attention-local sinusoidal mechanisms remain scientifically interesting:

1. **AddRoPE:** an additive Fourier carrier on projected Q/K;
2. **pre-Q/K sinusoid:** add one tied, gated sinusoid to the inputs of the Q
   and K projections, then apply standard RoPE.

The clearest replicated result is the second method. At h768/d8 and 200k steps,
pre-Q/K + RoPE beat fixed RoPE in all three paired seeds, with mean delta
`-0.055334`. It also transferred to h1024/d12 (`-0.040581`) and survived
removing QKNorm (`-0.049523`); all three predeclared Phase-38 gates passed.

Phase 42 has now completed the paper-protocol component screen at 100k,
batch 32, and seed 123. Scalar pre-Q/K + RoPE was best at `3.146756`, beating
fixed RoPE by `-0.037011`. The one-shot input sinusoid (`3.153390`) and fixed
post-RoPE AddRoPE (`3.151333`) were competitive but weaker. A single learned
gate shared globally was tied with per-layer gates; a fixed gate and
first-layer-only injection were worse.

Phase 39 separated carrier location at 30k. A sinusoid written once at model
input helped only `-0.013170`, whereas repeated pre-Q/K access helped
`-0.073805`. A fixed native AddRoPE+RoPE carrier helped `-0.019362`; learning
separate direct Q/K amplitudes and phases improved that hybrid to `-0.027412`.
Standalone direct AddRoPE was stronger at `-0.053221`, so standard RoPE and
this native carrier interfered rather than composed in this screen.

The promoted carrier should remain simple. Separate Q/K gains, per-pair amplitude,
phase, smooth spectral amplitude, and globally shared learned frequencies did
not improve the scalar anchor at mature horizon. Content-dependent RoPE,
cumulative clocks, and EMA/linear-RNN controllers are also closed. Phase 43
does provide one new structural lead: a static rank-32 bottleneck with
dedicated Q/K readouts improved the scalar pre-Q/K parent by `-0.009401` at
20k. Phase 45 strengthened that lead: dense shared, dense separate, and
nonlinear rank-128 projected-space pathways all improved the same parent by
about `-0.014` and were statistically tied with one another. The simplest
interpretation is that a sufficiently expressive native Q/K positional map
matters, while Q/K untying and nonlinearity have not shown independent value.

## Strongest completed evidence

| Result | Protocol | Finding |
| --- | --- | ---: |
| AddRoPE amplitude 1.0 vs fixed RoPE | 30k, 3 paired seeds | `-0.076867` mean |
| AddRoPE amplitude 1.0 vs 0.3 | 30k, 3 paired seeds | `-0.014895` mean |
| pre-Q/K + RoPE vs fixed RoPE | 30k, 3 paired seeds | `-0.065235` mean |
| pre-Q/K + RoPE vs fixed RoPE | 200k, 3 paired seeds | `-0.055334` mean |
| pre-Q/K + RoPE vs fixed RoPE | 100k, batch 32, 1 seed | `-0.037011` |
| global-gate vs per-layer-gate pre-Q/K | 100k, batch 32, 1 seed | `+0.000421`, interval crosses zero |
| fixed-gate vs learned pre-Q/K | 100k, batch 32, 1 seed | `+0.006464` |
| dedicated rank-32 Q/K residual vs scalar pre-Q/K | 20k, batch 32, 1 seed | `-0.009401` |
| rank-32 input residual vs scalar input | 20k, batch 32, 1 seed | `-0.001772` |
| per-pair input amplitude vs scalar input | 20k, batch 32, 1 seed | `-0.002631` |
| dense shared projected-space Q/K map vs scalar pre-Q/K | 20k, batch 32, 1 seed | `-0.014128` |
| dense separate projected-space Q/K maps vs scalar pre-Q/K | 20k, batch 32, 1 seed | `-0.013855` |
| nonlinear rank-128 Q/K map vs scalar pre-Q/K | 20k, batch 32, 1 seed | `-0.013611` |
| dense pre-map vs scalar pre-Q/K | 20k, batch 32, 1 seed | `-0.005913` |
| linear rank-128 separate Q/K map vs scalar pre-Q/K | 20k, batch 32, 1 seed | `-0.013312` |
| linear rank-128 separate vs linear rank-32 separate | 20k, batch 32, 1 seed | `-0.003910` |
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
- one-shot input controls for a scalar, an exact-parent rank-32 linear
  residual, dense-linear residual, residual MLP, and direct signed per-pair
  amplitudes;
- breadth-screen controls for dense pre-maps, shared or separate dense native
  Q/K residuals, and a nonlinear low-rank Q/K residual;
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

The completed paper evidence cohort fixed context 1024, sequence batch 32, and
100k updates (3.277B nominal tokens). A measured RTX 5090 benchmark found batch
32 at roughly 215k target tokens/s and 14.3 GiB allocated; batch 64 gained only
3.2% throughput while allocating 26.6 GiB. All ten Phase-42 jobs survived an
interrupted interactive launcher through durable checkpoints and completed
under supervisor. The dependent Phase-43 and Phase-44 screens also completed.

Phase 45 completed its six-arm 20k breadth screen. Dense linear input was worse;
the input MLP improved the scalar input parent by only `-0.001691` and was
worse than the much smaller per-pair amplitude control. The dense pre-map
passed but was statistically tied to the earlier rank-32 pre-map. All three
native projected-space arms clustered within `0.00052`, while each beat the
earlier rank-32 Q/K residual by `0.0042--0.0047`.

The remaining efficient design question is whether a **linear rank-128**
native map, shared or separate between Q/K, recovers the top cluster. That
would separate capacity from nonlinearity without spending on mature
reproduction. After that narrow filter, only survivors should consume a 100k
confirmation, extra seeds, or a parameter-matched non-positional control. Rank
comparisons must control function-space update scale; see
[`INPUT_SINUSOID_DESIGN.md`](INPUT_SINUSOID_DESIGN.md).

Phase 46 completed exactly that two-arm screen. Linear rank-128 separate Q/K
readouts reached `3.441766`, statistically tying dense separate and nonlinear
rank-128 while using 2.36M positional parameters. The shared rank-128 form was
slightly but significantly worse at `3.442932`. Rank 128 improved the linear
rank-32 separate pathway by `-0.003910`.

That rank result does not yet isolate representational capacity: under the
same Adam LR, rank 128's measured carrier-function step was 3.40x larger
through step 64 and 1.80x larger at the median sampled post-warmup step. A
narrow update-calibration experiment is warranted before mature confirmation.

Phase 47 is running the predeclared calibration: separate linear Q/K readouts
at rank 32 with multiplier `sqrt(768/32)=4.898979`, and rank 128 with
`sqrt(768/128)=2.449490`. Only the zero-initialized output factors receive the
larger LR; bottlenecks and scalar gates remain at base LR. AdamW coefficients
are inversely adjusted so effective decay per step is unchanged. Interpretation
requires the median post-warmup carrier-step ratio to lie in `[0.8, 1.25]`.

The next experiments test the method rather than search its local shape space:

1. **component necessity:** Phase 42 has completed the RoPE/carrier factorial,
   fixed/global/layerwise gates, and one-block versus repeated injection;
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
