# MLPRope current status

_Authoritative as of 2026-09-15. Older mechanisms and protocols are preserved
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

Phase 49 has now confirmed the calibrated rank-32 dedicated Q/K readout at the
same 100k paper budget. It reached `3.159413`, beating the scalar pre-Q/K
parent by `-0.010140` and the nearly exact FFN-capacity control by `-0.009576`.
Rank 128 improved by only another `-0.000750`, so rank 32 remains the efficient
candidate and has advanced to training-seed replication.

Phase 50 completed that replication. Rank 32 beat the scalar carrier in every
seed by `-0.010140`, `-0.006124`, and `-0.011046` NLL (mean `-0.009103`), and
beat the parameter-matched FFN control in every seed by `-0.009576`,
`-0.006466`, and `-0.004215` (mean `-0.006752`). Both registered three-seed
gates passed; the rank-32 extension is robust at this architecture and recipe.

A post-hoc checkpoint audit has materially changed the interpretation of that
rank-32 extension. Across the mature seed-123 and seed-456 checkpoints,
`94.65%--99.56%` of each direct Q/K carrier's pre-RoPE energy lies in its
positional mean. The original scalar branch is only `0.00135--0.03724` of the
direct branch RMS across layers. A nearly constant vector is still positional
after RoPE—it induces a relative kernel—but this result makes learned Q/K
bias-like structure a serious alternative to the claimed rich Fourier map.
Phase 51 therefore performs registered checkpoint counterfactuals before any
new mapper training.

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
about `-0.014` and were statistically tied with one another. Before the
checkpoint audit, the simplest interpretation was that a sufficiently
expressive native Q/K positional map matters. The surviving alternatives are
now a rich position-varying map versus bias-like structure rotated by RoPE;
Q/K untying and nonlinearity still have not shown independent value.

Phase 51 then evaluated six causal checkpoint interventions across all three
rank-32 seeds. Removing the direct branch was catastrophic (`+3.512501` NLL on
average), but keeping only its positionwise mean cost just `+0.011093`;
removing that mean cost `+3.475527`. Removing the scalar anchor at the trained
endpoint changed NLL by only `+0.000109`. Thus the learned constant/rotated
bias-like component is foundational, while the small centered component
accounts for essentially all of rank 32's incremental gain over the scalar
method. Endpoint ablation cannot determine whether the scalar was useful as
an optimization scaffold.

Phase 52 completed the final narrow close-out before transfer experiments.
Ordinary separate Q/K projection biases moved substantially from zero but
left the scalar parent unchanged: `3.110102` versus `3.110001`, delta
`+0.000101`. They do not explain rank 32's improvement. Training rank 32 with
the scalar carrier absent reached `3.107455`, better than scalar by
`-0.002546` but worse than full rank 32 (`3.099160`) by `+0.008295`, with both
IID and block-32 intervals excluding zero. The no-anchor arm trailed full rank
32 at every 5k development checkpoint and retained only about 23% of its
improvement over scalar. Thus the scalar can be removed from the trained
endpoint but acts as important optimization scaffolding when learning the
projected Fourier readout.

These conclusions use seed 123 with paired evaluation on the untouched
validation blocks `[5120, 6143]`; Phase 50 remains the three-seed evidence for
the full rank-32 method. The new runs saved no recovery checkpoints or final
weights. Their complete model outputs occupy only 2.1 MiB, and all compact
reference evaluations and paired analyses are retained.

Phase 53 completed the registered three-seed mechanism pass without new
training. All nine RoPE/scalar/rank-32 endpoint re-evaluations matched their
saved losses within `1.0e-4`. Scalar-minus-RoPE was negative in every target
position bin and each seed; the rank-32 extension's seed-average gain was
negative in all seven bins. The carrier therefore does not derive its endpoint
gain only from late-context tokens.

The stable attention signature is concentration, not uniformly shorter
context. Relative to RoPE, scalar normalized entropy fell by `0.039706` and
first-token mass rose by `0.012133`; both directions held in every seed.
Attention mass consistently shifted from relative distances 64--255 toward
1--3, but expected attended distance increased slightly for seed 123 and fell
for seeds 456 and 789. Rank 32 left the coarse geometry close to scalar.

Using a common trained Q/K RMS denominator, the four content/position score
terms matched independently recomputed full-QK logits to maximum absolute
error `4.005e-5`. Scalar and rank 32 had combined
position-involving centered-logit RMS `0.836647` and `1.644923`; their
full-versus-content-only local attention KL was `0.705940` and `1.913295`.
Both use substantial content--position cross terms, so neither is merely an
additive content-independent bias. These are descriptive local decompositions;
Phases 51 and 52 remain the model-level endpoint and training-path
interventions.

Phase 54 completed a registered inference-time carrier-origin audit across
both retained methods and all three seeds. Replacing `s(p)` by `s(p+c)` was
essentially neutral for offsets 1 and 4: scalar penalties were `+0.000025` and
`+0.000112`, and rank-32 penalties were `+0.000017` and `+0.000439`. Both
methods retained nearly all of their RoPE advantage through offset 64. Large,
out-of-distribution shifts were damaging: scalar crossed above RoPE around
offset 1024, while rank 32 crossed around offset 256 and degraded more steeply.
This rules out fragile dependence on the exact neighboring phase origin, but
not dependence on the broad absolute phase region seen during training.

Phase 55 completed the symmetric learning-rate audit. Scalar-minus-RoPE was
`-0.029164`, `-0.036355`, and `-0.044780` at peak LRs `1.5e-4`, `3e-4`, and
`6e-4`; every IID and block-32 interval excluded zero. Both arms achieved their
best tested endpoint at the upper boundary, so the gain is clearly not a
single-LR artifact but the grid did not bracket the recipe optimum. All new
metrics were finite, and cleanup reclaimed 6.86 GiB of recovery states.

Phase 56 completed the one-point learning-rate boundary check. At `1.2e-3`,
RoPE and scalar development NLL were `3.089271` and `3.035378`, improving over
their `6e-4` values `3.099196` and `3.055308`. On the common selection window,
scalar beat RoPE by `-0.054474`, with block-32 interval
`[-0.056115,-0.052756]`. Both runs were finite, so the frozen rule selects
`1.2e-3` as the prospective common recipe and stops LR expansion. This is the
best of a bounded candidate set, not an optimizer-optimum claim. Blocks
`[7168,8191]` remain uninspected for later confirmation.

Phase 57 is now active. It trains fresh RoPE, scalar pre-Q/K + RoPE, NoPE,
fixed input sinusoid without RoPE, learned absolute position, 25% partial
RoPE, and ALiBi arms under that common `1.2e-3` recipe. All seven 100-step GPU
preflights completed with finite metrics, including the actual ALiBi
FlexAttention path. The first two 100k jobs (RoPE and scalar) launched under a
hard two-GPU `gpu-claim` cap. The final endpoint is the previously uninspected
window `[7168,8191]`; no result has been inspected. No final weights will be
saved, and each completed run's single recovery checkpoint will be removed
immediately after its final evaluation is verified.

## Strongest completed evidence

| Result | Protocol | Finding |
| --- | --- | ---: |
| AddRoPE amplitude 1.0 vs fixed RoPE | 30k, 3 paired seeds | `-0.076867` mean |
| AddRoPE amplitude 1.0 vs 0.3 | 30k, 3 paired seeds | `-0.014895` mean |
| pre-Q/K + RoPE vs fixed RoPE | 30k, 3 paired seeds | `-0.065235` mean |
| pre-Q/K + RoPE vs fixed RoPE | 200k, 3 paired seeds | `-0.055334` mean |
| pre-Q/K + RoPE vs fixed RoPE | 100k, batch 32, 1 seed | `-0.037011` |
| pre-Q/K + RoPE vs fixed RoPE | 100k, batch 32, 3 seeds | `-0.036654` mean |
| pre-Q/K + RoPE vs fixed RoPE across peak LR | 100k, batch 32, 1 seed | `-0.029164/-0.036355/-0.044780` at `1.5e-4/3e-4/6e-4` |
| pre-Q/K + RoPE vs fixed RoPE at boundary LR | 100k, batch 32, 1 seed | `-0.054474` at `1.2e-3`; block-32 CI excludes zero |
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
| calibrated linear rank-128 separate vs scalar pre-Q/K | 20k, batch 32, 1 seed | `-0.016755` |
| calibrated linear rank-128 vs calibrated rank-32 | 20k, batch 32, 1 seed | `-0.002946`; function-step gate failed |
| calibrated rank-128 vs empirically matched rank-32 | 20k, batch 32, 1 seed | `-0.001058`; function-step gate passed |
| calibrated rank-32 vs scalar pre-Q/K | 100k, batch 32, 1 seed | `-0.010140` |
| calibrated rank-32 vs scalar pre-Q/K | 100k, batch 32, 3 seeds | `-0.009103` mean |
| calibrated rank-32 vs matched FFN capacity | 100k, batch 32, 1 seed | `-0.009576` |
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

The rank-32 module also exposes six evaluation-only, non-persistent
counterfactual modes: full, positional-mean-only, mean-removed, direct-zero,
scalar-zero, and all-zero. Any non-full mode raises during training. These are
analysis controls, not new trainable methods.

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

Phase 47 completed the predeclared calibration: separate linear Q/K readouts
at rank 32 used multiplier `sqrt(768/32)=4.898979`, and rank 128 used
`sqrt(768/128)=2.449490`. Only the zero-initialized output factors received the
larger LR; bottlenecks and scalar gates stayed at base LR. AdamW coefficients
were inversely adjusted so effective decay per step was unchanged.

Both arms improved materially over their uncalibrated counterparts. Rank 32
reached `3.441269` (`-0.004408`), statistically tying dense separate; rank 128
reached the best current 20k endpoint, `3.438322` (`-0.003444` versus its
uncalibrated version and `-0.002900` versus dense separate). Rank 128 beat
calibrated rank 32 by `-0.002946` on the paired final holdout.

The predeclared validity gate nevertheless failed: the median post-warmup
rank-128/rank-32 carrier-step ratio was `1.300`, just above the allowed `1.25`.
The ratio was noisy (`0.920--1.790`) and its late-half median was `1.221`, but
those are post-hoc sensitivity checks. Therefore the endpoint ordering is a
real optimization result, while the claim that rank 128 wins specifically
because of representational capacity remains unresolved. Persistent gradient
clipping is not the explanation: clipping ended by step 64 in both arms.

If rank is intended as a paper claim, the cheapest clean close-out is one
empirically recalibrated rank-32 arm. Using only Phase-47's optimization
diagnostic, not its holdout loss, its readout multiplier would move from
`4.898979` to approximately `4.898979 * 1.300 = 6.37`; the existing calibrated
rank-128 run remains the frozen comparator. This should be predeclared as a
calibration check, not another shape search. If rank itself is not a claim, no
further local screen is necessary: treat calibrated rank 128 as the strongest
optimizer-tuned candidate and move to mature confirmation and generalization.

Phase 48 completed the authorized one-shot close-out. It reran only rank 32 at
the diagnostic-derived multiplier `6.367487`, selected as
`4.898979 * 1.299758` without consulting validation loss, while retaining the
Phase-47 rank-128 arm as the frozen comparator. The function-step gate passed:
the median post-warmup rank-128/rank-32 ratio was `1.090`, inside `[0.8, 1.25]`.

Empirical rank 32 reached `3.439380`, improving the theoretical-calibration
rank-32 arm by `-0.001889` and the scalar parent by `-0.015698`. Rank 128
retained a small `-0.001058` advantage, with a paired-example interval barely
excluding zero. Function matching therefore closed about 73% of the original
equal-LR rank gap. The residual is compatible with a modest capacity effect,
but it is below the project's `0.003` scout materiality margin and comes from
one training seed. It is not robust evidence that rank 128 is the intrinsically
better operating point.

The local architecture/parameterization scout is now complete. No further
rank LR tuning or sinusoid-map expansion is planned. Rank 32 at multiplier
`6.367487` is the efficient projected-space extension candidate, conditional
on replication and Phase-51 attribution; the scalar carrier remains the
minimal primary method. Rank 128 at `2.449490` is a useful scaling ablation and
the best observed 20k endpoint.

Phase 49 completed the frozen 100k mature confirmation cohort. Its five
seed-123, batch-32, context-1024 arms were fixed RoPE, scalar pre-Q/K + RoPE,
scalar pre-Q/K plus a parameter-matched FFN widening, calibrated rank-32 Q/K
readout, and calibrated rank-128 Q/K readout. The final evidence window was the
previously unused validation block range `[4096, 5119]`; scouting used
`[2048, 3071]`, so design selection did not tune this endpoint.

The rank-32 branch adds 589,824 parameters over its scalar parent. Its control
widens four evenly spaced GeGLU blocks from 3072 to 3136, adding 590,336
non-positional parameters—an error of only 512 parameters. Rank 32 advances
to seed replication only if it beats both its scalar parent and this FFN
control by at least `0.003` NLL with paired intervals below zero, remains
better at every 80k--100k development checkpoint, and stays finite. It passed
every gate. The median rank-128/rank-32 carrier-step ratio was `1.029`, but
rank 128's `-0.000750` endpoint delta was below materiality and its paired
interval crossed zero.

Phase 50 now adds seeds 456 and 789 for four arms: fixed RoPE, scalar pre-Q/K,
the matched FFN control, and calibrated rank 32. Together with Phase 49 seed
123, success requires rank 32 to beat both scalar controls in every seed with
a mean delta at most `-0.003`, plus finite diagnostics. A durable launcher has
a hard ceiling of two live `gpu-claim` jobs, so the eight new runs occupy at
most two GPUs and execute in four waves. No new architecture or method
hyperparameter is being selected in this replication.

Because seed 123 selected the rank-32 candidate, the Phase-50 report preserves
the frozen three-seed gate but also reports the mean over fresh seeds 456 and
789 separately. It checks finite metrics and optimizer histories rather than
allowing an empty diagnostic list to pass, and adds a contiguous-block
bootstrap sensitivity analysis for neighboring validation blocks.

The remaining experiments test generalization rather than search the local
shape space:

1. **component necessity:** Phase 42 has completed the RoPE/carrier factorial,
   fixed/global/layerwise gates, and one-block versus repeated injection;
2. **mechanism:** Phase 53 has completed position-stratified loss, attention
   geometry, and exact local carrier-logit decomposition across three seeds;
   Phase 54 has completed the carrier-origin sensitivity audit;
3. **required generalization:** another corpus, a modernized decoder backbone,
   and recognized positional baselines at the canonical scale;
4. **reviewer-proofing:** the completed symmetric learning-rate robustness grid
   and a fresh matched larger-scale pair after transfer succeeds;
5. **optional broader claim:** separable 2D pre-Q/K carriers in a ViT, followed
   by a spatial DiT only if image-classification transfer succeeds.

Phases 55 and 56 close optimizer sensitivity for the current paper stage. The
prospective common recipe uses `1.2e-3`; beta, warmup, decay, and
method-specific optimization remain outside scope.
The next architecture-relevant priorities are the recognized positional
baseline table and a modern-backbone RoPE/scalar pair. FineWeb-Edu remains a
useful corpus-selection robustness test, but it follows those more diagnostic
architecture controls. No refinement arm is admitted unless new evidence
exposes a distinct, predeclared failure mode.

## Repository and storage state

- Compact phase reports and analysis JSON remain in `results/`; complete
  historical configs remain in `sweep_configs/`.
- Historical source and deleted protocols remain recoverable from git.
- On 2026-09-05, 14 redundant completed endpoint checkpoints plus one smoke
  checkpoint were deleted after verification, reclaiming about 24 GiB. Final
  model weights, evaluations, metrics, configs, and provenance remain.
- On 2026-09-11, 52 additional completed-run resume directories were removed
  from an exact manifest after verifying a completion marker and standalone
  final model for every parent. This reclaimed 93.02 GiB; active Phase-50
  rolling checkpoints and all compact evidence were protected. See
  `results/storage_cleanup_20260911/`.
- New runs default to no periodic checkpoints and no final weights. A rolling
  recovery checkpoint or final model is enabled only for a named recovery or
  downstream-analysis purpose; completed recovery state is removed after use.
- `/workspace` is not a persistent Vast volume. Irreplaceable weights must be
  copied off-box before instance recycle or destruction.
