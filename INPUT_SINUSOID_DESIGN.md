# Input sinusoid: evidence and design space

_Consolidated 2026-09-07. This note separates computations that are easy to
conflate and records the conclusions of Phases 42--44. The numerical results
are development evidence from one training seed unless stated otherwise._

## The established carriers

Let `s(p)` be the fixed model-width sinusoid and `h_l(p)` the normalized token
state entering layer `l`.

The one-shot input carrier is

```text
x_0(p) = token_projection(p) + P(s(p)).
```

It is computed once and then propagated through the residual stream. The
promoted attention-local carrier instead gives every layer a fresh positional
read:

```text
q_l(p) = W_q,l (h_l(p) + alpha_l s(p))
k_l(p) = W_k,l (h_l(p) + alpha_l s(p)).
```

Both may be followed by standard fixed RoPE. In the second expression `W_q`
and `W_k` give position separate full linear reads, but those matrices are
also used for content. They are not bespoke position projections.

Phase 43's dedicated Q/K adapters are genuinely separate position maps. They
use a shared rank-32 bottleneck and distinct Q/K output factors, either in
place of or in addition to the scalar pre-Q/K pathway. They remain static
functions of position; none conditions on token content.

## Input-map taxonomy

These parameterizations answer different questions:

| Name | Map `P(s)` | Initialization | Capacity |
| --- | --- | --- | --- |
| scalar | `alpha s` | `alpha=1` | one global amplitude |
| per-pair amplitude | `A s` | `A=I`, tied within each sine/cosine pair | spectral amplitude only |
| low-rank replacement | `U V s` | cannot equal identity when `r<d` | rank at most `r` |
| low-rank residual | `alpha s + U V s` | `alpha=1, U=0` | full-rank anchor plus rank-`r` correction |
| dense linear residual | `g s + D s` | `g=1, D=0` | unrestricted linear map |
| residual MLP | `g s + W_2 phi(W_1 s)` | `g=1, W_2=0` | nonlinear, exact-parent start |

The Phase-44 rank-32 input run was the **low-rank residual**, not a rank-32
replacement. While `alpha` is nonzero, the map `alpha I + UV` is full rank;
rank 32 limits only the learned correction. Calling it simply "a rank-32 full
projection" or ordinary LoRA would be misleading.

A separate identity gate in `g s + F(s)` is usually redundant for eventual
linear expressivity, but it is not necessarily redundant for optimization: it
provides one coherent coordinate that can suppress or restore the known-good
carrier. A multiplicative `g(s+F(s))` is a different and less diagnostic
choice because it suppresses the correction at the same time.

## Completed evidence

Phase 42 used h768/d8, context 1024, sequence batch 32, 100k steps, and seed
123. The main final holdout NLLs were:

| Arm | NLL |
| --- | ---: |
| fixed RoPE | 3.183766 |
| scalar pre-Q/K + RoPE | **3.146756** |
| fixed-gate pre-Q/K + RoPE | 3.153220 |
| global-gate pre-Q/K + RoPE | 3.147177 |
| first-layer-only pre-Q/K + RoPE | 3.149452 |
| scalar input sinusoid + RoPE | 3.153390 |
| fixed post-RoPE AddRoPE + RoPE | 3.151333 |

Thus the learned per-layer scalar mattered (`-0.006464` against its fixed-gate
ablation), but sharing one learned scalar globally was statistically tied to
separate layer gates in this seed. Repeated pre-Q/K injection beat one-shot
input injection by `-0.006635`, much less than in the earlier 30k screen.

Phase 43 used matched 20k schedules and found:

| Static pathway | NLL | Matched-parent delta |
| --- | ---: | ---: |
| rank-32 shared pre-map | 3.449950 | `-0.005127` vs scalar pre-Q/K |
| rank-32 dedicated Q/K replacement | 3.460873 | `-0.037260` vs RoPE |
| rank-32 dedicated Q/K residual | **3.445677** | `-0.009401` vs scalar pre-Q/K |

This supports bespoke positional capacity inside attention, especially
separate Q/K readouts. It does not yet show that rank 32 is the right rank or
that the improvement persists at the 100k paper horizon.

Phase 44 isolated one-shot input maps at 20k:

| Input map | NLL | Delta vs scalar input |
| --- | ---: | ---: |
| scalar control | 3.471862 | -- |
| rank-32 linear residual | 3.470090 | `-0.001772` |
| per-pair amplitude | **3.469231** | `-0.002631` |

The paired-example intervals exclude zero, but these are single-seed training
runs and neither gain reaches the Phase-43 promotion margin of `-0.003`.

The learned functions are more informative than the endpoint alone. In the
rank-32 residual, the scalar anchor fell to `0.0174`; adapter RMS was `0.1672`,
about 13.6 times anchor RMS. The model therefore used the branch almost as a
replacement rather than as a small correction. In the per-pair run, signed
amplitudes ranged from `0.0115` to `1.0356` (RMS `0.7070`), showing strong
spectral selection. Weight decay was active on these parameters, so amplitude
shrinkage should not be interpreted as pure task preference without a
no-decay control.

## Compute and parameter implications

At model width 768, one dense input map has 589,824 weights. A width-preserving
two-layer MLP has about 1.18M weights before biases; a 4x MLP has about 4.72M.
Because an input carrier is position-only and is evaluated once before batch
broadcast, these heavier maps are relatively affordable.

Inside attention, separate dense Q/K position maps would be repeated in every
block: roughly 9.44M weights across eight h768 layers. Low-rank structure is
therefore much more attractive there. The Phase-43 shared-down/separate-up
construction implements `d -> r -> 2d` with 589,824 weights across the model
at rank 32.

## Rank and optimizer scaling

Rank is not only a capacity choice. A narrow output factor has fan-in `r`, so
its variance-preserving entries and function-space scale differ from those of
a width-`d` matrix. Adam normalizes each parameter coordinate; it does not
automatically equalize the function perturbation of rank 32 and rank 128.

A useful calibration is approximately

```text
gain or output-factor LR multiplier ~= sqrt(d / r).
```

For `d=768`, this is 4.90 at rank 32 and 2.45 at rank 128. A forward gain and
an output-factor LR multiplier can look similar early under Adam, but they are
not identical: a gain also changes upstream gradients and clipping, while LR,
weight decay, and optimizer-state dynamics act directly on parameters.
Comparisons should log unclipped/clipped gradients, Adam moments, update norms,
and actual carrier-function movement.

Phase 47 implements the LR form without changing the rest of the position
module. At `d=768`, the rank-32 Q/K output factors receive 4.898979 times base
LR and rank-128 output factors receive 2.449490 times base LR. The shared down
projection and scalar anchor remain at base LR. Readout weight decay is divided
by the multiplier so AdamW's actual per-step shrink coefficient is preserved.
The comparison is considered rank-calibrated only if the median carrier-step
ratio from steps 1k--19k lies between 0.8 and 1.25.

The theoretical correction substantially improved both ranks but missed that
gate narrowly. Rank 128 still moved `1.300x` as far at the median sampled
post-warmup step. Its final NLL (`3.438322`) was better than rank 32
(`3.441269`), but that contrast cannot yet be read as a pure capacity effect.
This is useful in its own right: `sqrt(d/r)` is an effective optimization
default, not an exact function-space normalization for a trained two-factor
map. Any final rank ablation needs either empirical recalibration on a
development run or an explicitly adaptive function-step control.

A minimal empirical close-out would keep the calibrated rank-128 run fixed and
rerun only rank 32 at readout multiplier `4.898979 * 1.300 ~= 6.37`. That
number is selected from the function-step diagnostic rather than validation
loss. The rank contrast becomes interpretable only if the same predeclared
function-step gate passes; otherwise rank should be omitted as a causal claim.
Phase 48 implements this one-shot close-out at the unrounded multiplier
`6.367487`. It reached a matched post-warmup function-step ratio of `1.090`
and closed about 73% of the original rank gap. Rank 128 retained only a
`0.001058` NLL advantage, below the scout materiality margin and unreplicated
across training seeds. This completes the local LR adjustment program.

## Current interpretation and next clean tests

The main method remains scalar pre-Q/K + fixed RoPE. Phase 43 is the strongest
new design signal and merits mature confirmation before it is promoted.
Phase 44 says that one-shot input shaping is real but small; its surprising
adapter dominance makes a true dense input map scientifically cleaner than
assuming a tiny low-rank correction is sufficient.

Phase 45 performed a breadth screen before any mature confirmation. Its six
exact-parent arms were:

1. one-shot dense-linear and width-768 residual-MLP maps;
2. a dense per-layer map before the existing Q/K projections;
3. a shared dense native Q/K residual;
4. separate dense native Q/K residuals;
5. a nonlinear rank-128 shared trunk with separate Q/K readouts.

These cover the untested placement, coupling, and linearity cells directly.
They use the existing Phase-43/44 20k scalar controls because schedule, seed,
paired initialization, data order, and final holdout are identical.

The completed results were:

| Arm | NLL | Delta vs matched scalar parent |
| --- | ---: | ---: |
| input dense linear | 3.473583 | `+0.001721` |
| input width-768 MLP | 3.470171 | `-0.001691` |
| dense pre-map | 3.449165 | `-0.005913` |
| dense shared native Q/K map | **3.440949** | `-0.014128` |
| dense separate native Q/K maps | 3.441223 | `-0.013855` |
| nonlinear rank-128 native Q/K map | 3.441467 | `-0.013611` |

The three native projected-space arms are statistically tied head-to-head:
their largest endpoint separation is `0.000517`, and all paired intervals
cross zero. Each nevertheless beats the earlier linear rank-32 Q/K residual by
`0.0042--0.0047`, with intervals below zero. Dense pre-mapping is statistically
tied to rank-32 pre-mapping, so adding full-rank capacity before `W_q/W_k` did
not help. The input MLP is worse than the 384-parameter per-pair amplitude
control by `+0.000940`; heavy one-shot input maps are therefore deprioritized.

Function diagnostics confirm that the direct pathways were active rather than
winning through a dormant branch. Their direct carrier RMS is substantially
larger than the remaining scalar-anchor RMS in every layer. In the input MLP,
the scalar gate collapsed to `0.00045` while the MLP carrier RMS reached
`0.169`, so its small loss gain is not an inactivity artifact.

After this breadth filter, the most informative remaining screen is a linear
rank-128 native Q/K map in shared and separate forms. It isolates capacity from
the nonlinear rank-128 arm and asks whether the 4.7M-parameter dense shared map
can be compressed. Phase 46 completed this comparison:

| Linear rank-128 arm | NLL | Delta vs scalar parent |
| --- | ---: | ---: |
| shared Q/K readout | 3.442932 | `-0.012146` |
| separate Q/K readouts | **3.441766** | `-0.013312` |

The separate form statistically tied dense separate (`+0.000544`) and
nonlinear rank-128 (`+0.000300`), while beating linear rank 32 by `-0.003910`.
The shared rank-128 form was `+0.001166` worse than separate and `+0.001982`
worse than dense shared, with both intervals above zero. Thus a linear
rank-128 bottleneck with separate readouts is the smallest current member of
the top cluster.

However, its carrier-function updates were not scale matched: rank 128 took
3.40x larger steps than rank 32 through step 64 and 1.80x larger steps at the
median sampled post-warmup point under the same Adam LR. Phase 47 reduced the
post-warmup ratio to `1.300x`, but narrowly failed the predeclared `1.25x`
upper bound. The observed rank gain therefore still mixes capacity and
optimization. A truly matched rank comparison should precede any capacity
claim in the confirmation cohort:

1. confirm the Phase-43 dedicated Q/K residual at the paper horizon, alongside
   its scalar parent and a parameter-matched non-positional control;
2. promote only Phase-45 function classes that clear the development margin;
3. if rank itself is studied, compare rank 32 and rank 128 with both ordinary
   and explicitly calibrated update scales.

These are static position-only maps. They do not reopen token-conditioned
frequencies, EMA controllers, dynamic RoPE, or noncausal sequence reductions.

Detailed machine-readable and written reports are in
`results/phase42_paper_components/`, `results/phase43_lowrank_qk_pathways/`,
`results/phase44_input_adapters/`, `results/phase45_novel_static_maps/`,
`results/phase46_linear_rank128/`, and `results/phase47_rank_calibration/`.

## Phase 51 correction: projected readout versus rotated bias

The mature rank-32 result does not yet establish that a richly varying Fourier
map is the useful object. The actual pre-RoPE Q branch in layer `l` is

```text
z_q(p) = W_q h(p) + alpha W_q s(p) + U_q D s(p)
q(p)   = R(p) Gamma_q z_q(p) / rms(z_q(p)).
```

`Gamma_q` is the learned coordinatewise gain inside QK RMSNorm. The K branch
is analogous. A weight audit over the 1,023 positions actually entering each
attention call found that `U_q D s(p)` and `U_k D s(p)` are overwhelmingly
constant across position: their mean accounts for `94.65%--99.56%` of raw
direct-carrier energy across the mature seed-123 and seed-456 layers.

This is not equivalent to a position-independent logit bias. If
`U_q D s(p) approximately b_q` and `U_k D s(p) approximately b_k`, standard
RoPE turns the pure bias term into

```text
b_q^T R(r-p) b_k,
```

a relative Toeplitz kernel, while the mixed terms remain content dependent.
The finding nevertheless means the rank-32 model may mostly be learning Q/K
bias vectors—omitted from the backbone's bias-free Q/K projections—through an
indirect Fourier parameterization.

Phase 51 therefore freezes new mapper design and first runs trained-checkpoint
counterfactuals: full; direct mean only; direct mean removed; direct zero;
scalar zero; and both zero. QK normalization and downstream hidden states are
recomputed for every intervention. If mean-only preserves the improvement,
the next training controls are a matched constant carrier and a tiny direct
Q/K-bias parameterization. If mean removal preserves it, the centered Fourier
component remains the likely mechanism. If scalar removal is neutral only at
the endpoint but no-anchor training loses, the scalar should be described as
optimization scaffolding.
