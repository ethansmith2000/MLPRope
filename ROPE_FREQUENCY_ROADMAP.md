# Learned RoPE frequency roadmap

> Current update (2026-09-04): learned RoPE is closed and removed from the
> active runtime. Phase 34's globally shared carrier-frequency variants did not
> improve the fixed carrier. RoPE is now immutable standard RoPE or disabled;
> all learned positional changes act on a separate sinusoidal carrier. See
> [`SINUSOID_INTERVENTION_POLICY.md`](SINUSOID_INTERVENTION_POLICY.md). The
> remainder of this document is the historical decision record.

_Consolidated 2026-08-03. Eval loss at context 1024 is the sole primary
endpoint. Length extrapolation is not a current hypothesis._

## Current evidence

Phase 20 tested direct, static frequency learning with

```text
omega[layer, head, pair] = omega_base[pair] * exp(delta)
```

at h768/d8 for 5,000 steps and three paired seeds. Both learned schedules won
against fixed RoPE in every seed, but only by about `0.0019`; layer-head and
layer-shared were tied:

| Contrast | Mean candidate-minus-reference loss |
| --- | ---: |
| layer-shared vs fixed | -0.001921 |
| layer-head vs fixed | -0.001883 |
| layer-head vs layer-shared | +0.000038 |

The parameters were active (frequency multipliers reached roughly
`0.75-1.16`), so this was not a dead-gradient result. It says that unconstrained
static per-pair freedom is not materially useful under the present 5k recipe;
it does not distinguish an optimization problem from a bad inductive bias.

Phase 21 then compared static parameterizations across the same three seeds.
Free additive frequency (`omega = omega0 + u`) improved 5k loss by `0.011992`
on average and cleared the screen gate; full identity-backward exponential
improved by `0.004541`, while softplus and bounded-log variants remained near
the phase-20 null. Phase 22 promoted fixed vs additive to 30k and refuted the
5k result:

| Seed | Additive minus fixed at 30k |
| --- | ---: |
| 123 | -0.004342 |
| 456 | +0.004510 |
| 789 | -0.004947 |
| mean | -0.001593 |

The additive spectra stayed finite and throughput was neutral, so this is a
scientific null rather than an implementation failure. Static base-frequency
learning is closed at this scale/horizon unless new evidence appears. The
structured-static stage is skipped: constraining a direct 384-parameter table
that already failed at 30k has low expected value.

Earlier content-conditioned frequency multipliers were much worse than their
control (`4.5369` / `4.7788` vs `4.2840`). Their phase perturbation was
proportional to absolute position. That failure remains evidence against a raw
token-dependent `omega = omega_base * exp(delta(x))`; any new dynamic arm must
either bound phase at the training horizon or serve as an explicit unsafe
control.

Phase 23 tested a safer token-conditioned form,
`delta_phase=(t/1024)*tanh(mapper(norm_x))`, using three paired seeds. Full
linear was worse than fixed RoPE (`+0.003057` mean). Rank-32 linear and SiLU
were favorable in all seeds but only by `-0.001893` and `-0.002063`; neither
cleared the locked `-0.01` screen gate. Most low-rank outputs remained away from
the nominal one-radian boundary (maximum per-layer phase p95 `0.32-0.42` rad),
but rare extrema approached it. The dynamic branch therefore stopped under the
precommitted rule. Whether the arbitrary one-radian/tanh trust region obscured
a useful free controller remains an explicit design question rather than a
positive result.

All RoPE positions, frequencies, angle products, and trigonometry now remain in
fp32 under autocast and after model dtype conversion. The result is cast only
when it is applied to Q/K.

## The two hypotheses

These should remain separate in code, naming, and conclusions.

### A. A learned function for the static spectrum

The frequency is fixed after training for a given layer, head, and rotary pair:

```text
omega[l,h,i] = transform(omega_base[i], f(layer_l, head_h, log_omega_i))
```

This can preserve exact translation relativity, strict rotation, and fused
SDPA. Direct per-pair parameters from phase 20 are the capacity ceiling. A
function is interesting only as a coordination or regularization prior, not as
a cheaper replacement: the direct table was already tiny.

### B. A token-conditioned rotation

The phase or frequency also depends on token content:

```text
omega[b,l,h,t,i] = transform(omega_static[l,h,i], g(x[b,t]))
```

This tests content-position interaction, not merely a better RoPE schedule. If
the content delta is multiplied by position, shifting an otherwise identical
token sequence changes the phase by an additional term proportional to the
shift. Q/K sharing does not remove this problem because different tokens
usually predict different deltas.

A content-dependent phase residual is an important structural control:

```text
phase[b,l,h,t,i] = position[t] * omega_static[l,h,i] + delta_phase(x[b,t])
```

Unlike a content-dependent frequency, the residual does not grow with position
and remains equivariant when the entire content sequence is shifted. It is not
a learned frequency, and earlier phase-only rotary arms were poor, but it tells
us whether a dynamic result needs the `position * delta_omega` interaction.

## Shapes and terminology

Let model width be `D`, number of heads `H`, head width `d=D/H`, and rotary-pair
count per head `F=d/2`.

- One valid frequency or phase value is required per pair, not per scalar Q/K
  coordinate.
- A Q/K-shared, per-head controller emits `H*F = D/2` values per token.
- A Q/K-separate controller emits two `D/2` tensors, so concatenating its
  outputs gives `D` values. In that precise sense `Linear(D, D)` is the natural
  full Q/K-separate mapper; the shared counterpart is `Linear(D, D/2)`.
- Emitting unrelated deltas for both coordinates inside a rotary pair is not a
  frequency change and should be rejected by the implementation.

Here `norm_x` means the full layer-normalized residual vector passed to
`Attention`. It retains content direction and is a valid controller input. Its
scalar L2/RMS norm is nearly constant after LayerNorm and is not informative.
If a scalar magnitude controller is desired, it must instead read the pre-norm
residual stream or the raw Q/K projections before QK normalization.

## Controller inputs

Keep the input choice explicit; it changes the hypothesis.

| Source | Shape presented to controller | Interpretation |
| --- | --- | --- |
| normalized residual (`norm_x`) | `[B,L,D]` | token direction, already available in `Attention` |
| pre-norm residual RMS | `[B,L,1]` | residual magnitude; requires passing pre-norm state |
| pre-QKNorm Q/K RMS | `[B,H,L,1]` | head-local projected magnitude |
| raw projected Q/K | `[B,H,L,d]` | branch- and head-local content; highest coupling |
| static coordinates | layer/head/log base frequency | static-spectrum function, no token dependence |

The first dynamic screen should use normalized residual input. Magnitude-only
sources are cheap follow-ups, not aliases for it.

## Mapper families

For normalized residual input, compare the user's proposed family directly:

1. `linear`: `Linear(D, O)`.
2. `low_rank_linear`: `Linear(D, r) -> Linear(r, O)`.
3. `low_rank_silu`: `Linear(D, r) -> SiLU -> Linear(r, O)`.

Use `O=D/2` for Q/K-shared per-head outputs and `O=D` for Q/K-separate outputs.
Start with `r=32`; change rank only after a mapper has earned a follow-up.

At h768/d8, including biases and using a separate controller in every one of
the eight layers, the initial cost is:

| Mapper | Shared-Q/K parameters | Separate-Q/K parameters |
| --- | ---: | ---: |
| full linear | 2,362,368 | 4,724,736 |
| rank-32 linear or SiLU | 298,240 | 399,616 |

The low-rank separate count uses one shared down projection and one `D`-wide
output projection. Giving Q and K independent down projections is a later,
strictly more expensive coupling choice.

Initialization must produce exact standard RoPE:

- `linear`: zero output weight and bias. It is an exact null and the weight
  receives gradients immediately.
- factorized mappers: initialize the down projection normally and the up
  projection to zero. This is an exact null; the up projection learns first.
- use rank-independent initialization/scaling so changing `r` does not repeat
  the phase-14 rank/initialization confound.

For magnitude-only sources, first use a scalar gate times a learned spectral
readout. A general MLP is unnecessary until this interpretable version fails.

Static functions should be tried from most constrained to least constrained:

1. affine in centered `log(omega_base)`, with layer/head coefficients;
2. a small fixed spectral basis or 8-knot spline;
3. a small MLP over layer, head, and `log(omega_base)` coordinates.

Before training them, fit these functions to the phase-20 learned deltas. If a
family cannot reconstruct the small learned adjustment, it is not a plausible
coordination prior.

## Output parameterizations

Parameterization and backward behavior are separate axes. Log both explicitly.

| Name | Forward map | Relevant property |
| --- | --- | --- |
| ordinary exponential | `omega0 * exp(u)` | positive; gradient is scaled by current `omega` |
| multiplier identity-backward exp | `omega0 * exp_ste(u)` | same forward; derivative w.r.t. `u` is `omega0` |
| full identity-backward exp | forward `omega0 * exp(u)`, backward `d omega / d u = 1` | removes both exponential and base-frequency gradient scaling |
| bounded log multiplier | `omega0 * exp(B_log * tanh(u))` | positive and limits multiplicative drift |
| normalized softplus multiplier | `omega0 * softplus(a+u)/softplus(a)` | positive, exact anchor, different local curvature |
| free additive frequency | `omega0 + u` | simplest gradient; permits zero crossings and reversal |
| horizon-bounded frequency | `omega0 + B_phase/L_ref * tanh(u)` | extra phase is at most `B_phase` at `L_ref` |
| free horizon-normalized phase | `p*omega0 + (p/L_ref)*u(x)` | no artificial output boundary; extra phase can wind and grows beyond `L_ref` |
| rationally bounded phase | `p*omega0 + (p/L_ref)*B*u/sqrt(1+u^2)` | strict smooth bound with polynomial gradient tails; boundary is still a prior |
| STE-clamped phase | forward clamp of `(p/L_ref)*B*u`, identity backward | hard forward trust region without saturation gradients; raw values can drift outside it |
| bounded phase residual | `p*omega0 + B_phase*tanh(u)` | no position-amplified delta; dynamic-phase control |

The existing `exp_ste` utility implements the multiplier identity-backward
case, not the full identity-backward frequency transform. The latter needs a
separate custom autograd operation around the complete map.

Phase 23 used the horizon-bounded form with `L_ref=1024`. That was a conservative
screening prior, not a geometrically required range. If the dynamic branch is
reopened, first isolate boundedness with the best existing rank-32 SiLU mapper:
compare the existing `tanh` result with a free horizon-normalized phase. A
rational squash and an STE clamp answer narrower optimizer questions but do not
resolve whether any boundary belongs in the hypothesis. Ordinary exponential
remains only an unsafe historical control.

## Coupling and dimensionality order

Do not factorially cross mapper, parameterization, Q/K coupling, head coupling,
and rank.

1. Start with one controller per layer, per-head outputs, and Q/K-shared deltas.
   This addresses the original per-head expressivity question while changing
   one coupling axis at a time.
2. If a dynamic mapper wins, compare head-shared (`O=F`) with per-head
   (`O=H*F`) output. Phase 20 gives a real prior that head-specific static
   spectra may be unnecessary.
3. Only then compare Q/K-shared with Q/K-separate readouts. Separate frequencies
   add asymmetry but also introduce another absolute-position term even for a
   common static input.
4. Sweep rank around a winning low-rank mapper (`8`, `32`, `128`) last. Include
   the full linear map as a capacity ceiling, not as the default.
5. Cross-layer controller sharing is a later efficiency experiment. Initially,
   layer-specific controllers avoid conflating depth conditioning with token
   conditioning.

## Staged experiment plan

### Stage 0 — numerical and offline checks

- Fit affine, spline, and small-coordinate-MLP descriptions to phase-20 learned
  frequency deltas.
- Compare gradient scale across rotary bands for ordinary exp, multiplier STE,
  full identity-backward exp, softplus, and additive output.
- Estimate parameters and FLOPs for `D -> D/2`, `D -> r -> D/2`, and the SiLU
  version at h768/d8.
- Lock one config schema before adding launch families.

No loss claims come from this stage.

The intended config surface is one structured object rather than more unrelated
top-level flags:

```yaml
rope_frequency:
  mode: fixed                 # fixed | static | content
  source: normalized_residual # direct | static_coordinates | normalized_residual | pre_norm_rms | qk_rms | projected_qk
  mapper: low_rank_silu       # direct | affine | spline | linear | low_rank_linear | low_rank_silu
  rank: 32
  parameterization: horizon_bounded # exp | exp_multiplier_ste | exp_full_ste | bounded_log | softplus | additive | horizon_bounded | phase_residual
  qk_coupling: shared         # shared | separate
  head_coupling: per_head     # shared | per_head
  log_bound: 1.0
  phase_bound: 1.0
  reference_length: 1024
```

Validation should make irrelevant fields illegal rather than silently ignoring
them.

### Stage 1 — static parameterization screen

_Completed in phases 21-22. Additive passed at 5k but failed paired 30k
confirmation; the static branch is closed._

Hold the phase-20 layer-shared representation fixed and compare:

- fixed RoPE;
- ordinary exp (reuse phase-20 results);
- full identity-backward exp;
- normalized softplus;
- free additive frequency;
- bounded log multiplier.

One paired seed may eliminate unstable arms, but it may not rank close arms.
Run three paired seeds for any decision. This stage asks whether phase 20 was
limited by log-space optimization, not whether more parameters help.

### Stage 2 — structured static spectrum

_Skipped after phase 22. Reopen only with a concrete inductive-bias argument
that predicts a 30k improvement despite the direct-table null._

Using the best-behaved parameterization from stage 1, compare direct
layer-shared deltas with the affine and spline/static-basis functions. Add the
coordinate MLP only if offline reconstruction indicates that it expresses
meaningfully different structure.

This stage ends the static branch. A result around the phase-20 `0.002` gain is
recorded as null rather than promoted.

### Stage 3 — dynamic source and mapper screen

_Completed in phase 23; no arm cleared the promotion gate._

The locked screen held Q/K-shared, per-head outputs, `r=32`, normalized-residual
input, and horizon-bounded frequency fixed. It compared fixed RoPE with full
linear, low-rank linear, and low-rank SiLU controllers:

| Mapper | Mean candidate-minus-fixed loss | Seed signs |
| --- | ---: | --- |
| full linear | +0.003057 | all unfavorable |
| rank-32 linear | -0.001893 | all favorable |
| rank-32 SiLU | -0.002063 | all favorable |

The precommitted rule required at least `-0.01`, so source, rank, head, Q/K, and
phase-residual axes were not opened. The result closes this particular bounded
screen, not every possible dynamic phase map. The saved-checkpoint zero,
sequence-mean, and token-shuffle ablations should precede any new training. A
free-vs-bounded output audit is conceptually clean but lower-value than first
determining whether the existing controller uses same-token alignment at all.

A causal cumulative content clock is a more structurally distinct hypothesis:
for a positive token-dependent increment `d_t`, use
`tau_t=sum_{s<=t} d_s` and phase `omega_i*tau_t`. Phase differences then depend
on the content increments along an interval rather than the absolute position
index. Calling this a monotone time warp requires the positivity constraint;
an unconstrained per-pair cumulative delta does not earn that interpretation.
This is not a phase-23 follow-up unless the checkpoint ablations reveal a
material token-aligned effect and the higher-priority headline confirmation is
settled.

### Stage 4 — coupling and dimensionality

Only for a stage-3 winner:

- head-shared vs per-head output;
- Q/K-shared vs Q/K-separate readout;
- ranks `8/32/128` vs full linear where relevant;
- optional pre-QKNorm projected content instead of normalized residual input.

Promote axes sequentially. Do not launch their Cartesian product.

### Stage 5 — longer-horizon and composition gate

A 30k run is earned only by a material, reproducible 1024-context result. The
minimal promoted set is:

- fixed RoPE;
- best static-frequency control;
- best dynamic-frequency arm;
- position-only additive carrier;
- dynamic frequency plus the position-only additive carrier, if the standalone
  mechanism survives.

The combined arm asks whether frequency adaptation adds decay-profile control
that the additive amplitude mechanism does not already supply.

## Training and evaluation discipline

- Screening recipe: h768/d8, batch 8, 5,000 optimizer steps, 200-step warmup,
  then linear learning-rate decay to zero, peak LR `3e-4`, AdamW betas
  `0.9/0.98`, weight decay `0.01`.
- Primary endpoint: paired held-out loss at context 1024 only.
- Use paired initialization and data order for seeds `123/456/789`.
- A single seed is a smoke or failure screen, never evidence for a small
  ordering. Five-thousand-step differences below about `0.02` have historically
  failed to predict 30k ordering.
- Promotion requires mean gain of at least `0.01`, consistent seed signs, sane
  spectra/phase diagnostics, and an acceptable measured throughput cost.
- Measure steady-state throughput separately from warmup and report added
  parameter count. Full linear controllers require a parameter/capacity control
  if they appear competitive.
- Do not add 2048/4096 cells to test extrapolation. They are outside the current
  hypothesis.

## Required diagnostics and ablations

Every trained dynamic arm should log, by layer and head:

- frequency multiplier or additive delta mean, RMS, p95, min, and max;
- actual extra phase `position * delta_omega` at positions 256, 512, and 1024;
- minimum log-frequency spacing and near-duplicate/collapsed pair counts;
- controller output variance across tokens and correlation with token position;
- Q/K disagreement for separate readouts;
- controller gradient and update RMS by spectral band;
- steady-state tokens/s and positional parameter count.

At evaluation, run three forward ablations without retraining:

1. zero the controller output (returns to the learned static anchor);
2. replace token features with their sequence mean;
3. shuffle controller features across token positions while keeping language
   model activations fixed.

The zero ablation measures whether the dynamic path contributes at all. The
mean and shuffle ablations distinguish a generic learned schedule from genuine
token-conditioned use; their losses are mechanism diagnostics, not primary
model comparisons.

## Engineering gate

Before queueing training, tests must establish:

- exact fixed-RoPE equality at null initialization in fp32 and under autocast;
- fp32 controller-to-angle arithmetic and fp32 trig, with cast only at use;
- valid pairwise output shapes for shared/separate Q/K and shared/per-head modes;
- nonzero gradients for every intended mapper and parameterization;
- causality/locality: a controller uses only the token whose Q or K it rotates;
- bounded-phase guarantees at position 1024;
- fused SDPA remains active;
- config validation rejects incompatible or unknown combinations;
- checkpoint save/load and frequency diagnostics cover every mode.

All GPU work must be submitted through `gpu-claim`. For the experiment batch
started on 2026-08-04, MLPRope may use at most two GPUs concurrently because a
higher-priority project is allocated the other six. This is a temporary
allocation, not a permanent project-wide cap. Long sweeps should run under
supervisor so queueing and terminal disconnects are harmless.

## Default decisions unless new evidence changes them

- Interpret `norm_x` as the normalized residual vector, not its scalar norm.
- Use Q/K-shared, per-head outputs first.
- Use rank 32 for the first factorized mapper.
- Phase 23 has completed the first horizon-bounded screen; do not treat its
  one-radian `tanh` boundary as a settled property of phase.
- Keep bounded phase residual as a named structural control.
- Keep ordinary dynamic exp only as a bridge to the historical negative.
- Treat `0.01` as the minimum material 1024-context gain.
- Always use `gpu-claim`; the two-GPU cap applied to phases 22-23 and is not a
  permanent project-wide allocation.
- Do not run a factorial sweep and do not optimize for length extrapolation.
