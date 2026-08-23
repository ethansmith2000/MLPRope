# MLPRope current status

_Authoritative as of 2026-08-22. This page supersedes the old sequencing in
`HANDOFF.md` and `CONSOLIDATED_RESEARCH_PLAN.md`; those files remain historical
records._

The staged next-run plan is [`NEXT_EXPERIMENT_ROADMAP.md`](NEXT_EXPERIMENT_ROADMAP.md).

## Evidence that currently matters

At 30k steps, phase 19 confirmed that a position-only mapped additive Q/K
carrier beats standard RoPE by about `0.051` mean held-out loss over paired
seeds `123/456/789`. A matched-FFN control tracked RoPE, so generic parameter
count does not explain the gain. The position-only hypernetwork and the simpler
mapped carrier were unresolved (`-0.0023`), making the mapped carrier the
economical interpretation. Content conditioning added `-0.0108` over
position-only on the disjoint holdout, favorable in all seeds, but its endpoint
ablation cost is much larger than its causal training contribution.

Phase 24 is complete at 5k steps. On the durable 256-example holdout at context
1024:

| Arm | Mean loss | Delta vs fixed RoPE |
| --- | ---: | ---: |
| fixed RoPE | 4.579567 | — |
| compact basis, anchor 0.3 | 4.472233 | -0.107333 |
| compact basis, anchor 1.0 | **4.428000** | **-0.151567** |
| full native RoPE basis, anchor 1.0 | 4.442033 | -0.137533 |

The amplitude anchor was the largest lever. The compact 16-dimensional basis
plus scalar features beat the full native RoPE embedding by `0.014033`, with
the same sign in every seed. This is a screen, not a durable positive claim.
At the exploratory 4096 context, the native-basis arm was `+0.356` worse than
fixed RoPE, so the primary in-distribution result must not be generalized to
length extrapolation.
The large run directories were intentionally deleted; logs preserve aggregate
losses but not per-example arrays, so phase-24 paired-example confidence
intervals cannot be reconstructed. See
`results/phase24_rope_embed_basis/PHASE24_RESULTS.md`.

## Closed, retained, and removed

- Static learned per-layer/per-head RoPE frequencies are closed at this scale:
  the promoted 30k result was a mixed-sign `-0.0016` mean.
- Local token-conditioned phase is closed as a material result: the best
  bounded phase-23 arm was about `-0.0021` at 5k.
- Raw content-dependent frequency multipliers are an unsafe historical
  control. In `theta_t = omega * t * exp(g_t)`, the Jacobian contains the
  factor `t`; late-position errors and gradients are amplified.
- Relative logit-bias/Inkling paths and several failed geometries were removed.
  The disabled logit config remains loadable for archived JSON compatibility.
- `transformer_old.py` and model checkpoints/output directories were removed
  intentionally. Durable configs, logs, analyses, and git history remain.
- Legacy frequency code remains only because the repository still validates
  archived experiment configs. It is not the implementation site for new
  dynamic work.

## Newly supported, untrained mechanisms

### Tied Q/K preprojection sinusoid

A frozen full-width Fourier vector `z_t` can be added only to the inputs of the
Q and K projections:

```text
q_t = W_q (x_t + alpha z_t) = W_q x_t + alpha W_q z_t
k_t = W_k (x_t + alpha z_t) = W_k x_t + alpha W_k z_t
v_t = W_v x_t
```

This gives the sinusoid a full learned linear read through the existing Q/K
matrices, keeps it out of V and the residual stream, and is exactly a tied
additive carrier after projection. It may be tested as the sole explicit PE
(`use_rope=false`) or followed by ordinary RoPE (`use_rope=true`); those are
different hypotheses and must not share a label.

### Spectrally locked causal rotary clock

The new rotary clock predicts one bounded local speed per token and either one
shared group or one group per head:

```text
s_t   = 1 + rho * tanh(f(x_<=t)),       0 < rho < 1
tau_0 = 0
tau_t = sum_{j<t} s_j
phase[t,i] = omega_i * tau_t
```

The base frequencies `omega_i` are fixed. All frequency planes in a head share
the same `tau_t`, Q and K share the same phase, and positive speed makes the
clock strictly monotone. This removes the direct `t * delta_omega_t` actuator
that made raw dynamic frequency unstable. It does not remove all cumulative
drift: a persistent speed error still accumulates, so diagnostics report speed,
clock drift, and phase drift.

The controller supports a pointwise linear/low-rank SiLU map and a short
identity-initialized depthwise causal convolution in the low-rank state. Both
have an exact fixed-RoPE anchor because the final readout is zero-initialized.
Full-sequence and incremental execution are tested for equivalence, and prefix
invariance is tested to prevent future-token leakage.

A learned EMA/linear-RNN backend is deliberately deferred. The geometry and
controller are separated in `position/clock.py` and `position/temporal.py`, so
an associative-scan or custom-kernel backend can be added without changing the
clock definition. It should be added only with forward/backward parity,
incremental-state parity, compile, and throughput tests; PyTorch's currently
private prototype scan is not yet a dependency of the training path.

This is a controlled implementation of an explored family, not a novelty
claim. CARoPE, Selective RoPE, and PaTH already establish accumulated
content-dependent rotations at different levels of constraint. The present
clock is intentionally narrower: fixed standard spectrum, one bounded speed,
exact RoPE anchor, and ordinary softmax SDPA. Recent theory also predicts that
rotation-only accumulated transforms eventually lose far-mass control, so this
is not proposed as an unlimited-length solution.

## Current taxonomy

Keep these axes separate in names and comparisons:

| Axis | Examples | What it changes |
| --- | --- | --- |
| Injection site | residual, pre-Q/K projection, post-Q/K additive, rotation | where position enters |
| Geometry | Euclidean addend, polar additive carrier, orthogonal rotation | algebra of the intervention |
| Actuator | amplitude, phase offset, static spectrum, cumulative warp | which degree of freedom moves |
| Conditioning | position-only, token-local, short causal summary | information available to the mapper |
| Sharing | Q/K, head, frequency-plane, layer | which branches use one signal |
| Mapper | linear, low-rank nonlinear, causal convolution, future scan | how the signal is computed |

“Amplitude” should not be collapsed into temperature. For an additive carrier
it changes content-carrier cross terms and the carrier Gram term. For
multiplicative Q/K gains, query gain is row sharpness/temperature-like, while
key gain is token salience.

## Experimental priority

Phase 24's large anchor-1.0 screen result is now the first promotion target.
The earned primary attribution experiment remains the position-only
gain/salience decomposition: query-only, key-only, and both. The preprojection
sinusoid is a clean architectural comparator. The rotary clock is worth one
small, decisive mechanistic screen with a weak prior, not a broad mapper sweep.
Start pointwise versus short causal-convolution clocks; do not add EMA kernels,
frequency-wise clocks, or combinations with additive carriers until a simpler
clock shows a meaningful signal.
