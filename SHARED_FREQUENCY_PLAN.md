# Globally shared frequency experiment (completed)

_Historical Phase-34 decision and protocol, 2026-09-03; results added
2026-09-04. Current interventions are carrier-only; see
[`SINUSOID_INTERVENTION_POLICY.md`](SINUSOID_INTERVENTION_POLICY.md)._

## Result

All five arms completed at 200k steps. Relative to the fixed pre-Q/K carrier,
learned-log frequency was `+0.000861` worse with 95% paired CI
`[-0.000339,+0.002060]`; horizon-normalized frequency was `+0.001341` worse
with CI `[+0.000300,+0.002382]`. Neither clears the promotion gate.

The learned-RoPE calibration was `-0.001431` at the endpoint, below the
`0.002` threshold, and its late advantage was shrinking. That arm is preserved
in the report but learned-RoPE support has been removed from the active model.
See
[`results/phase34_shared_frequency_200k/PHASE34_RESULTS.md`](results/phase34_shared_frequency_200k/PHASE34_RESULTS.md).

## Research question

Can frequency learning help when it changes one position coordinate system
coherently, rather than allowing every layer, head, Q/K branch, or token to
invent a different clock?

This was a static experiment. The learned frequencies depend only on global
parameters and position, never on token content. It therefore introduces no
future-token leakage and does not invalidate causal KV caching.

## Why this branch is narrowly reopened

The earlier local implementation learned

```text
omega[layer, head, pair] = omega_base[pair] * exp(delta)
```

and later tested direct additive frequencies. The resulting table was more
granular than necessary, and the decisive follow-up used a different, unstable
parameterization. Those experiments remain evidence against unconstrained
local frequency freedom.

[LeRoPE (2026)](https://arxiv.org/abs/2607.10134) supplies new, directly
relevant evidence: it learns one scalar per RoPE band, shares the same bank
across every layer and head, and reports consistent gains from 52M to 2.5B
parameters. Its per-layer/per-head ablation did not improve on global sharing.
The paper also explicitly derives the relative-distance factor in frequency
gradients, keeps the frequency parameters out of weight decay, and clips them
independently. This evidence was not represented in the active roadmap.

Long-context methods such as [LongRoPE](https://arxiv.org/abs/2402.13753) and
[CLEX](https://arxiv.org/abs/2310.16450) provide additional evidence that
coordinated spectral rescaling is a real lever, although their objective is
context extension rather than our in-distribution training comparison.

## Geometry

Ordinary RoPE uses

```text
theta_i(p) = p * omega_i.
```

A phase shared by Q and K cancels from their relative angle. A shared frequency
change does not:

```text
Delta theta_i(p,q) = (q-p) * omega_i'.
```

Consequently frequency is the substantive rotary intervention. It must be
shared across Q and K: separate frequencies would produce
`q*omega_i^k - p*omega_i^q`, which depends on the two absolute positions and
does not define one relative coordinate system.

The pre-Q/K carrier uses the same principle. One full-width bank constructs
`[cos(p*omega_i), sin(p*omega_i)]` once per forward pass. Every layer receives
that exact carrier; the existing per-layer scalar gates and Q/K projection
matrices remain the readout mechanism.

## Parameterizations

### Positive log frequency

```text
omega_i' = omega_i * exp(alpha_i),  alpha_i = 0 at initialization.
```

This exactly recovers the fixed bank at initialization and keeps every
frequency positive. One `alpha_i` vector is shared by all consumers.

### Horizon-normalized endpoint phase

```text
omega_i' = omega_i + rho_i / L_ref
theta_i(p) = p*omega_i + (p/L_ref)*rho_i.
```

`rho_i` is the extra phase accumulated at the reference horizon. For
`0 <= p <= L_ref`, `|d theta_i(p)/d rho_i| <= 1`; the raw position multiplier
cannot amplify this parameter's gradient. This is an unsaturated coordinate,
not a `tanh` bound. It permits zero crossing, so nonpositive frequencies and
frequency-order violations are mandatory diagnostics rather than silently
forbidden outcomes.

Both modes keep their parameters and trigonometry in fp32 under bf16 training.
Frequency coordinates receive no weight decay and are clipped as their own
gradient group, separately from ordinary model parameters.

## Phase-33 prerequisite

The six 200k runs completed before this protocol was frozen:

| Contrast | Final delta | Paired-example 95% CI |
| --- | ---: | ---: |
| tied pre-Q/K + RoPE vs fixed RoPE | `-0.062831` | `[-0.064426,-0.061237]` |
| tied pre-Q/K + RoPE vs tied without RoPE | `-0.030773` | `[-0.032424,-0.029123]` |
| split Q/K scalar vs tied | `+0.000990` | `[-0.000096,+0.002076]` |
| pair amplitude vs split scalar | `-0.000116` | `[-0.001212,+0.000979]` |
| pair phase vs pair amplitude | `+0.000165` | `[-0.000847,+0.001178]` |

Thus Phase 34 keeps fixed RoPE and the tied carrier, and drops the unsupported
Q/K-split, pair-amplitude, and pair-phase axes. The complete report is
[`results/phase33_static_qkpre_200k/PHASE33_RESULTS.md`](results/phase33_static_qkpre_200k/PHASE33_RESULTS.md).

## Phase-34 matrix

All five arms use h768/d8, eight heads, context 1024, batch 8, seed 123, paired
initialization, one common 200k schedule, and disjoint development/final
holdouts.

| Arm | RoPE bank | Pre-Q/K carrier bank | Primary role |
| --- | --- | --- | --- |
| `rope-fixed` | fixed | absent | LeRoPE-style reference |
| `rope-global-log` | globally shared learned log | absent | external-method calibration |
| `qkpre-fixed` | fixed | fixed, tied Q/K | best Phase-33 anchor |
| `qkpre-frequency-log` | fixed | globally shared learned log | new intervention locus |
| `qkpre-frequency-horizon` | fixed | globally shared endpoint phase | normalized optimization contrast |

Do not combine learned RoPE and a learned carrier in this stage. A combination
is justified only if both separate contrasts win.

## Readout and decision gate

Evaluate every 10k steps and run all finite arms to 200k. The primary endpoint
is paired loss on the 1,024-example holdout beginning at validation batch
2,048. Report:

- learned-frequency candidate minus its paired fixed reference;
- horizon carrier minus log carrier;
- complete development curves and late-window gap slope;
- frequency multipliers and endpoint phase displacement;
- nonpositive-frequency and order-violation fractions;
- the complete learned spectrum, throughput, memory, and wall time.

A candidate earns seed replication only if its 200k improvement is at least
`0.002` nats, its paired-example 95% interval excludes zero, its late gap is not
clearly collapsing, and its spectrum remains numerically usable. This is a
screening rule, not a substitute for additional training seeds.

## Deferred extensions

- smooth spline or order-preserving deformation over log frequency;
- one dedicated relative Fourier logit-bias band;
- combining learned RoPE with a learned pre-Q/K carrier;
- per-layer, per-head, or Q/K-separate frequency banks;
- content-dependent or cumulative coordinate warps.

The first two become interesting only after a globally shared free bank shows
that frequency adaptation is useful in this training regime.
