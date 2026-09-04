# Sinusoidal intervention policy

_Active design contract, 2026-09-04. This supersedes learned-RoPE proposals in
older roadmaps; those documents and results remain historical evidence._

## Architectural boundary

Every experiment now makes two independent choices:

1. **attention backbone:** standard, fixed RoPE or NoPE;
2. **sinusoidal carrier intervention:** absent, pre-Q/K, or additive on
   projected Q/K (AddRoPE).

Learned parameters may transform the sinusoidal carrier, but may not change the
RoPE rotation. In particular, enabling AddRoPE no longer silently disables
standard RoPE. This gives clean factorial comparisons:

| Carrier | `use_rope=false` | `use_rope=true` |
| --- | --- | --- |
| absent | NoPE | standard RoPE |
| pre-Q/K | carrier + NoPE | carrier + standard RoPE |
| AddRoPE | AddRoPE + NoPE | AddRoPE + standard RoPE |

The two carrier locations remain distinct. Pre-Q/K injection computes

```text
q_p = W_q(x_p + A_q z(p)),   k_p = W_k(x_p + A_k z(p)),
```

so the existing projections jointly read content and position. AddRoPE instead
adds a transformed carrier after projection. Neither writes the positional
signal into the residual stream or V.

## Intervention axes

For Fourier pair `i`, use

```text
z_i(p) = [cos(theta_i(p)), sin(theta_i(p))]
A_i^b = a_i^b R(phi_i^b),   b in {q,k}.
```

The experiment taxonomy is:

- **backbone:** fixed RoPE or NoPE;
- **location:** pre-Q/K or additive projected Q/K;
- **component:** amplitude, phase, frequency, or coherent position warp;
- **Q/K coupling:** tied or untied;
- **granularity:** global, Fourier-pair, head, or layer;
- **dependency:** static, position-dependent, or causally content-dependent.

These axes should not be conflated. For example, an untied amplitude changes
how strongly Q and K read one shared clock, whereas untied frequencies create
different clocks. Static or token-local untied frequencies do not inherently
leak future content, but they lose the clean relative-position geometry and
are not an active priority.

Tied Q/K carrier values do not imply identical Q/K use: `W_q` and `W_k` remain
separate. Untied carrier amplitude or phase asks the narrower question of
whether the positional input itself benefits from distinct Q/K transforms.

## What the evidence currently supports

Phase 33 established the strongest active anchor at 200k steps:

- tied pre-Q/K carrier + fixed RoPE beat fixed RoPE by `-0.062831` loss;
- fixed RoPE contributed `-0.030773` relative to the same carrier with NoPE;
- separate Q/K gain, pair amplitude, and pair phase were all within about
  `0.001` of their parent and were not promoted.

Phase 34 tested one frequency bank shared by every carrier consumer. Relative
to the fixed carrier at 200k steps:

- log-frequency learning was `+0.000861`, 95% paired CI
  `[-0.000339,+0.002060]`;
- horizon-normalized frequency learning was `+0.001341`, 95% paired CI
  `[+0.000300,+0.002382]`.

The normalized coordinate removed the dangerous absolute-position derivative
but did not improve loss. This closes free per-band carrier-frequency learning
as a promotion candidate. The historical learned-RoPE calibration arm had a
small endpoint advantage, but its late advantage was collapsing and learned
RoPE is outside the active architectural boundary.

Consequently the default anchor remains the fixed tied pre-Q/K carrier with
standard RoPE. Static carrier amplitude/phase variants remain valid controlled
ablations, not established improvements. Dynamic frequency and arbitrary
tokenwise warp remain closed.

## Optimization-aware evaluation

An intervention can fail because its inductive bias is unhelpful or because its
coordinates optimize poorly. Adam does not remove this distinction. If the
carrier is `c(theta)` and the loss gradient at the carrier is `g_c`, then

```text
g_theta = J(theta)^T g_c,       J(theta) = dc/dtheta,
Delta c approximately J(theta) Delta theta.
```

Adam rescales `g_theta` using its first and second moments; it does not bound
`J Delta theta`. A rare large gradient can inflate `v` and suppress later
steps, while stale momentum can disagree with the current gradient. Therefore
learned interventions are sampled at steps 1, 2, 4, ..., 512 and every 1,000
steps thereafter. `intervention_optimization.jsonl` records, by mechanism:

- raw and clipped gradient L2/RMS/max and clipping ratio;
- Adam momentum alignment and second-moment RMS, max, and max/RMS;
- the actual parameter update and its alignment with the current descent
  direction;
- for learned carrier frequencies, endpoint-phase movement, the coordinate
  Jacobian, and the exact sinusoidal-function movement over the context;
- for static pre-Q/K and AddRoPE transforms, sampled Q/K carrier-function
  movement spanning the complete context and its ratio to parameter movement.

These are diagnostic, not automatic rejection thresholds. Interpret them with
the loss curve:

| Pattern | Likely interpretation |
| --- | --- |
| negligible gradients, updates, and functional movement | inactive or poorly initialized path |
| small parameter step but large carrier/phase step | ill-conditioned forward parameterization |
| transient gradient maxima followed by large `sqrt(v)` and tiny updates | Adam second-moment suppression |
| poor or negative descent-update/gradient alignment | momentum interference or rapidly changing objective |
| sane, sustained functional movement with no validation gain | evidence against the intervention itself |

Before pruning a new parameterization, inspect its early transient, late
optimizer state, functional step size, and validation trajectory together.

## Near-term experiment rule

Do not reopen a broad sweep immediately. New ideas should first satisfy:

1. exact no-op initialization against a fixed-carrier parent;
2. causal and KV-cache-safe dependency structure;
3. a coherent Q/K clock, or an explicit hypothesis for breaking it;
4. finite, interpretable functional derivatives across the full context;
5. one-seed 10k--20k screening with the optimization trace enabled;
6. longer or repeated runs only for a materially favorable, non-collapsing
   candidate.

The active follow-up is a rank-4 smooth static amplitude/phase deformation of
the pre-Q/K carrier, implemented with low-order DCT modes over log-frequency
index. It is screened as a tied amplitude ladder, a tied polar ladder, and an
untied Q/K polar rung under both fixed RoPE and NoPE.
