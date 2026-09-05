# Sinusoidal intervention policy

_Active design contract, 2026-09-05. Historical implementations and protocols
are available from git; their compact evidence remains in `results/`._

## Architectural boundary

Every experiment makes two independent choices:

1. **backbone:** standard fixed RoPE or NoPE;
2. **carrier:** absent, injected before Q/K projection, or added to projected
   Q/K (AddRoPE).

| Carrier | `use_rope=false` | `use_rope=true` |
| --- | --- | --- |
| absent | NoPE | standard RoPE |
| pre-Q/K | carrier + NoPE | carrier + RoPE |
| AddRoPE | AddRoPE + NoPE | AddRoPE + RoPE |

No active intervention changes the RoPE rotation. Neither carrier writes to V
or the residual stream.

## Promoted pre-Q/K method

For block-normalized content `x_p` and fixed full-width Fourier features
`z(p)`, each layer computes

```text
q_p = W_q(x_p + alpha z(p))
k_p = W_k(x_p + alpha z(p))
v_p = W_v x_p
```

The scalar `alpha` is shared between Q and K within a layer, initialized to
1.0, and learned per layer. The positional input is tied, but its use is not:
`W_q` and `W_k` provide separate learned linear reads of content and position.
Standard RoPE is then applied to the normalized Q/K heads when enabled.

The gate and Fourier table remain fp32 under model-wide bf16/fp16 conversion.
The completed carrier is cast to the activation dtype before addition. A
generic `position_lr_multiplier` may change the optimizer LR for positional
parameters; the default is 1.0.

## Why the method stays scalar

Phase 33 established at 200k steps that pre-Q/K + RoPE beats fixed RoPE by
`-0.062831`, and that RoPE adds `-0.030773` beyond the carrier alone. Separate
Q/K gains, pair amplitudes, and pair phases were all within roughly `0.001` of
their parents.

Phase 34 found learned shared carrier frequency null or harmful at 200k.
Horizon-normalized coordinates fixed the raw absolute-position derivative but
did not improve loss. Phase 37 then paired scalar and two rank-4 smooth
amplitude parameterizations for 200k; both were null on the disjoint primary
holdout despite healthy, substantial functional movement.

Therefore the active pre-Q/K runtime contains only `tied_scalar`. Frequency,
phase, spectral shape, and Q/K-untying are closed unless a new structural
hypothesis—not merely a new parameterization—justifies reopening them.

## AddRoPE boundary

AddRoPE remains a distinct post-projection carrier:

```text
q_p = W_q x_p + e_q(p)
k_p = W_k x_p + e_k(p)
```

The active generic channel supports the static additive carrier and retains
the pointwise content-conditioned reference because it produced a replicated
30k increment. EMA/scan conditioning and content-dependent RoPE frequency are
removed. AddRoPE and pre-Q/K may be crossed explicitly, but their 15k result
was sub-additive, so another combination run needs a mature factorial reason.

## Causality rule

Static parameters and token-local functions are causally safe. Any
sequence-dependent controller must ensure that position `p` reads only tokens
at or before `p` and must have a well-defined incremental KV-cache update. A
full-sequence reduction, bidirectional convolution, or noncausal normalization
would leak future content even if its output were described as a positional
parameter.

This rule is necessary but not sufficient: tokenwise frequency changes can be
causal while still destroying a coherent shared clock and producing phase
sensitivity proportional to absolute position.

## Optimization-aware evaluation

Adam normalizes coordinate gradients, not the functional Jacobian. For carrier
parameters `theta`,

```text
g_theta = J(theta)^T g_carrier
Delta carrier approximately J(theta) Delta theta.
```

Sparse optimizer diagnostics therefore record raw/clipped gradients, Adam
moment alignment and second moments, actual parameter updates, and sampled
carrier-function movement. These distinguish an inactive or ill-conditioned
intervention from a healthy scientific null.

## Experiment rule

Local carrier-shape search is paused. Evidence-strengthening experiments should
hold the method fixed and test:

- paired mature-seed replication;
- QKNorm/normalization robustness;
- model-scale transfer;
- evaluation-length behavior and attention mechanisms.

Any future architectural proposal must have exact parent initialization,
causal and KV-cache-safe dependencies, interpretable derivatives over the full
context, and a predeclared promotion gate before training.
