# Phase-23 bounded content-conditioned RoPE screen

_Locked before launch on 2026-08-05; completed the same day._

## Question

Can a token-local function of the normalized residual improve RoPE by changing
each head's rotary frequencies within a strict phase-at-horizon bound?

This is distinct from the static-frequency branch, which phases 20-22 closed at
30k. It is also distinct from the old unbounded content-frequency multiplier:
the new controller can add at most one radian of phase at position 1024.

## Fixed mechanism

For normalized residual vector `x[b,t]`, each layer predicts one value per head
and rotary pair:

```text
raw[b,h,t,i] = mapper(norm_x[b,t])
delta_phase[b,h,t,i] = (t / 1024) * tanh(raw[b,h,t,i])
phase = t * omega_base[i] + delta_phase
```

- Controller input: the full LayerNorm output already passed to `Attention`,
  shape `[B,L,D]`; this is not its nearly constant scalar norm.
- Output: per-head `[B,H,L,F]` phase deltas.
- Q/K coupling: shared. The same token-local phase rotates Q and K.
- One independent controller per transformer layer.
- Phase bound: `1.0` radian at position 1024; proportionally smaller earlier.
- Exact fixed-RoPE anchor through zero output initialization.
- Strict rotation, fused SDPA, and fp32 phase/trigonometric arithmetic.

Token-dependent frequency still does not preserve joint shift equivariance:
the `t * delta_omega(x)` term depends on absolute position. The bound controls
its magnitude rather than changing that fact.

## Arms

The completed phase-20 fixed runs are reused as three-seed controls. New arms:

1. `linear-horizon`: `Linear(D,D/2)`.
2. `lowrank-linear-horizon`: `Linear(D,32) -> Linear(32,D/2)`.
3. `lowrank-silu-horizon`: `Linear(D,32) -> SiLU -> Linear(32,D/2)`.

At h768/d8 these add 2,362,368 parameters for full linear and 298,240 for each
rank-32 mapper. The full map is a capacity ceiling. Factorized maps use a
randomly initialized down projection and zero up projection; full linear uses a
zero weight and bias. All are exact nulls, with the output projection receiving
gradient on the first step.

A bounded content-phase residual is not included yet. It will be run using the
best mapper only if a horizon-frequency arm first shows a material effect; this
keeps mapper and dynamic parameterization from becoming a factorial sweep.

## Locked training protocol

- h768/d8, 8 heads, context 1024, batch 8.
- 5,000 steps; 200-step warmup to `3e-4`, then linear decay to zero.
- AdamW betas `0.9/0.98`, weight decay `0.01`; zero-anchored controller
  parameters excluded from weight decay.
- Seeds and paired initialization seeds: `123`, `456`, `789`.
- Development: 25 batches at step 5,000.
- Final primary holdout: 256 batches beginning at batch 2,048, context 1024.
- No long-context evaluation.
- Final weights saved, no intermediate optimizer checkpoints.

## Decision rule

This is a failure/materiality screen, not a durable-result claim. Promote at
most one mapper if its mean 1024 improvement is at least `0.01`, all seed signs
agree, controller outputs are content-varying rather than constant, phase bounds
hold, and throughput/parameter cost is acceptable. Any promoted mechanism must
subsequently survive a paired longer-horizon confirmation.

If no mapper clears the gate, do not open Q/K separation, head sharing, source,
rank, or phase-residual axes.

## Results

Candidate-minus-fixed deltas on the disjoint 256-batch context-1024 holdout:

| Arm | Seed 123 | Seed 456 | Seed 789 | Mean |
| --- | ---: | ---: | ---: | ---: |
| full linear | +0.000489 | +0.004463 | +0.004217 | +0.003057 |
| rank-32 linear | -0.001351 | -0.001522 | -0.002807 | -0.001893 |
| rank-32 SiLU | -0.001100 | -0.002135 | -0.002954 | -0.002063 |

Full linear was uniformly worse. Both low-rank maps had favorable signs in all
seeds, but their effects were about one fifth of the materiality threshold.
No arm was promoted, and the downstream axes named in the decision rule were
not opened.

The low-rank controllers were content-varying and not broadly pinned against
the boundary. Across seeds, the maximum over layers of raw-output RMS was
`0.27-0.40`, and the maximum over layers of phase p95 was `0.32-0.42` rad.
Rare extrema did enter tanh's tail: maximum absolute raw values were `2.4-4.4`,
and maximum observed absolute phase values were `0.95-0.98` rad. Full linear
was more heavily compressed, with maximum per-layer raw RMS around `0.96-1.00`
and phase p95 around `0.74-0.76` rad.

Primary loss results are in
`model-output/position_bias_phase23_dynamic_frequency/DYNAMIC_FREQUENCY_RESULTS.md`;
machine-readable paired results and throughput are in the adjacent
`dynamic_frequency_analysis.json`.

## Post-screen parameterization question

The one-radian `tanh` range was a conservative trust region motivated by the
old unbounded multiplicative-frequency failures. It is not a natural domain of
rotary phase. A smooth alternative such as
`u/sqrt(1+u^2)` retains the range while weakening gradient saturation; a clamp
with a straight-through backward retains a hard forward range while allowing
raw outputs to drift outside it. Neither answers whether the range itself is
appropriate.

The clean unbounded alternative is

```text
delta_phase[b,h,t,i] = (t / 1024) * raw[b,h,t,i]
```

It preserves the exact zero anchor and has bounded position scaling over the
training context, but it permits arbitrary phase winding and grows outside the
reference horizon. Since long-context extrapolation is not the current target,
this is a defensible focused control, but the trained low-rank raw RMS values
show that most existing outputs were already in tanh's near-linear region. Its
expected information value is therefore low.

Before any new training, run the originally required zero, sequence-mean, and
token-shuffle ablations on the saved rank-32 checkpoints. These require GPU
inference but no optimization. They can distinguish reliance on same-token
controller alignment from a quasi-static/sequence-level schedule, although
they do not prove what caused a training-time comparison. Only if those
ablations show a material token-aligned effect should another dynamic mechanism
outrank the main phase-19 confirmation work.

If the free form is eventually run, keep the rank-32 SiLU mapper, Q/K/head
coupling, seeds, and evaluation protocol fixed so the output map is the only
changed variable. Log raw quantiles, phase winding, and ablations; do not reopen
the larger architecture grid on the strength of a small 5k effect.

## Resources

All jobs used `gpu-claim`. Two sequential workers capped this screen at two
concurrent GPUs while allowing the shared scheduler to choose free devices.
That cap described this batch, not a permanent project allocation.
