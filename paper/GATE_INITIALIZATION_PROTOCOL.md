# Phase 58 protocol: scalar-carrier initialization audit

Status: frozen before GPU preflight or outcome inspection on 2026-09-18.

## Question

Phase 57 found that the learned per-layer scalar pre-Q/K carrier gates moved
from `1.0` to roughly `0.021--0.057` by step 100,000. That endpoint does not by
itself imply that training should begin near `0.1`: the Q/K projections and
QK normalization co-adapt, and the final raw scalar is not a standalone
measure of carrier importance. This scout asks whether the scalar method's
early optimization improves when it starts at effective amplitude `0.1`, and
whether any effect comes from initialization, functional step size, or merely
fixing the carrier near that scale.

The audit does not reopen learned phase, frequency, per-frequency amplitude,
dynamic mappers, EMA, or linear-RNN controllers.

## Matched recipe

Every arm uses the Phase-57 architecture and selected optimizer recipe:

- OpenWebText canonical token cache and GPT-2 tokenizer;
- width 768, depth 8, eight heads, GeGLU multiplier 4;
- context 1,024 and sequence batch 32;
- standard RoPE plus the scalar pre-Q/K sinusoidal carrier in every layer;
- paired training and initialization seed 123;
- AdamW at peak LR `1.2e-3`, betas `(0.9, 0.98)`, weight decay `0.01`,
  200-step warmup, linear decay, and global clipping at `1.0`;
- 20,000 optimizer steps (655.36M nominal target tokens);
- 128-block development evaluations every 2,000 steps;
- one matched 1,024-block endpoint evaluation on validation blocks
  `[8192, 9215]`.

All four arms are rerun from scratch. In particular, the `alpha1-direct`
control is not borrowed from Phase 57, because a 20k schedule has a different
linear-decay trajectory from the prefix of a 100k schedule.

## Frozen arms

Write the effective carrier as `alpha * s(p)`.

| Arm | Stored parameter | Effective alpha at initialization | Meaning |
|---|---|---:|---|
| `alpha1-direct` | `alpha`, initialized `1.0` | 1.0 | exact current parameterization |
| `alpha01-direct` | `alpha`, initialized `0.1` | 0.1 | initialization-only change |
| `alpha01-scaled` | `g`, initialized `1.0`, with `alpha=0.1g` | 0.1 | same forward start, about 10x smaller functional Adam step |
| `alpha01-fixed` | no learned scalar | 0.1 | tests whether learning the scalar is needed at this scale |

`gate_output_scale` is a fixed positive coordinate scale, not a learned
constraint or sigmoid/softplus parameterization. For the scaled arm, AdamW
decay of `g` induces the same relative decay of effective `alpha` as direct
optimization, avoiding the decay confound that would arise from simply using
a 0.1x parameter-group LR.

## Decision rule and interpretation

The primary contrast is `alpha01-direct - alpha1-direct`. The scaled and fixed
arms diagnose mechanism. Report paired per-block endpoint differences with IID
and contiguous-block-32 bootstrap 95% intervals, development curves, effective
per-layer gate trajectories, parameter-update and carrier-function-step
telemetry, throughput, memory, and finite-metric checks.

This is a one-seed design scout, not paper-level confirmation. Promote at most
one alternative to a matched 100k run only if it beats `alpha1-direct` by at
least `0.003` NLL and its contiguous-block-32 interval is entirely below zero.
If no arm passes, retain the original initialization and close this axis. A
fixed arm that is statistically tied is scientifically informative but is not
automatically promoted because the mature Phase-42 fixed-at-1.0 arm was worse.

## Preflight and artifacts

Each arm receives a 100-step GPU preflight. Main runs save no periodic recovery
checkpoints and no final weights: at the measured Phase-57 throughput a 20k run
is short enough to rerun. Preserve configs, provenance, metrics, optimization
telemetry, evaluation details, completion markers, and the paired analysis.
All GPU work runs through `gpu-claim` with owner `mlprope` and hard concurrency
two.
