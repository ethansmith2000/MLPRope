# MLPRope next-experiment roadmap

_Revised 2026-09-05. This roadmap implements the decision in
[`SINUSOID_INTERVENTION_POLICY.md`](SINUSOID_INTERVENTION_POLICY.md)._

## Stage 0 — preserve and simplify

Completed:

- Phase 29-32 implementation, configs, compact results, and analyses were
  recorded in provenance commit `c8362c3`.
- Closed dynamic-RoPE, rotary-clock, EMA, residual/write, and position-gain
  paths were removed from the runtime and dedicated scripts/configs.
- After the Phase 33/35 provenance commits, the pre-Q/K split-Q/K, phase, and
  free per-pair modes were also removed. The active adapter is now
  `tied_scalar` or `tied_smooth_amplitude`.
- Disabled/fixed compatibility validators remain for understandable failures
  from archived resolved configs.
- The retained config corpus, CPU suite, and eager/compiled bf16 CUDA smoke all
  pass.

- All 18 affected parent runs were checked for completion markers, configs,
  metrics, summaries, final evaluation details, and final weights. The 50
  intermediate `step_*` directories were then removed; final weights and
  compact evidence were left untouched.

Do not combine the provenance commit and removal commit. A bisectable boundary
is more valuable than preserving dormant compatibility in the active runtime.

## Stage 1 — historical static pre-Q/K spectral adapter (completed)

Phase 33 extended `qk_preprojection` without introducing a separate mechanism
family. The evaluated modes were:

| Mode | Q/K coupling | Spectral granularity | Learned geometry |
| --- | --- | --- | --- |
| tied scalar | tied | all pairs | amplitude |
| split scalar | separate | all pairs | amplitude |
| split pair amplitude | separate | each Fourier pair | amplitude |
| split pair polar | separate | each Fourier pair | amplitude + phase |

Keep parameters per layer. A per-head mode is out of scope because this carrier
is transformed before heads are produced by `W_q/W_k`.

Required tests:

- exact equivalence to the current carrier at `a=1, phi=0`;
- distinct Q/K outputs and gradients after perturbation;
- pair rotation agrees with direct trigonometric phase addition;
- global/spectral scale factorization is non-redundant;
- state-dict and resolved-config round trip;
- fp32 carrier construction under bf16 training;
- eager and compiled forward/backward;
- unchanged V and no residual-stream write;
- parameter-count assertions for every granularity.

All four modes and the listed invariants now pass the CPU suite and claimed-GPU
eager/compiled bf16 smoke. The implementation uses `P-1` orthonormal zero-sum
coordinates for `P` log-amplitude deltas, so global and spectral amplitude are
identifiable rather than merely mean-centered in the forward pass.

These implementation details describe the historical test surface. After the
null long-horizon results and the Phase 35 smooth follow-up, only the tied scalar
and tied smooth-amplitude modes remain executable.

## Stage 2 — long-run operational safeguards (completed)

Before launching long runs:

1. Add rolling checkpoint retention, keeping only the latest resumable state
   plus explicitly requested milestones.
2. Point generated configs directly at
   `/workspace/data/tokenized/openwebtext_gpt2_bs1024`.
3. Record source commit, resolved config, package/GPU versions, dataset
   manifest, and dirty-tree status in every run.
4. Preflight all arms for 20-50 full-size steps through `gpu-claim`.
5. Measure stable throughput and peak memory after compilation warmup.

Completed at source commit `6aa859d`. All six separate 50-step h768/d8
preflights passed through `gpu-claim` with clean provenance. Throughput ranged
from 190.8k tokens/s for fixed RoPE to 185.6k for the pair-polar adapter;
reserved memory ranged from 5.08 to 5.36 GiB. A real Accelerate save/prune/
resume smoke retained only the newest marked checkpoint and restored model,
optimizer, scheduler, sampler, and RNG state. Compact evidence is in
`results/phase33_static_qkpre_preflight/`.

## Stage 3 — one-seed 200k consolidation screen (completed)

The six resolved long-run configs are frozen under
`sweep_configs/phase33_static_qkpre_200k/`. At measured mean throughput, one
arm requires about 2.41 compute hours before validation, compilation, and
checkpoint overhead.

All runs use h768/d8, eight heads, context 1024, microbatch 8, paired seed 123,
the same data order, and a learning-rate scheduler configured for 200k from
step zero.

| Arm | Direct comparison |
| --- | --- |
| fixed RoPE | common reference |
| tied pre-Q/K, no RoPE | versus fixed RoPE and tied pre-Q/K + RoPE |
| tied pre-Q/K + RoPE | versus fixed RoPE |
| split Q/K scalar + RoPE | versus tied pre-Q/K + RoPE |
| split Q/K pair amplitude + RoPE | versus split scalar |
| split Q/K pair amplitude+phase + RoPE | versus pair amplitude |

Evaluate at 10k, 30k, 60k, 100k, 150k, and 200k on fixed development and
disjoint holdout slices. Save paired per-example losses. Report:

- endpoint loss and paired confidence intervals;
- the candidate-reference curve at every milestone;
- late-window loss slope and whether the gap is closing;
- tokens/second, wall-clock time, peak memory, and parameter count;
- learned global gains, amplitude spectra, phase spectra, and Q/K differences
  by layer.

An arm may stop early only for divergence, clear sustained harm, or a mechanism
health failure. Do not prune a close arm from a 10k endpoint.

Final result: tied pre-Q/K plus fixed RoPE beat fixed RoPE by `-0.062831`, and
RoPE contributed `-0.030773` relative to the tied no-RoPE arm. Every added
static Q/K/pair amplitude or phase degree of freedom was within about `0.001`
of its parent. See the Phase-33 report linked from `CURRENT_STATUS.md`.

## Stage 4 — globally shared frequency screen (completed)

New evidence supports a materially different static-frequency hypothesis from
the removed local tables: one learned bank shared across all layers, heads, and
Q/K branches. The implementation and five-arm protocol are specified in
[`SHARED_FREQUENCY_PLAN.md`](SHARED_FREQUENCY_PLAN.md).

The five historical arms used one paired seed and a common 200k horizon. The
carrier log-frequency arm was `+0.000861` worse than its fixed parent with a
paired interval crossing zero. The horizon-normalized arm was `+0.001341`
worse, with its interval excluding zero. Neither earned replication.

The learned-RoPE calibration was `-0.001431`, below the `0.002` gate, and its
late advantage was collapsing. It is historical evidence only: current
interventions do not modify RoPE. Full results are in
[`results/phase34_shared_frequency_200k/PHASE34_RESULTS.md`](results/phase34_shared_frequency_200k/PHASE34_RESULTS.md).

## Stage 5 — carrier-only consolidation and optimization audit (completed)

Enforce the fixed-RoPE/NoPE backbone boundary and remove learned-RoPE launch
machinery. Every surviving learned positional change acts on the sinusoidal
carrier. AddRoPE and pre-Q/K carriers can be crossed explicitly with either
backbone.

For each learned intervention, persist sparse diagnostics for:

- raw and clipped parameter gradients;
- Adam first-moment alignment and second-moment concentration;
- realized parameter updates and their current-gradient alignment;
- the resulting functional carrier movement and, where applicable, endpoint
  phase Jacobian and movement.

This audit distinguishes poor conditioning, outlier-inflated Adam moments, and
momentum interference from a well-optimized but scientifically null method.

## Stage 6 — promote only a new clear finalist (Phase 35 completed)

Start a new idea with one seed and a 10k--20k horizon. Add seeds 456 and 789
only after a materially favorable, non-collapsing result under healthy
optimization. A durable claim requires:

- favorable signs across all paired seeds;
- a practically material late-horizon mean improvement;
- no clear convergence of the advantage toward zero;
- sane learned spectra without frequency-pair collapse;
- acceptable wall-clock and memory cost.

Only after this gate consider model-width transfer or longer-context testing.

Phase 35 tested rank-4 smooth pre-Q/K carrier amplitude/phase under fixed RoPE
and NoPE. NoPE amplitude passed its direct-parent gate at `-0.010644`, but the
matched RoPE arm remained better by `0.037168`; this is not an absolute model
finalist. RoPE amplitude was `-0.002604`, just below the practical gate. Phase
and Q/K untying were null or subthreshold under healthy optimization.

Disposition: do not automatically add seeds or a 200k run. Retain NoPE smooth
amplitude only as a conditional lead for a specifically RoPE-free objective.
Otherwise Phase 35 closes the current carrier-shape ladder. Protocol and result
are in [`SMOOTH_CARRIER_PLAN.md`](SMOOTH_CARRIER_PLAN.md) and the
[`Phase-35 report`](results/phase35_smooth_carrier_20k/PHASE35_RESULTS.md).

## Stage 7 — direct-coordinate screen (completed)

Phase 36 tested whether the Phase-35 amplitude result or the Phase-34 frequency
null was an artifact of exponential/log coordinates. The new parameters used
no positivity or saturation transform:

```text
amplitude: a_i = g * (1 + (B c)_i)
global frequency: omega_i' = omega_i + omega_i * tau/(L*omega_max) * u
hybrid frequency: omega_i' = omega_i + min(omega_i, tau/L) * (B c)_i
```

All modes started at the live scalar carrier (`g=1`, zero deformation), used
fixed RoPE, method-aware QKNorm, fp32 coordinates, no positional weight decay,
and separately calibrated LR groups. Seven seed-123 arms ran for 20k steps.

Direct amplitude passed at `-0.003661`, CI `[-0.004247,-0.003074]`. Global
frequency was null. Hybrid frequency at LR1 reached `-0.000941`; LR4 reached
`-0.002258` but broke `6.27%` of adjacent spectral ordering. Adding LR1 hybrid
frequency to direct amplitude improved it by only `-0.000457`. Optimizer traces
were finite and frequency gradients were never clipped, so frequency's result
is a scientific null/subthreshold result rather than obvious optimizer failure.

Disposition:

- promote only direct rank-4 amplitude to longer confirmation;
- keep the scalar carrier as its paired parent;
- do not promote frequency or the amplitude+frequency combination as a distinct
  mechanism;
- retain factor-sign, spectrum, QKNorm-mixture, and optimizer diagnostics;
- add seeds only after the longer run shows a material, non-collapsing gap.

See the [Phase-36 report](results/phase36_direct_carrier_20k/PHASE36_RESULTS.md).

## Stage 8 — long-horizon amplitude confirmation (queued)

Phase 36 promotes one narrow comparison to 200k: scalar pre-Q/K versus
exponential rank-4 amplitude versus direct rank-4 amplitude, all with fixed
RoPE. The exponential arm makes the parameterization question paired rather
than relying on the exploratory Phase-35/36 comparison. Frequency is excluded.

The primary gate is direct amplitude versus scalar: at least `-0.002` nats on
the disjoint final holdout, paired interval below zero, non-collapsing late
trajectory, and healthy signed-factor/QKNorm/optimizer diagnostics. Only a pass
earns additional seeds. The protocol and frozen matrix are in
[`DIRECT_AMPLITUDE_CONFIRMATION_PLAN.md`](DIRECT_AMPLITUDE_CONFIRMATION_PLAN.md).

## Explicitly deferred after Phase 36

- any learned or content-dependent RoPE frequencies or frequency multipliers;
- further carrier-frequency maps without a new structural hypothesis;
- backward-only surrogate gradients for dynamic frequencies;
- cumulative clocks or arbitrary tokenwise warps;
- EMA/linear-RNN conditioning;
- per-head post-projection sinusoidal branches;
- full or low-rank mixing between Fourier pairs;
- additional static carrier-phase or Q/K-untying ladders;
- combinations of AddRoPE and pre-Q/K injection, absent a specific factorial
  hypothesis;
- broad mapper or coupling sweeps.

Learned RoPE+carrier arms remain outside the active architectural boundary.
These paths are recoverable from history if new evidence creates a specific
reason to reopen them.
