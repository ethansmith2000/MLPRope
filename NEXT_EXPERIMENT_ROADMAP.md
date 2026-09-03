# MLPRope next-experiment roadmap

_Revised 2026-09-03. This roadmap implements the decision in
[`CONSOLIDATION_PLAN.md`](CONSOLIDATION_PLAN.md)._

## Stage 0 — preserve and simplify

Completed:

- Phase 29-32 implementation, configs, compact results, and analyses were
  recorded in provenance commit `c8362c3`.
- Closed dynamic-RoPE, rotary-clock, EMA, residual/write, and position-gain
  paths were removed from the runtime and dedicated scripts/configs.
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

## Stage 1 — static pre-Q/K spectral adapter (completed)

Extend `qk_preprojection` without introducing a separate mechanism family.
Required modes are:

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

## Stage 4 — globally shared frequency screen (active)

New evidence supports a materially different static-frequency hypothesis from
the removed local tables: one learned bank shared across all layers, heads, and
Q/K branches. The implementation and five-arm protocol are specified in
[`SHARED_FREQUENCY_PLAN.md`](SHARED_FREQUENCY_PLAN.md).

The screen contains fixed and learned-log pure-RoPE arms, plus fixed,
learned-log, and horizon-normalized versions of the tied pre-Q/K carrier. All
use one paired seed and a common 200k horizon. No content conditioning,
per-layer/head frequency axis, or learned-RoPE/learned-carrier combination is
included.

The five compiled bf16 50-step preflights pass at 5,076--5,220 MiB reserved
memory. All learned spectra were active, finite, positive, and ordered. The
long runs are cleared to launch through `gpu-claim`; compact preflight evidence
is in `results/phase34_shared_frequency_preflight/`.

## Stage 5 — promote finalists

Promote no more than two candidates. Add seeds 456 and 789 under the identical
long-horizon protocol. A durable claim requires:

- favorable signs across all paired seeds;
- a practically material late-horizon mean improvement;
- no clear convergence of the advantage toward zero;
- sane learned spectra without frequency-pair collapse;
- acceptable wall-clock and memory cost.

Only after this gate consider model-width transfer or longer-context testing.

## Explicitly deferred

- content-dependent RoPE frequencies or frequency multipliers;
- backward-only surrogate gradients for dynamic frequencies;
- cumulative clocks or arbitrary tokenwise warps;
- EMA/linear-RNN conditioning;
- per-head post-projection sinusoidal branches;
- full or low-rank mixing between Fourier pairs;
- combinations of AddRoPE and pre-Q/K injection;
- broad mapper or coupling sweeps.

Also defer smooth/order-preserving shared spectra and combined learned
RoPE+carrier arms until the free globally shared banks clear Phase 34.

These are recoverable from history if new evidence creates a specific reason to
reopen them.
