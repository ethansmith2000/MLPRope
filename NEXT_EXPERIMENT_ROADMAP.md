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

Remaining storage work:

1. Verify retained result artifacts before any storage deletion.
2. Remove intermediate `step_*` checkpoints after confirming final weights are
   untouched and every run retains compact scientific evidence.

Do not combine the provenance commit and removal commit. A bisectable boundary
is more valuable than preserving dormant compatibility in the active runtime.

## Stage 1 — implement the static pre-Q/K spectral adapter

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

## Stage 2 — add long-run operational safeguards

Before launching long runs:

1. Add rolling checkpoint retention, keeping only the latest resumable state
   plus explicitly requested milestones.
2. Point generated configs directly at
   `/workspace/data/tokenized/openwebtext_gpt2_bs1024`.
3. Record source commit, resolved config, package/GPU versions, dataset
   manifest, and dirty-tree status in every run.
4. Preflight all arms for 20-50 full-size steps through `gpu-claim`.
5. Measure stable throughput and peak memory after compilation warmup.

## Stage 3 — one-seed 200k consolidation screen

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

## Stage 4 — promote finalists

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

These are recoverable from history if new evidence creates a specific reason to
reopen them.
