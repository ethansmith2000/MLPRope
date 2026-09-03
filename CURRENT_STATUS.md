# MLPRope current status

_Authoritative as of 2026-09-03. The detailed active decision is
[`CONSOLIDATION_PLAN.md`](CONSOLIDATION_PLAN.md). Older protocols and roadmap
sections are historical records._

## Bottom line

Two attention-local additive mechanisms have earned longer evaluation:

1. **AddRoPE**: a learned additive Fourier carrier on projected Q/K;
2. **pre-Q/K sinusoidal injection**: add a sinusoid only to the inputs of the Q
   and K projections, usually followed by fixed RoPE.

The static, constrained extension is now implemented: separate Q/K amplitudes
and phases can act on the sinusoidal carrier itself. Dynamic RoPE, cumulative
clocks, and EMA are no longer active research priorities.

## Strongest completed evidence

| Result | Protocol | Finding |
| --- | --- | ---: |
| AddRoPE amplitude 1.0 vs fixed RoPE | 30k, 3 paired seeds | `-0.076867` mean loss |
| AddRoPE amplitude 1.0 vs 0.3 | 30k, 3 paired seeds | `-0.014895` mean loss |
| pre-Q/K sinusoid + RoPE vs RoPE | 30k, 3 paired seeds | `-0.065235` mean loss |
| position Q/K gain vs RoPE | 5k, 3 paired seeds | `-0.024215` mean loss |
| pointwise AddRoPE content vs position only | 30k, 3 paired seeds | `-0.010812` mean loss |
| rotary clock vs RoPE | 15k, 1 paired seed | `-0.000360`, unresolved |
| AddRoPE scalar EMA vs pointwise | 15k, 1 paired seed | `-0.010626` step-matched |

The EMA result is not being promoted. Its estimated equal-wall-clock advantage
is only about `-0.0015`, and per-head/per-dimension decays did not improve over
one scalar decay. The EMA scan also adds code and runtime complexity.

At 15k, AddRoPE and the pre-Q/K carrier were strongly sub-additive: combining
them was `+0.004934` worse than AddRoPE alone. Treat this as evidence that they
overlap, not as a mature-model conclusion.

## Critical limitation

The h768/d8 baseline contains approximately 153.4M parameters and trains on
8,192 tokens per step. Thus 5k, 15k, and 30k runs see only 41M, 123M, and 246M
tokens: approximately 0.27, 0.80, and 1.60 tokens per parameter. These runs
mostly measure early optimization.

The three-seed 30k effects are reproducible early-training results, but none of
the current comparisons establishes the mature-model ranking. The 5k
pre-Q/K-without-RoPE result especially does not answer whether RoPE remains
necessary.

## Implemented static adapter

For Fourier pair `i`:

```text
A_i^q = a_i^q R(phi_i^q)
A_i^k = a_i^k R(phi_i^k)
q = W_q(x + A_q z(p))
k = W_k(x + A_k z(p))
```

Test a nested ladder: tied scalar, separate Q/K scalars, separate Q/K pairwise
amplitudes, then separate pairwise amplitudes and phases. Initialize at
`a=1, phi=0` so every new variant exactly matches the current pre-Q/K carrier.

This ladder is now implemented as `qk_preprojection.mode`. Pairwise amplitudes
use an orthonormal zero-sum log-amplitude basis, making their geometric-mean
scale exactly one and leaving the separate global Q/K gains identifiable. At
model width 64 the four modes use 1, 2, 64, and 128 parameters per layer.
All adapter state with angular or scale meaning stays fp32 under bf16/fp16
module conversion.

## Next execution

Run one paired seed of six arms to a common 200k-step horizon: fixed RoPE,
pre-Q/K without RoPE, pre-Q/K plus RoPE, separate Q/K scalar amplitudes,
pairwise amplitudes, and pairwise amplitude+phase. Evaluate the same training
trajectories at 10k/30k/60k/100k/150k/200k and replicate only the best one or
two candidates.

All runs must begin with the same 200k learning-rate horizon. A short run whose
linear schedule has decayed to zero cannot be treated as the prefix of a long
run.

The six resolved long-run configs are frozen under
`sweep_configs/phase33_static_qkpre_200k/`. Rolling completion-marked
checkpoint retention, periodic paired-loss persistence, and per-launch source,
config, package/GPU, and dataset provenance are implemented. All six 50-step
full-size preflights passed; their mean throughput implies roughly 2.41
compute hours per 200k arm before validation, compilation, and checkpoint I/O.

## Repository state

- Phase 29-32 code, configs, compact results, and analyses are preserved in
  provenance commit `c8362c3`.
- The active runtime has been consolidated around fixed RoPE, NoPE, AddRoPE,
  frozen Fourier bases, and the pre-Q/K carrier. Learned/dynamic multiplicative
  RoPE frequencies, rotary phase residuals/clocks, EMA, residual/write
  channels, and the completed position-gain attribution path have been removed.
- All 184 retained sweep JSONs load. The CPU suite passes 109 tests with one
  explicitly CUDA-gated skip, and all nine consolidated bf16 CUDA cases pass
  eager and compiled forward/backward.
- Phase 33 operational preflights passed on all six h768/d8 arms at
  185.6k-190.8k tokens/s and 5.08-5.36 GiB reserved memory. The real
  Accelerate checkpoint save/prune/resume integration also passed.
- The 50 intermediate `step_*` checkpoints were removed after validating all
  18 parent runs. This freed 92.8GB (87GiB). The 18 final weights, completion
  markers, configs, metrics, summaries, final evaluation details, position
  profiles, and compact phase reports remain.
- `/workspace` is not a persistent Vast volume. Copy irreplaceable weights
  off-box before deleting them or destroying/recycling this instance.

See [`CONSOLIDATION_PLAN.md`](CONSOLIDATION_PLAN.md) for the keep/remove map and
[`NEXT_EXPERIMENT_ROADMAP.md`](NEXT_EXPERIMENT_ROADMAP.md) for execution order.
