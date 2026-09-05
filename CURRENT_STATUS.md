# MLPRope current status

_Authoritative as of 2026-09-05. The active architectural contract is
[`SINUSOID_INTERVENTION_POLICY.md`](SINUSOID_INTERVENTION_POLICY.md). Older
protocols and roadmap sections are historical records._

## Bottom line

Two attention-local sinusoidal mechanisms have earned longer evaluation:

1. **AddRoPE**: a learned additive Fourier carrier on projected Q/K;
2. **pre-Q/K sinusoidal injection**: add a sinusoid only to the inputs of the Q
   and K projections, usually followed by fixed RoPE.

The 200k static-adapter and shared-frequency experiments are complete. The tied
pre-Q/K carrier plus fixed RoPE remains strongly favorable. Separate Q/K
amplitude, pairwise amplitude, pairwise phase, and learned carrier frequency
did not improve it. Dynamic RoPE, learned RoPE, cumulative clocks, and EMA are
closed in the active runtime.

The repository now treats the backbone and carrier as orthogonal axes: every
intervention uses either immutable standard RoPE or NoPE, while learned
amplitude, phase, or frequency acts only on the sinusoidal carrier. AddRoPE no
longer implicitly replaces RoPE.

Phase 35 found that rank-4 smooth spectral amplitude helped the carrier
materially under NoPE (`-0.010644`) but remained `0.037168` worse than its
matched RoPE model. Under RoPE the amplitude gain was a smaller `-0.002604`,
below the predeclared practical gate. Learned phase and separate Q/K transforms
did not earn promotion.

Phase 36 revisited amplitude and carrier frequency with direct, nonsaturating,
endpoint-conditioned coordinates under RoPE. Direct signed rank-4 amplitude
improved the scalar carrier by `-0.003661`, paired-example 95% CI
`[-0.004247,-0.003074]`, clearing the `0.003` screen threshold. A conservative
rank-4 frequency deformation added only `-0.000457` beyond amplitude; global
frequency was null, and the faster rank-4 frequency arm violated spectral
ordering.

Phase 37 completed the required 200k confirmation. Direct amplitude was
`+0.000111` versus scalar, CI `[-0.001041,+0.001263]`; exponential amplitude
was `-0.000363`, CI `[-0.001559,+0.000833]`. Direct versus exponential was also
null. The short-horizon amplitude result did not survive the mature primary
holdout, so no carrier-shape refinement remains a promotion candidate.

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
| tied pre-Q/K + RoPE vs fixed RoPE | 200k, 1 paired seed | `-0.062831` |
| tied pre-Q/K + RoPE vs no RoPE | 200k, 1 paired seed | `-0.030773` |
| split/pair amplitude/pair phase ladder | 200k, 1 paired seed | all within about `0.001` |
| shared log-frequency carrier vs fixed carrier | 200k, 1 paired seed | `+0.000861`, null |
| horizon-frequency carrier vs fixed carrier | 200k, 1 paired seed | `+0.001341`, slightly worse |
| smooth amplitude vs scalar carrier, NoPE | 20k, 1 paired seed | `-0.010644`, gate pass |
| smooth amplitude vs scalar carrier, RoPE | 20k, 1 paired seed | `-0.002604`, below gate |
| direct smooth amplitude vs scalar carrier, RoPE | 20k, 1 paired seed | `-0.003661`, gate pass |
| direct amplitude + hybrid frequency vs direct amplitude | 20k, 1 paired seed | `-0.000457`, below gate |
| global/hybrid direct frequency vs scalar carrier | 20k, 1 paired seed | null/subthreshold; fast hybrid broke ordering |
| direct smooth amplitude vs scalar carrier | 200k, 1 paired seed | `+0.000111`, null |
| exponential smooth amplitude vs scalar carrier | 200k, 1 paired seed | `-0.000363`, null |

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

The three-seed 30k effects are reproducible early-training results, not
mature-model rankings. Phase 33 extended the pre-Q/K comparison to 200k
(1.638B tokens, 10.68 tokens per parameter) and established that fixed RoPE
remains materially useful on top of the carrier for seed 123. That is strong
long-horizon evidence for the mechanism comparison, but it is not yet a
multi-seed or compute-optimal scaling result.

## Active pre-Q/K adapter

The runtime exposes `tied_scalar`, the historical exponential
`tied_smooth_amplitude`, and the Phase-36
`tied_smooth_direct_amplitude`. All add one shared sinusoidal carrier before
the Q and K projections; `W_q` and `W_k` still learn separate reads. The direct
mode uses `a_i = g(1 + (Bc)_i)` with rank-4, zero-mean, unit-RMS DCT coordinates.
It starts exactly at the scalar carrier, is nonsaturating, permits signed
factors, and leaves the global strength identifiable through `g`. At 20k its
factors remained positive but spanned `0.057--1.698`, so longer runs must retain
the signed-factor diagnostics.

The Phase 33 split-scalar/free-pair ladder and Phase 35 phase/QK-split modes are
preserved in configs and result reports but removed from active execution.
Enabled historical modes fail with a direct migration message; disabled
archival blocks canonicalize to `tied_scalar`. This removes separate Q/K gates,
phase tensors, and full per-pair tables from the model while retaining the only
shape variants with positive evidence.

Phase 36 also added direct global and rank-4 hybrid carrier-frequency
coordinates with bounded endpoint-phase Jacobians, independent LR multipliers,
no weight decay, and a dedicated gradient group. They remain executable for
reproducibility but did not earn scientific promotion. Phase 37 likewise
leaves both smooth-amplitude maps executable for reproduction, but neither is
an active scientific finalist.

## Phase 34 conclusion

All five historical Phase-34 arms reached 200k. The fixed carrier finished at
`3.324979`; learned-log finished at `3.325839`, and horizon-normalized finished
at `3.326320`. The normalized form successfully kept the endpoint phase
Jacobian at one, demonstrating that it fixes the absolute-position
conditioning problem, but it did not improve modeling loss. No carrier
frequency arm clears the replication gate.

The learned-RoPE calibration arm finished `-0.001431` below fixed RoPE, but
failed the `0.002` promotion threshold and its late advantage was shrinking.
It is preserved as historical evidence, not retained as active machinery. See
[`results/phase34_shared_frequency_200k/PHASE34_RESULTS.md`](results/phase34_shared_frequency_200k/PHASE34_RESULTS.md).

## Next execution

The default anchor remains the scalar pre-Q/K carrier with standard fixed RoPE.
No GPU experiment is automatically next. Phase 37 closes the smooth carrier-
shape branch, and the default remains the scalar pre-Q/K carrier with fixed
RoPE. Frequency, amplitude shape, phase, and Q/K-untying refinements should be
shelved. The appropriate next action is repository consolidation or a distinct
hypothesis about the broader pre-Q/K/AddRoPE mechanisms. See the
[`Phase-37 result`](results/phase37_direct_amplitude_200k/PHASE37_RESULTS.md).

## Repository state

- Phase 29-32 code, configs, compact results, and analyses are preserved in
  provenance commit `c8362c3`.
- The active runtime is consolidated around fixed RoPE or NoPE, AddRoPE,
  frozen Fourier bases, and the pre-Q/K carrier. Learned frequency remains
  available only inside the carrier as a diagnostic/research coordinate;
  model-wide learned-RoPE machinery is removed.
- The old content-conditioned `adaptive_gain` escape hatch is removed because
  it scaled complete Q/K tensors rather than the sinusoidal carrier.
- Sparse intervention diagnostics distinguish raw/clipped gradients, Adam
  moment behavior, actual parameter updates, and functional carrier movement.
- The consolidated CPU suite passes 121 tests with one explicitly CUDA-gated
  skip. A 12-step compiled-bf16 optimizer-monitor smoke passed on an RTX 5090;
  all five scheduled traces were finite. All ten retained carrier/backbone
  cases also pass eager and compiled bf16 forward/backward. Compact evidence is in
  `results/carrier_optimization_monitor_smoke/`.
- Phase 33 operational preflights passed on all six h768/d8 arms at
  185.6k-190.8k tokens/s and 5.08-5.36 GiB reserved memory. The real
  Accelerate checkpoint save/prune/resume integration also passed.
- All eight Phase-35 20k arms completed from clean source commit `cee3214`.
  Throughput was 178.9k--184.5k target tokens/s and peak reserved memory was
  5.22--5.36 GiB. Paired final losses, every development point, compact DCT
  profiles, and optimizer-health summaries are preserved in `results/`.
- The 50 intermediate `step_*` checkpoints were removed after validating all
  18 parent runs. This freed 92.8GB (87GiB). The 18 final weights, completion
  markers, configs, metrics, summaries, final evaluation details, position
  profiles, and compact phase reports remain.
- All seven Phase-36 20k arms completed from clean source commit `d7991d9`.
  Throughput was 179.5k--184.5k target tokens/s and peak memory was
  5.08--5.22 GiB. The runs used direct nonsaturating amplitude/frequency
  coordinates, explicit positional/frequency LR groups, and full QKNorm
  mixture diagnostics. The CPU suite passes 125 tests with one CUDA-only skip;
  all new mechanisms also passed eager and compiled bf16 GPU checks.
- All three Phase-37 arms completed 200k steps from clean source commit
  `781a8ae`. Each resumed once from its complete step-70k checkpoint with model,
  optimizer, scheduler, sampler, and RNG state restored. The primary holdout,
  full development trajectories, learned-factor/QKNorm diagnostics, and
  optimizer traces are preserved in `results/` and `model-output/`.
- `/workspace` is not a persistent Vast volume. Copy irreplaceable weights
  off-box before deleting them or destroying/recycling this instance.

See [`CONSOLIDATION_PLAN.md`](CONSOLIDATION_PLAN.md) for the keep/remove map and
[`NEXT_EXPERIMENT_ROADMAP.md`](NEXT_EXPERIMENT_ROADMAP.md) for execution order.
