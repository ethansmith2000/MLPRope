# MLPRope research and repository consolidation

_Decision record, 2026-09-03. This is the active plan. Historical protocols,
configs, and the experiment journal remain evidence, not current priorities._

## Research focus

The active question is now deliberately narrow:

> Can an attention-local sinusoidal signal be made more useful by learning a
> static, constrained transform of the positional signal before the Q and K
> projections, while retaining ordinary RoPE as a stable relative-position
> backbone?

The active mechanism families are:

1. fixed standard RoPE;
2. additive Fourier Q/K position (AddRoPE), with amplitude 1.0 as the active
   initialization;
3. a sinusoidal signal injected only before the Q/K projections, optionally
   followed by fixed RoPE; and
4. static Q/K-specific spectral adapters on that preprojection signal.

Dynamic RoPE frequency, tokenwise rotary phase, cumulative rotary clocks, and
EMA controllers are closed as active directions. Their results remain useful
negative or attribution evidence, but they should not define the active API.

## Implemented preprojection adapter

_Implemented 2026-09-03. The configuration modes are `tied_scalar`,
`split_scalar`, `split_pair_amplitude`, and `split_pair_polar`._

For Fourier pair `i`, let

```text
z_i(p) = [cos(omega_i p), sin(omega_i p)]
A_i^q = a_i^q R(phi_i^q)
A_i^k = a_i^k R(phi_i^k)

q = W_q(x + A_q z(p))
k = W_k(x + A_k z(p))
```

`A_q` and `A_k` act only on the positional carrier. `W_q` and `W_k` still read
content and position jointly afterward; the proposal does not add an
unrestricted independent positional projection.

The exact current pre-Q/K anchor is `a_i^q=a_i^k=1` and
`phi_i^q=phi_i^k=0`. Use nested variants:

1. one gate tied across Q/K (the original anchor);
2. separate scalar Q/K amplitudes;
3. separate Q/K amplitudes per Fourier pair; and
4. separate Q/K amplitude and phase per Fourier pair.

Keep these parameters per layer. Do not add a per-head axis at this injection
site: heads are defined only after the Q/K projections. A per-head carrier is a
different, post-projection mechanism and should not be folded into this test.

The implementation removes global/spectral scale redundancy with an
orthonormal basis for the zero-sum subspace:

```text
a_i = g * exp(delta_i),  sum_i delta_i = 0.
```

Here `g` controls total carrier strength and `P-1` independent coordinates
generate the centered `delta_i` values that redistribute strength across the
spectrum. The initialization is `g=1`, `delta=0`, and `phi=0`.

## Evidence motivating the focus

- Phase 25: compact AddRoPE initialized at amplitude 1.0 beat fixed RoPE by
  `-0.076867` mean held-out loss at 30k steps and beat amplitude 0.3 by
  `-0.014895`, favorable in all three seeds.
- Phase 28: pre-Q/K sinusoid plus RoPE beat fixed RoPE by `-0.065235` mean at
  30k steps, favorable in all three seeds, with essentially equal throughput.
- Phase 30: AddRoPE and the pre-Q/K carrier were strongly sub-additive at 15k;
  their combination was `+0.004934` worse than AddRoPE alone. This is evidence
  of overlap, not a mature-model conclusion.
- Position-only Q/K gains explained a reproducible but smaller portion of the
  benefit (`-0.024215` mean versus RoPE at 5k across three seeds).
- Cumulative rotary clocks were null at 15k (`-0.00036` versus RoPE), with or
  without EMA.
- AddRoPE content EMA improved step-matched 15k loss by `-0.010626` over the
  pointwise controller, but its estimated equal-wall-clock advantage was only
  about `-0.0015`. Per-head and per-dimension EMA coefficients did not improve
  meaningfully over one scalar decay.
- Direct token-dependent RoPE frequency multipliers were unstable, and
  bounded phase interventions were materially null. These are different from
  static positional-carrier amplitude and phase.

## Evidence limitation: the models are undertrained

At batch 8 and sequence length 1024, each step consumes 8,192 tokens. The
h768/d8 baseline has approximately 153.4M parameters:

| Steps | Training tokens | Tokens per parameter |
| ---: | ---: | ---: |
| 5k | 41.0M | 0.27 |
| 15k | 122.9M | 0.80 |
| 30k | 245.8M | 1.60 |
| 100k | 819.2M | 5.34 |
| 200k | 1.638B | 10.68 |

Consequently, the completed experiments establish early-training behavior.
Even the replicated 30k results are not mature-model rankings. In particular,
the existing pre-Q/K-without-RoPE comparison is only a 5k screen and does not
answer whether the sinusoidal input ultimately needs RoPE.

## Long-horizon consolidation experiment

Use one paired seed initially and run every arm under the same long-horizon
schedule:

| Arm | Question |
| --- | --- |
| fixed RoPE | same-box reference |
| current tied pre-Q/K carrier, no RoPE | does the local sinusoid suffice? |
| current tied pre-Q/K carrier + RoPE | established preprojection candidate |
| separate Q/K scalar amplitudes + RoPE | does Q/K tying constrain it? |
| separate Q/K pair amplitudes + RoPE | does spectral selection help? |
| separate Q/K pair amplitude and phase + RoPE | does constrained phase add value? |

Configure all arms for 200k steps from step zero. Evaluate at 10k, 30k, 60k,
100k, 150k, and 200k. The current linear learning-rate schedule depends on
`max_train_steps`; a completed short-horizon run whose learning rate reached
zero is not equivalent to the prefix of a 200k run.

Use successive halving only for clear failures. Compare the full learning
curves and late-curve slopes, not just a single endpoint. Additional seeds are
reserved for the best one or two candidates after this screen.

Before launch:

1. run all arms on the same box and data cache;
2. add rolling checkpoint retention so only the latest resumable checkpoint is
   kept during training;
3. use the canonical dataset path
   `/workspace/data/tokenized/openwebtext_gpt2_bs1024` in generated configs;
4. freeze the source and resolved configs in git; and
5. pass CPU tests, compiled bf16 CUDA smoke, exact-anchor tests, and parameter
   count checks.

## Repository consolidation policy

The repository should distinguish active implementation from historical
evidence. Git history and durable result reports provide the archive; the
runtime does not need to execute every historical configuration.

### Keep active

- fixed RoPE and NoPE controls;
- frozen Fourier basis utilities;
- static AddRoPE and its promoted mapped-carrier reference;
- the current pre-Q/K sinusoid and the new constrained Q/K spectral adapter;
- fused SDPA training, evaluation, diagnostics, and paired initialization;
- compact result reports and analysis JSON.

Retain the pointwise content-conditioned AddRoPE implementation initially as a
frozen reference because it produced a real, replicated 30k increment. It is
not part of the next sweep. Reconsider it only after the static consolidation.

### Remove from the active runtime after a provenance commit

- learned static and content-dependent multiplicative RoPE frequencies;
- rotary phase-residual special cases;
- cumulative rotary clocks and their pointwise/convolution/EMA controllers;
- EMA conditioning for AddRoPE;
- residual-stream positional channels and attention-output write channels,
  which were already retired experimentally;
- position-only Q/K gain machinery after preserving its attribution result;
- launch, preparation, and smoke code used only by those removed mechanisms.

For removed top-level config blocks, prefer a small compatibility validator
that accepts only the historical disabled/fixed form and raises a clear error
for an enabled removed mechanism. Do not retain dormant model machinery merely
to execute old sweep JSONs.

### Consider in a second simplification pass

- extract the surviving AddRoPE implementation from the generic
  `position/channels.py` framework;
- retain only AddRoPE geometries and mapper forms supported by durable results;
- remove v1 upgrade paths and historical presets from the active loader;
- move superseded protocols out of the repository root or rely on git history;
- reduce top-level phase-specific prepare/launch/analyze scripts after their
  compact result reports are verified.

This second pass should follow, not precede, the first long-horizon config
freeze. It is higher risk because the generic AddRoPE code also contains the
best historical mechanisms.

## Storage cleanup

_Completed 2026-09-03: all 50 intermediate `step_*` directories were removed
after the checks below, freeing 92.8GB (87GiB). All 18 final model weights and
compact evidence artifacts remain._

Before cleanup, the checkout was about 97GB. Approximately 87GiB was in 50
intermediate checkpoint directories; 18 final model files used another
10.37GB, while configs, metrics, position profiles, summaries, and per-example
evidence used only a few megabytes.

This workspace is not a persistent Vast volume. Before deleting model states:

1. commit the Phase 29-32 code/config/result state;
2. verify every retained run has `COMPLETED`, a final evaluation, resolved
   config, metrics, summary, and compact result report;
3. copy any irreplaceable final weights off-box; then
4. delete intermediate checkpoints first. Final weights are a separate
   retention decision.

No model artifacts should be deleted merely as a side effect of code cleanup.
