We have a useful menu now. I’d separate enabling code changes from experiment packs and avoid a full Cartesian sweep.

1. Foundation refactor — recommended first
Medium code change; enables everything else cleanly.

Replace overloaded Q/K config with independent axes:

qk:
  application: additive | rotary
  geometry: free_direct | free_residual | phase | amplitude_phase | projected_phase
  input: fourier | fourier_scalar | learned_temp | learned_freq
  mapper: identity | affine | linear | low_rank | bottleneck_mlp | mlp
  qk_coupling: shared | separate_readout | separate
  head_coupling: shared_head | per_head_independent | per_head_joint
  basis_dim: 96
Implementation pieces:

A standalone Fourier basis module, adapted from FourierEmbedder.
Consistent cos/sin pair layout.
Output heads specialized by geometry:
free additive → D;
phase → D/2;
amplitude+phase → D;
projected phase → D then pair-normalize.
Shared trunk plus distinct Q/K readouts.
Backward-compatible expansion of old configs.
Diagnostics for amplitude, phase, output norm, learned frequencies, and head specialization.
2. Faithful AddRoPE pack — highest-priority experiment
Our Phase-1c runs did not test canonical AddRoPE. Implement:

q' = q + a_q · cis(ωp + δ_q)
k' = k + a_k · cis(ωp + δ_k)
Compact screening pack:

learned amplitude only, δ=0;
learned phase only, a=1;
amplitude + phase;
amplitude + phase, per_head_independent;
amplitude + phase, per_head_joint;
free linear additive map as the existing comparison.
Use separate Q/K readouts and reproduce the author’s parameter initialization if available. Amplitude initialization matters because the fixed unit addend was large.

3. Additive geometry pack
Question: does additive position benefit from freedom or trigonometric structure?

Hold basis, mapper budget, Q/K coupling, and head coupling fixed:

fixed sinusoidal;
free direct f(z);
free residual z + f(z);
phase-only cis(ωp+δ);
amplitude+phase;
arbitrary D output followed by pair normalization.
Start with linear/low-rank mappers. Only run MLP versions if a geometry earns them.

4. Strict rotary pack
Question: what kind of learning helps without abandoning valid rotations?

RoPE baseline;
learned shared temperature;
learned per-frequency residuals: ω = ω_base·exp(Δlogω);
position-dependent D/2 phase residual;
phase from mixed Fourier/scalar input;
projected-phase output;
optional scaled rotary, clearly labeled non-orthogonal.
Separate exact-relative variants from relaxed ones:

learned shared affine frequencies preserve offset structure;
nonlinear δ(p) generally introduces absolute-position dependence.
5. Q/K and head-coupling pack
Run only on the best additive and rotary geometry.

Q/K:

shared;
shared trunk, separate readouts;
fully separate.
Heads:

shared_head;
per_head_independent;
per_head_joint.
Do this one axis at a time around the preferred default rather than all nine combinations.

6. Fourier input and efficiency pack
After choosing a geometry:

basis widths: 8, 16, 32, and native 48 frequency pairs;
frozen frequencies;
learned temperature;
residual learned frequencies;
optional mixed input:
[Fourier(p), p/max_train_length, log1p(p)]
This determines whether a compact Fourier basis plus mapper is sufficient.

7. Residual-stream positional sector
A separate hypothesis from Q/K locality:

no position control;
standard sinusoidal PE;
learned absolute PE;
Linear(Fourier_K(p));
MLP(Fourier_K(p));
zero-gated per-layer functional reinjection.
Initially test these as the sole positional mechanism. Later combine the winner with RoPE.

8. Attention-output positional writes
Higher-effort, but conceptually strong:

content_i = Σ_j A_ij V_j
pos_i     = Σ_j A_ij g(position_j)
y_i       = W_content(content_i) + gate_i·W_pos(pos_i)
Menu:

query-position reinjection;
attended key-position summary;
attended relative-offset summary.
Start with key-position summary and a zero-initialized gate. It can potentially share attention weights by augmenting the value stream.

9. Content-aware Q/K
After position-only geometry is settled:

f(q) / f(k): content-modulates a fixed position prior;
f(q,z(p)) / f(k,z(p)): explicit content-position interaction;
additive D output;
rotary D/2 phase output.
Local Q/K conditioning remains KV-cache compatible. Pairwise Q–K conditioning does not and should be a later mechanism.

10. Relative-logit sector
Keep independent:

existing linear and low-rank offset functions;
Inkling table;
Inkling CosNet;
query-content-conditioned offset profiles;
causal visible-key/count conditioning where appropriate.
Once each sector has a winner, test unions:

best rotary + linear logit;
best AddRoPE + linear logit;
best Q/K geometry + Inkling;
residual-stream winner + attention-local winner.
Experimental hygiene
Several observed gaps are only ~0.01 loss, so single-seed rankings are not definitive.

Recommended workflow:

one-seed screens;
three-seed confirmation for baseline and top two;
parameter-matched FFN control;
length extrapolation;
output-norm, phase, amplitude, and head-specialization diagnostics.
Recommended immediate order
Foundation refactor.
Faithful AddRoPE six-run pack.
Strict rotary six-run pack.
Confirm winners with three seeds.
Then choose between residual-stream functional PE and content-aware Q/K.
That gives two queue-friendly batches of at most six jobs without exploding into a Cartesian sweep.

## 2026-07-27 evidence-backed search status

The original menu above remains the historical design map. The active search
space is now narrower:

- **Active:** scalar-augmented canonical amplitude+phase AddRoPE with a static
  linear relative-logit channel; radius-0.3 pair-normalized additive geometry;
  method-aware Q/K RMS placement; compact Fourier and low-rank linear-logit
  efficiency controls. Dedicated-content phase remains stable only on a fixed
  carrier and is not a promotion candidate.
- **Controls only:** standard RoPE, linear-logit-only, free additive Q/K, and
  parameter-matched wider-FFN models.
- **Retired from active sweeps:** residual-stream positional encodings,
  attention-output writes, Inkling and pairwise content-aware logit variants,
  direct Q/K local-residual and content-gate conditioning, learned rotary
  phase/scale alternatives, learned Fourier temperatures/frequencies, and
  broad mapper/QK/head-coupling grids. These implementations remain available
  for reproducibility.
- **Historically invalid or mislabeled:** the Phase-1 `low_rank` logit was a
  nonlinear bottleneck rather than the later corrected factorized-linear
  mechanism; Phase-1c `add_rope` was not canonical amplitude+phase AddRoPE;
  pre-fix conditioning collapse and zero-stride pairwise failures are
  implementation/parameterization failures rather than clean hypothesis
  tests.

The current promotion criterion is deliberately conservative. A single-seed
5k gap below `0.01` is treated as tied. A mechanism that is consistently worse
by at least `0.02`, remains effectively null, or opens a large positional
branch without improving loss is removed from future sweeps. NaNs and runaway
Q/K ratios are classified as broken parameterizations, not ordinary negative
results.

Before increasing model size or training horizon, only three orthogonal
questions remain:

1. reconcile canonical versus pair-normalized geometry with scalars and
   legacy versus method-aware Q/K normalization;
2. test whether dedicated-content additive phase transfers onto that winning
   full stack, and retire the union if learned carrier amplitude runs away;
3. test compact Fourier bases and corrected low-rank linear logits against
   parameter-matched FFN controls.

The h768/d8 10k scale gate promoted exactly two stacks:

- full linear logit for training-length quality (`4.0345` at 1024);
- rank-32 linear logit for extrapolation (`4.0556` at 4096 versus `4.1516`
  for the full channel).

Canonical amplitude+phase with method-aware RMS is fixed for both. Basis 32,
branch-only RMS, pair-normalized geometry on the scalar stack, and the
full-stack content-phase union are pruned. Basis 16 remains an optional
compression setting but not a quality promotion candidate.

The h1024/d12, batch-8, 10k scale run confirmed both endpoints. Full and
rank-32 logits reached `3.9523` and `3.9545` at 1024; at 4096 the rank-32
variant led `3.9601` to `3.9693`. The matched-FFN control was `4.0094` at
1024, so additional generic capacity does not explain the positional gain.
The breadth search is closed unless a future scale or extrapolation failure
provides a specific reason to reopen a retired axis.

## 2026-07-27 50k scale freeze

The h1024/d12, effective-batch-32, 50k gate supersedes the provisional 10k
ordering:

- **Additive quality/default:** basis-16 canonical scalar AddRoPE +
  method-aware Q/K RMS + full linear relative logit (`2.9835`, `3.0456`,
  `3.1142` at 1024/2048/4096).
- **Extrapolation positional default:** RoPE + full linear relative logit
  (`3.0132`, `3.0551`, `3.0658`). The FFN-4160 control is slightly better but
  is retained as a capacity control.
- **Throughput/control default:** standard RoPE (`3.0337`, `3.0864`,
  `3.1489`) at about 90.3k tokens/s versus 47–48k for FlexAttention/logit
  variants.
- **Pruned after reversal:** corrected rank-32 relative logit. Its 4096 loss
  worsened to `3.3595`, so the apparent 10k extrapolation advantage did not
  survive the longer horizon.
- **Dominated:** the full Q/K Fourier basis. Basis 16 ties it at 1024, improves
  longer contexts, uses fewer parameters, and has lower positional addend
  magnitude.

Any future work should treat quality, extrapolation, and throughput as three
separate objectives. Reopening an axis requires beating the relevant frozen
endpoint, and logit-bias experiments must report their roughly 2x training
cost relative to fused-SDPA RoPE.