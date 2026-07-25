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