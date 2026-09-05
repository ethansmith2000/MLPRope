# MLPRope Experiment Journal

This is the living record of experiments, implementation changes, conclusions,
and deferred research directions. Add dated entries rather than rewriting old
results when the interpretation or code changes.

## Current status

As of 2026-07-22:

- Phase-1, 1b, and 1c are complete at 10k steps.
- Position foundation refactor (v2 schema + Q/K coupling) is landed.
- Best overall remains Phase-1 **linear logit bias** (4.076 / 58.9).
- **Phase-2 coupling ablation launched**: additive-linear and phase-mlp ×
  `{shared, sep_readout, separate}` (6 jobs), h768/d8, 10k steps.
  Fresh post-refactor RoPE baseline also queued into
  `model-output/position_bias_phase2_coupling/rope-h768d8` (Phase-1 baseline
  at `position_bias_phase1/rope-h768d8` remains intact: 4.129 / 62.1).
- **Phase-2 follow-up launched**: combined
  `add-linear sep_readout + linear logit`, plus sep_readout wideners
  (`mlp`, `low_rank`).
- **Phase-2 follow-up complete.** Combined is a new best: **4.063 / 58.2**.
  Sep_readout helps additive mappers generally, but linear remains strongest.

As of 2026-07-21 (historical snapshot before the refactor):

- Phase-1c tested fixed and mapped **additive Fourier Q/K embeddings**, not a
  faithful reproduction of canonical AddRoPE. Mapped variants (`linear`/`mlp`)
  slightly beat RoPE, while the fixed unit-amplitude addend performed poorly.
- The next Q/K design separates application, output geometry, positional input,
  feature mapper, Q/K coupling, and head coupling rather than overloading one
  `feature_map` name.

## 2026-07-20 — Phase 1: position-only logit biases

### Setup

- Model: hidden size 768, depth 8, 8 heads.
- Batch size 8, sequence length 1024.
- 10,000 optimizer steps, 200 warmup steps.
- Dataset: OpenWebText with the GPT-2 tokenizer.
- Baseline: standard multiplicative RoPE, no learned logit bias.
- Learned variants: standard RoPE on Q/K plus a position-only logit-bias curve.

### Final results

| Variant | Eval loss | Perplexity |
| --- | ---: | ---: |
| RoPE baseline | 4.129 | 62.1 |
| AddRoPE logit bias | 4.126 | 61.9 |
| Phase-1 low-rank implementation | 4.102 | 60.5 |
| MLP logit bias | 4.095 | 60.0 |
| Linear logit bias | **4.076** | **58.9** |

### Interpretation

- Linear is not the baseline; it is the best learned Phase-1 variant.
- A learned relative-position logit correction can improve over standard RoPE.
- More nominal expressiveness did not monotonically improve loss: linear beat
  both nonlinear variants.
- AddRoPE was effectively tied with the RoPE baseline.
- Parameter matching is not yet controlled, but the added parameter counts are
  not large enough to assume parameter count is the sole explanation.

### Important confound

The Phase-1 `low_rank` name did not match its intended design:

```text
documented: D -> r -> D, purely linear
implemented: D -> r -> GELU -> scalar
```

That run remains a valid empirical result for its actual architecture, but it
cannot answer whether a factorized full linear map matches the full-rank linear
variant. The corrected implementation is now called low-rank and keeps the
feature output at `D`; the nonlinear counterpart is separately named
`bottleneck_mlp`.

The completed MLP run also predates the stricter exact-identity initialization
now in the code. Its scalar readout was zero, so the initial logit bias was still
zero, but its residual MLP output matrix used a small random initialization
rather than exact zeros.

## 2026-07-20 — Runtime and compiler findings

The first FlexAttention runs failed because the generated Triton kernels needed
122,880 bytes of shared memory while the GPU limit reported by the compiler was
101,376 bytes.

Working configuration:

- Pin FlexAttention block sizes to 32 and use two stages / four warps.
- Keep FlexAttention outside the outer compiled graph.
- Allow graph breaks (`fullgraph=false` for the outer model).
- Compile FlexAttention with `mode="default"`.

`max-autotune` enabled CUDAGraphs and caused a severe progressive slowdown:
roughly 0.8–1.0 seconds per step initially, growing to 4–6 seconds. A controlled
benchmark reproduced the growth, while `default` stayed flat. Current runs and
future launch configs therefore use default compile mode.

## 2026-07-20 — Dual-channel architecture

Position mechanisms are now factored into two independently enabled channels:

```yaml
qk:
  enabled: false
  feature_map: identity
  sharing: per_head
  apply: phase_residual

logit_bias:
  enabled: false
  feature_map: identity
  sharing: per_head
```

Each channel independently selects:

- Feature map: identity, AddRoPE affine, linear, low-rank,
  bottleneck MLP, or MLP.
- Sharing: shared across heads, independent per head, or a joint full-dimension
  map.

The Q/K channel additionally selects:

- `add`: additive Q/K replacement — add `e(p)` to Q/K and **skip**
  multiplicative RoPE. This has the same application site as AddRoPE, but the
  current implementation shares one addend across Q/K and does not implement
  canonical AddRoPE's separate amplitude and angular-offset parameters.
- `phase_residual`: predict a phase delta and compose it with standard RoPE.
  A zero phase delta converts to the identity rotation.

The logit channel always ends in a zero-initialized scalar readout. Both channels
disabled is the standard RoPE baseline; both channels may be enabled together.

Legacy `pos_variant` configs remain accepted as logit-channel presets so the
completed Phase-1 configs remain interpretable.

### Verification

- CPU shape and zero-initialization tests cover all feature maps and sharing
  modes.
- Tests verify that the Q/K phase residual exactly matches RoPE at
  initialization. The additive path deliberately starts with a nonzero
  positional addend and skips multiplicative RoPE, so it does not match RoPE.
- Compiled CUDA forward/backward smoke tests passed for Q/K phase, Q/K add,
  logit-only, and both channels together.

## 2026-07-21 — Phase 1b launched

Queued four jobs (`GPU_SELECTOR=any`, `PARALLEL=true`, WandB on):

1. Corrected linear low-rank logit bias: `D -> r -> D` (`r=32`, per-head).
2. Bottleneck-MLP logit bias: `D -> r -> GELU -> D` (`r=32`, per-head).
3. Per-head linear phase residual on Q/K.
4. Per-head MLP phase residual on Q/K (`mlp_hidden=128`).

Three claimed free GPUs immediately; `qk-phase-linear` waited via `gpu-claim --wait`.
Configs: `sweep_configs/`. Logs: `logs/<job>.log`. Project: `mlprope-position-bias`.

## 2026-07-21 — Phase 1b results

Same recipe as Phase 1 (h768 d8, bs8, 10k steps). Anchors not re-run.

| Variant | Channel | Eval loss | Perplexity |
| --- | --- | ---: | ---: |
| RoPE baseline (Phase 1) | — | 4.129 | 62.1 |
| Linear logit (Phase 1 best) | logit | **4.076** | **58.9** |
| Corrected low-rank logit `r=32` | logit | 4.081 | 59.2 |
| Bottleneck-MLP logit `r=32` | logit | 4.084 | 59.4 |
| Phase-1 confounded "low_rank" | logit | 4.102 | 60.5 |
| Phase-1 MLP logit | logit | 4.095 | 60.0 |
| Linear phase residual | Q/K | 4.109 | 60.9 |
| MLP phase residual | Q/K | 4.131 | 62.2 |

### Interpretation

- **Full linear logit bias remains best**, but corrected factorized linear
  (`D→r→D`, r=32) closes almost all of the gap (−0.048 vs −0.053 vs RoPE).
- Bottleneck nonlinearity does not help over the factorized linear map
  (4.084 vs 4.081); both beat the confounded Phase-1 "low_rank" and the older MLP.
- **Logit channel >> Q/K phase residual** under this recipe. Linear phase helps
  modestly (−0.020 vs RoPE); MLP phase is effectively tied with / slightly worse
  than RoPE.
- Rank-32 is enough to approximate the useful part of the full `D→D` logit map.
- Bias diagnostics: corrected low-rank curves have large per-layer abs-max
  (up to ~59) and positive mean; bottleneck_mlp tends toward negative mean with
  smaller abs-max. Q/K-only runs do not emit logit-bias diagnostics.

### Implication for next steps

Prefer staying on the logit channel (or combining channels) rather than
pursuing richer Q/K phase maps alone. Natural follow-ups: content-conditioned
logit bias (Inkling), param-matched FFN controls, and optionally
`logit linear + qk phase linear` together.

## 2026-07-21 — Fix: additive replacement semantics for `qk.apply=add`

Two bugs in the prior Q/K `add` path relative to the intended additive
replacement semantics:

1. Multiplicative RoPE still ran after the addend (`RoPE(q+e)`).
2. A zero-initialized readout erased the sinusoid addend at step 0.

Fix: `apply=add` uses the feature-map output directly as `e(p)`, adds it to Q/K,
and disables multiplicative RoPE. Phase residual is unchanged. Phase-1b only
ran `phase_residual`, so completed numbers are unaffected. Phase-1c then tested
identity, Euclidean affine, linear, low-rank, and MLP additive Fourier maps.

Important later correction: this fixed the *application site* (`q+e`, no
`R(theta)q`) but did not make the implementation canonical AddRoPE. Canonical
[AddRoPE](https://jonathanc.net/blog/additive-rotary-embedding) uses separate
Q/K amplitudes and offsets inside the angle:

```text
q' = q + a_q * cis(omega*p + delta_q)
k' = k + a_k * cis(omega*p + delta_k)
```

The Phase-1c implementation instead shared one Q/K addend; its `add_rope`
feature map used an unconstrained Euclidean coordinate scale/offset rather than
an angular offset.

## 2026-07-21 — Phase 1c results (additive Fourier Q/K)

Same recipe. Multiplicative RoPE off; `q' = q + e(p)`.

| Variant | Eval loss | Perplexity |
| --- | ---: | ---: |
| RoPE baseline (Phase 1) | 4.129 | 62.1 |
| Linear logit (Phase 1 best) | **4.076** | **58.9** |
| Q/K phase linear (Phase 1b) | 4.109 | 60.9 |
| Additive Fourier + MLP | 4.116 | 61.3 |
| Additive Fourier + linear | 4.116 | 61.3 |
| Additive Fourier + low_rank | 4.124 | 61.8 |
| Additive Fourier + Euclidean affine | 4.139 | 62.7 |
| Fixed unit sinusoidal addend | 4.212 | 67.5 |

### Interpretation

- A **fixed unit sinusoidal Q/K addend** is worse than RoPE here
  (+0.084 loss / +5.4 PPL).
- Learned feature maps on the addend recover a lot: identity 4.212 → mlp/linear
  ~4.116, which slightly beats RoPE (−0.013).
- Within this additive family, the extension hypothesis holds: `f(cis)` helps
  over raw `cis`. Residual low-rank is close to linear/MLP.
- Still **far behind RoPE + linear logit bias**, and slightly behind keeping RoPE
  with a linear phase residual.
- These runs do **not** establish that canonical AddRoPE underperforms: they
  omitted separate Q/K amplitude/phase parameters, and the fixed run injected a
  large unit-amplitude vector. The learnable amplitude in AddRoPE may be
  important precisely because it can attenuate or disable frequencies.

## 2026-07-21 — Consolidated Q/K position research framing

### Central hypothesis

RoPE may derive its advantage over residual-stream positional embeddings from
two separable properties:

1. **Attention locality:** position acts directly on Q/K routing, where it is
   needed, without forcing a fixed positional representation into the semantic
   residual stream.
2. **Rotation geometry:** each paired subspace is transformed by a
   norm-preserving rotation.

The broader objective is to retain useful positional structure while making the
operation learnable. Rather than learning an unrelated vector for every
absolute index, learn a function of position (usually from a Fourier basis), so
nearby or structurally related positions still produce correlated outputs.

The logit-bias branch remains a separate, compatible research direction. It
acts directly on relative routing scores and is not mutually exclusive with Q/K
position transforms.

### Relationship to sinusoidal PE and canonical AddRoPE

Classic sinusoidal PE adds a full-model-dimensional deterministic basis to the
token/residual stream:

```text
x_0(p) = token(p) + z_D(p)
```

It is available to all subsequent layers and Q/K projections, but the network
must simultaneously preserve useful positional components, transform them for
routing, and suppress components that interfere with semantic representation.
This motivates the "pollution versus dilution/cancellation" hypothesis in the
central framing.

Canonical AddRoPE can be viewed as a local, per-layer cousin of sinusoidal PE:
it adds Fourier features to Q/K rather than to the residual stream:

```text
q' = q + a_q * cis(omega*p + delta_q)
k' = k + a_k * cis(omega*p + delta_k)
```

It therefore:

- keeps positional injection local to attention;
- uses separate learned Q/K parameters;
- learns per-frequency amplitude and **angular** offset (the offset is
  phase-aware, not a Euclidean coordinate shift);
- does not directly write the positional basis into the residual stream.

The equations above do not by themselves establish whether parameters must be
shared across heads. Head coupling is an experimental axis. Natural extensions
include head-specific amplitude/phase, independent Q/K positional maps, and a
joint full-dimension transform such as `Linear(fourier_basis_dim, dim)` whose
output is reshaped across heads.

The fixed Phase-1c identity addend had no learned amplitude. With normalized Q/K
components, a unit sinusoidal vector has norm about `sqrt(D/2)` versus roughly
`sqrt(D)` for a normalized Q/K vector, so the injected addend is not small. This
provides a plausible—not yet isolated—explanation for its poor result and makes
learned amplitude an important control.

### Additive and rotary geometry

Additive Q/K does not require the final addend to lie in cosine/sine space:

```text
free direct:    e(p) = f(z(p))
free residual:  e(p) = z(p) + f(z(p))
phase:          e(p) = cis(omega*p + delta(p))
amplitude+phase:e(p) = a(p) * cis(omega*p + delta(p))

q' = q + e_q(p)
k' = k + e_k(p)
```

Here `z(p)` is usually a Fourier embedding. Free direct output resembles a
generated learned absolute embedding; free residual output preserves a frozen
sinusoidal component alongside a zero-initialized learned correction. The
phase and amplitude+phase forms deliberately preserve more trigonometric
structure. Additive amplitude need not equal one and can learn to suppress
unhelpful frequencies.

Strict multiplicative RoPE is more constrained:

```text
q' = R(phi_q(p)) q
k' = R(phi_k(p)) k
phi(p) = omega*p + delta(p)
```

A valid norm-preserving rotation needs one phase per pair, so the natural final
readout is `head_dim/2` (shared-head) or `dim/2` (all heads). An arbitrary
`D`-dimensional cosine/sine output is not a valid rotation unless every pair is
projected back to the unit circle.

Scaling outside the rotation, `a*R(phi)q`, is a scaled rotation and changes Q/K
norms and attention-logit magnitude. It may be an interesting positional gate,
but it must be labeled separately from strict RoPE. Scaling the *angle*,
`R(a*phi + delta)`, changes frequency/temperature while remaining a valid
rotation.

Canonical AddRoPE's parameter types suggest useful multiplicative controls, but
they do not transfer identically:

- angular offset `delta` remains a valid rotary operation;
- changing `omega` or applying an angle scale changes frequency while retaining
  a valid rotation;
- additive amplitude has no strict-rotation analogue—multiplying rotated Q/K
  changes norm and becomes a gate/scaled rotation;
- constant Q/K phase offsets may be partly reparameterizable into learned Q/K
  projections (depending on Q/K normalization), so learned frequency and
  position/content-dependent phase are more substantive rotary tests.

### Phase parameterization and pair projection

Do not recover a shared phase by applying `arccos` and `arcsin` independently:
their principal branches disagree for much of the circle. The base angle
`omega*p` is already known. Compose a predicted phase residual directly:

```text
c' = c*cos(delta) - s*sin(delta)
s' = s*cos(delta) + c*sin(delta)
```

This pair rotation is equivalent to `cis(omega*p + delta)`, reuses cached base
cosine/sine values, and preserves unit-circle geometry without inverse trig.
The implementation must use one explicit, consistent pair layout (adjacent
`[cos_i,sin_i]` or split-half) throughout.

An alternative projected-phase parameterization is:

```text
u = f(z) in R^D
(c',s') = normalize_each_pair(u)
```

This is valid: every nonzero normalized pair defines a rotation. The discarded
radial component is not needed for a phase-only target. Its drawback is
parameter redundancy and poor behavior near zero, not lost useful phase
capacity. Predicting `D/2` phase residuals is usually cleaner and more
identifiable. Uniformly scaling a pair and then normalizing is a no-op; an
anisotropic or general map followed by normalization induces an indirect phase
warp.

Useful projected-phase variants are therefore:

1. `D/2` direct phase residual (preferred default);
2. `D` arbitrary vector followed by pair normalization;
3. per-pair anisotropic scale followed by normalization (indirect phase warp);
4. angle/frequency scale before trig, `cis(a*omega*p + delta)`;
5. scaled rotary `a*R(phi)` only when deliberately giving up norm preservation.

### Positional input and feature mapper are separate axes

Cosine/sine features are a deterministic Fourier embedding of scalar index:

```text
phi_i(p) = omega_i*p
z_i(p) = (cos(phi_i(p)), sin(phi_i(p)))
```

Candidate inputs:

- frozen RoPE/Fourier features (default);
- scalar features such as normalized `p` or `log1p(p)`, usually concatenated
  with rather than replacing Fourier features;
- learned-temperature Fourier features;
- independently learned frequencies;
- later, position features combined with token/residual/Q/K content.

One explicit mixed-input ablation is:

```text
position_input = concat(
    Fourier(p),
    p / max_train_length,
    log1p(p),
)
```

The scalar channels should be standardized. Length normalization changes the
meaning of a position across context lengths, so extrapolation must be measured;
`log1p(p)` is a useful count-like coordinate but is less naturally positional
than the multi-scale Fourier basis.

Fourier `basis_dim` should be independent of output dimension. A compact basis
may be sufficient when a mapper produces `head_dim`, `dim`, `head_dim/2`, or
`dim/2`. Learned frequencies should preferably start as residuals around a
known schedule, e.g. `omega = omega_base * exp(delta_log_omega)`. A shared
temperature preserves ordering; fully independent frequencies are more
expressive but can duplicate or collapse.

The existing `calib_attn.attention_scaling.FourierEmbedder` is a useful template:
it already separates scalar input, number of frequencies, and frozen versus
learned-temperature versus learned-frequency modes. For position experiments,
use a RoPE-compatible initial frequency schedule and keep Fourier basis width
independent from model output width.

Feature-mapper choices remain an independent axis:

- identity / affine;
- full linear;
- low-rank linear;
- bottleneck MLP;
- MLP;
- later, content-conditioned modulation.

### Head coupling

Standard RoPE broadcasts the same `head_dim/2` frequency schedule to every
head. Heads still learn different roles through their Q/K projections, but RoPE
itself does not assign different schedules to different heads.

Classic residual-stream sinusoidal PE spans the full model dimension before
Q/K projection. Q/K projections can mix all of those frequencies into every
head; it should not be described as directly assigning one frequency slice to
each head.

Use the following names:

- `shared_head`: one head-sized output broadcast to all heads;
- `per_head_independent`: independent mapper/readout for each head;
- `per_head_joint`: one model-dimension map jointly produces all heads, then
  reshapes.

Output sizes:

- free additive: `head_dim` shared, otherwise `dim`;
- phase-only: `head_dim/2` shared, otherwise `dim/2`;
- additive amplitude+phase: two phase-sized outputs, conceptually `(a,delta)`.

### Q/K coupling and relative invariance

Q and K may use:

- one fully shared function;
- a shared positional trunk with separate Q/K readouts (preferred default);
- fully separate functions.

This axis applies to additive and rotary variants. It is especially relevant
under causal attention: query position controls how many keys are visible,
while a key may serve many later queries. Existing Q/K projections already
create role asymmetry, but separate positional readouts may expose additional
useful structure.

For rotary transforms:

```text
q'(p)^T k'(s) = q(p)^T R(phi_k(s) - phi_q(p)) k(s)
```

Exact dependence only on offset requires compatible affine phase functions of
position. Shared frequency slopes with separate constant Q/K phase offsets
retain a clean relative form. Arbitrary nonlinear `delta(p)`, or arbitrary
separate Q/K phase functions, generally introduces absolute-position
dependence. That relaxation may help, but should be described and evaluated
explicitly rather than assumed to remain strictly relative.

### Content-aware Q/K position transforms

Position-only functions can later become content-aware:

```text
additive:
  e_q = f_q(q, z(p))
  e_k = f_k(k, z(p))

rotary:
  delta_q = f_q(q, z(p))
  delta_k = f_k(k, z(p))
  q' = R(omega*p + delta_q) q
  k' = R(omega*p + delta_k) k
```

Conditioning on each token's own residual/Q/K state remains compatible with KV
caching. `f(x)` without explicit position can still content-modulate a fixed
base rotation; `f(x,z(p))` permits explicit content-position interaction.
Pairwise query-key conditioning is a different, more expensive mechanism.

### Residual-stream position and attention-output writes

Residual-stream sinusoidal PE may create competing pressures: preserve
position for later routing, transform it into useful semantic features, and
cancel frequencies that interfere with content. Residual connections prevent
literal disappearance, but normalization and accumulated updates can dilute,
mix, or cancel the original positional signal.

This motivates a separate attention-output positional write:

```text
content_i = sum_j A_ij * V_j
pos_i     = sum_j A_ij * g(position_j)
y_i       = W_content(content_i) + W_pos(pos_i)
```

Unlike adding the query's absolute position after attention, this tells the
residual stream where retrieved information came from. A relative variant uses
`g(i-j)` and writes a summary of attended offsets. Zero-initialize `W_pos` or a
gate so the channel can collapse exactly to null when unused.

This tests a distinct hypothesis: positional information in the residual
stream may be useful when it is selectively transformed and reinjected at each
layer, rather than inserted once as a fixed basis.

Before testing attention-output writes, establish residual-stream baselines:

```text
standard sinusoidal:
  x_0 = token + z_D(p)

functional sinusoidal:
  x_0 = token + Linear(z_K(p))
  x_0 = token + MLP(z_K(p))

gated/per-layer functional:
  x_l = x_l + gate_l * f_l(z_K(p))
```

Here `K` may be smaller than `dim`; the mapper learns the residual-stream
representation. Include learned absolute position embeddings as the fully free
comparison. Zero-initialized per-layer gates test whether reinjection is useful
without forcing every layer to consume the positional channel.

### Revised experiment axes

Treat these as independent choices:

1. **Application:** additive Q/K or rotary Q/K.
2. **Output geometry:** free direct, free residual, phase,
   additive amplitude+phase, or projected phase.
3. **Position input:** frozen Fourier, learned-temperature Fourier,
   learned-frequency Fourier, optional scalar features.
4. **Feature mapper:** affine, linear, low-rank, bottleneck MLP, MLP.
5. **Q/K coupling:** shared, shared trunk/separate readouts, separate.
6. **Head coupling:** shared-head, per-head-independent, per-head-joint.
7. **Content conditioning:** none, local residual/Q/K content plus position, or
   content-only modulation of a fixed positional prior.
8. **Residual-stream write:** none, query-position reinjection, attended
   key-position summary, or attended relative-offset summary.

The relative-logit channel has its own orthogonal axes: offset representation,
head sharing, content conditioning, and causal/visible-key effects.

Develop each sector with clean individual controls first:

1. residual-stream position;
2. additive Q/K position;
3. rotary Q/K position;
4. relative-logit position.

After identifying strong variants within each sector, test their union. In
particular, Q/K geometry and relative-logit bias can be complementary rather
than competing replacements.

## Intended future work

Near-term:

1. **Faithful AddRoPE control**
   - Separate Q/K amplitude and angular-offset readouts.
   - Compare shared-head, per-head-independent, and per-head-joint.
   - Initialize amplitudes deliberately; the poor fixed-unit result suggests
     that its injection may have been too large.
2. **Clean additive geometry ablation**
   - Free direct, free residual, phase-only, and amplitude+phase from the same
     Fourier input and parameter-matched mapper.
3. **Clean rotary geometry ablation**
   - `D/2` phase residuals, learned frequency/temperature, and optionally
     projected-phase output; preserve strict rotation unless explicitly testing
     scaled rotary.
4. **Q/K coupling ablation**
   - Shared versus shared trunk/separate readouts versus fully separate.
5. **Input-basis efficiency**
   - Decouple Fourier basis width from model/head width; compare frozen,
     learned-temperature, and residual learned-frequency bases.
6. **Parameter-matched wider-FFN controls**
   - Spend the position module's extra parameters in the baseline FFN.
7. **Residual-stream functional sinusoidal controls**
   - Standard sinusoidal, learned absolute, `Linear(z_K)`, and `MLP(z_K)`.
   - Then test zero-init gated per-layer reinjection.

Later:

8. **Content-aware Q/K position functions**
   - Start with local, KV-cache-compatible `f(q,z(p))` / `f(k,z(p))`.
9. **Attention-output positional writes**
   - Attended key-position and relative-offset summaries with zero-init gates.
10. **Inkling / content-conditioned logit banks**
   - Query-dependent mixtures over learned relative-distance profiles.
   - Table and CosNet/function-parameterized versions.
11. **Combined channels**
    - Test whether the best Q/K mechanism complements linear or
      content-conditioned relative-logit bias.

Additional later analyses:

- Translation/offset invariance diagnostics for learned rotary functions.
- Q/K amplitude, phase, frequency, and head-specialization diagnostics.
- Attention entropy and length-extrapolation evaluation.
- Content-routing diagnostics once Inkling is implemented.

## Reproducibility pointers

- Design: `position_embedding_experiments.md`
- Config schema: `POSITION_CONFIG.md`
- Position package: `position/`
- Model: `transformer.py`
- Trainer and diagnostics: `train_gpt.py`
- Sweep launcher: `launch_position_bias.sh`
- Phase-1b/1c configs: `sweep_configs/`
- Run outputs: `model-output/`
- WandB project: `mlprope-position-bias`

---

## 2026-07-22 — Position foundation refactor (v2 schema + Q/K coupling)

Implemented the middle-scope foundation refactor without launching new training
sweeps or rewriting completed Phase 1/1b/1c artifacts.

**Code**

- New package `position/` with frozen Fourier basis, mappers, rotary helpers,
  head-coupled pipelines, Q/K + logit channels, config upgrade/validation, and
  state-dict adaptation.
- `Attention` consumes `QKPositionOutput` and supports separate Q/K phase
  deltas. FlexAttention compile boundaries unchanged.
- Trainer resolves all configs to `position_schema_version=2`, records
  `position_source_schema`, preserves legacy auto-run tags for v1 sources, and
  emits compact Q/K diagnostics alongside existing logit bias metrics.

**New capability (only intentional behavior addition)**

- `qk_coupling`: `shared` (v1 parity) |
  `shared_trunk_separate_readouts` (identity/zero dual readouts) |
  `separate` (deep-copied twin pipelines).

**Deferred (explicit errors if requested)**

- Canonical amplitude+phase AddRoPE, learned Fourier inputs, projected-phase,
  content-aware maps, residual-stream PE, attention-output writes, Inkling.

**Verification**

- `python -m unittest test_position_channels -v`
- Dry-run of all 14 `sweep_configs/` (param counts / SDPA-Flex / enabled channels)
- Claimed-GPU eager+compiled smoke (see `scripts/position_v2_cuda_smoke.py`)

**Checkpoint note**

- Shared-coupling model weights migrate via key adapter.
- Optimizer-state resume across the v1→v2 parameter rename is **not** guaranteed.

---

## 2026-07-23 — Phase 2 coupling + follow-up results

Same recipe (h768/d8, bs8, 10k, seq 1024). Fresh post-refactor RoPE matched the
Phase-1 baseline.

### Coupling ablation (additive-linear and phase-mlp)

| Variant | Eval loss | PPL |
| --- | ---: | ---: |
| RoPE (fresh) | 4.129 | 62.1 |
| add-linear **sep_readout** | **4.082** | **59.3** |
| add-linear separate | 4.104 | 60.6 |
| add-linear shared | 4.117 | 61.4 |
| phase-mlp shared / sep / separate | ~4.13–4.14 | ~62.2–62.5 |

Interpretation: for additive free geometry, separate Q/K readouts from a shared
trunk help a lot. Fully separate trunks help less than sep_readout. Rotary phase
MLP is insensitive to coupling and stays ~RoPE.

### Follow-up: combine + widen sep_readout

| Variant | Eval loss | PPL |
| --- | ---: | ---: |
| **add-linear sep_readout + linear logit** | **4.063** | **58.2** |
| linear logit only (Phase 1) | 4.076 | 58.9 |
| add-linear sep_readout only | 4.082 | 59.3 |
| add-low_rank sep_readout | 4.094 | 60.0 |
| add-mlp sep_readout | 4.098 | 60.2 |

Interpretation:

- Q/K additive PE and relative logit bias **stack** (new best overall).
- Sep_readout generalizes across additive mappers, but **linear** remains the
  best mapper under that coupling.
- Gains vs RoPE: combined −0.066 loss / −3.9 PPL.

---

## 2026-07-23 — Fully configurable position playground

Expanded the v2 foundation into the complete research playground after the
Phase-2 result established that additive Q/K and relative logits are
complementary.

### Implemented axes

- Position input: frozen Fourier, learned global temperature, independent
  learned frequencies, reduced basis width, and optional scalar features.
- Additive geometry: historical free Fourier addends plus canonical
  amplitude+phase AddRoPE with explicit low-amplitude initialization.
- Rotary geometry: phase residual, tangent-projected phase, and scaled rotary.
- Content-aware Q/K: zero-init local residual and content-gated positional
  outputs, preserving token-local/KV-cache-safe semantics.
- Residual stream: fixed/functional Fourier or learned absolute position at
  input, per layer, or both, with shared/layer-specific gates.
- Attention output: attended key-position and Fourier-derived relative-offset
  summaries, zero-gated into the residual stream.
- Relative logits: static curves, query-routed bounded Inkling tables, and
  query-routed bounded CosNet profile banks.
- Controls: explicit `ff_hidden_dim` and a dry-run parameter-matched GeGLU
  recommendation.

### Initialization contracts

- Learned Fourier inputs exactly match frozen Fourier at step 0.
- Projected phase and phase residual start as exact RoPE.
- Scaled rotary starts with phase 0 and scale 1.
- Local Q/K residuals and content gates preserve their positional base.
- Residual-stream and attention-output writes default to zero gates.
- Inkling profile banks use small symmetry-breaking profiles behind a zero
  gate, preserving baseline logits while allowing the gate to learn.

### Verification

- Existing foundation tests plus `test_position_playground.py`.
- Recursive loading of all historical and Phase-2 JSON configs.
- Claimed-GPU eager and compiled forward/backward coverage for every new
  family and representative combinations.

No new 10k training sweep was launched by this implementation.

---

## 2026-07-23 — Phase 3 canonical-geometry screen launched

Launched an isolated eight-run, 3k-step screening bundle under
`sweep_configs/phase3_geometry/` and
`model-output/position_bias_phase3_geometry/`. Validation runs every 500 steps;
WandB group is `phase3-geometry-screen`.

The comparison bundle contains:

- fresh RoPE;
- additive-linear shared-trunk/separate-readout;
- that additive control plus linear relative-logit bias;
- canonical amplitude+phase AddRoPE initialized at amplitudes
  `0.01`, `0.03`, `0.1`, and `0.3`;
- canonical amplitude `0.1` plus linear relative-logit bias.

Canonical and free additive Q/K controls are exactly parameter matched at
1,787,904 position parameters. Their logit combinations are also matched at
2,390,080. All eight configs passed CPU dry runs and entered training through
the shared lifetime-locking `gpu-claim` queue; no historical config or output
directory was reused.

### 3k screening result

All eight runs completed cleanly. Final eval loss ranked:

- canonical amplitude `0.1` + linear logit bias: `4.74317`;
- additive-linear + linear logit bias: `4.75412`;
- canonical amplitude `0.3`: `4.76764`;
- canonical amplitudes `0.1`, `0.03`, `0.01`: `4.79254`, `4.79935`,
  `4.80390`;
- additive-linear: `4.81469`;
- RoPE: `4.81696`.

Within the tested canonical range, larger initialization amplitude improved the
3k result monotonically. The strongest run was canonical amplitude `0.1` plus
linear logit bias (perplexity `114.80`), ahead of the parameter-matched
additive-linear/logit control (perplexity `116.06`). This is a screening result
from one seed; promotion should test the leading logit pair and extend the
canonical amplitude range before committing to longer multi-seed finalists.

---

## 2026-07-23 — Phase 3 canonical-amplitude follow-up launched

Launched a second isolated eight-run, 3k-step screen under
`sweep_configs/phase3_amplitude_followup/` and
`model-output/position_bias_phase3_amplitude_followup/`. Validation runs every
500 steps; WandB group is `phase3-amplitude-followup`.

The bundle extends the Q/K-only canonical amplitude range with `0.5`, `0.7`,
`1.0`, and `2.0`. It also brackets and extends the previous amplitude-`0.1`
winner with linear relative-logit bias at amplitudes `0.03`, `0.3`, `1.0`, and
`2.0`.

All eight configs passed CPU dry runs. Q/K-only runs have 1,787,904 position
parameters and logit combinations have 2,390,080, matching the corresponding
first-screen controls. The runs acquired all eight GPUs through the shared
lifetime-locking `gpu-claim` queue and each completed its first 500-step
validation without a startup error.

### 3k follow-up result

The initial launcher was interrupted after the four Q/K-only runs completed.
The four slower logit runs were requeued through a one-shot supervisor service
and then completed cleanly.

Final eval loss ranked:

- canonical amplitude `1.0` + linear logit bias: `4.7306`;
- canonical amplitude `0.3` + linear logit bias: `4.7353`;
- canonical amplitude `0.03` + linear logit bias: `4.7417`;
- canonical amplitude `1.0`: `4.7488`;
- canonical amplitudes `0.7`, `0.5`: `4.7504`, `4.7511`;
- canonical amplitude `2.0` + linear logit bias: `4.7549`;
- canonical amplitude `2.0`: `4.7723`.

Together with the first screen, both curves improve through roughly amplitude
`1.0` and regress at `2.0`. The new best 3k result is amplitude `1.0` plus
linear logit bias (perplexity `113.37`), improving on the prior amplitude-`0.1`
winner (`4.74317`) and the parameter-matched free-additive/logit control
(`4.75412`). The next promotion target is therefore canonical amplitude `1.0`
plus linear logit bias, with amplitude `0.3` as the nearest canonical/logit
control.

---

## 2026-07-23 — Phase 3 two-seed 10k promotion launched

Launched an isolated eight-run promotion bundle under
`sweep_configs/phase3_promotion/` and
`model-output/position_bias_phase3_promotion/`. The four-way comparison is run
at seeds `123` and `456`:

- RoPE;
- free additive-linear Q/K plus linear relative-logit bias;
- canonical amplitude `0.3` plus linear relative-logit bias;
- canonical amplitude `1.0` plus linear relative-logit bias.

All runs use FlexAttention to control the attention implementation across the
comparison. The three learned-position candidates are exactly matched at
2,390,080 position parameters; RoPE remains the zero-extra-parameter baseline.
All eight configs passed CPU dry runs, acquired GPUs through the shared
lifetime-locking `gpu-claim` queue, and completed their first 1,000-step
validation without a startup error. The suite runs under one-shot supervisor
management and cannot autostart after a supervisor restart. WandB group is
`phase3-promotion-10k`.

### 10k promotion result

All eight runs completed cleanly. Final eval loss by seed:

- RoPE: `4.1284` (seed `123`), `4.1549` (seed `456`);
- free additive + linear logit: `4.0615`, `4.0647`;
- canonical amplitude `0.3` + linear logit: `4.0525`, `4.0491`;
- canonical amplitude `1.0` + linear logit: `4.0515`, `4.0534`.

Two-seed mean eval loss is `4.1417` for RoPE, `4.0631` for free additive,
`4.0508` for canonical amplitude `0.3`, and `4.0525` for canonical amplitude
`1.0`. Canonical geometry therefore beats the parameter-matched free-additive
control on both seeds by roughly `0.01` loss and beats RoPE by roughly `0.09`.
The amplitude ordering flips across seeds and the two canonical means differ by
only `0.0017`; the durable result is the canonical geometry advantage, not a
resolved preference between amplitudes `0.3` and `1.0`.

---

## 2026-07-23 — Phase 3 basis-adaptability screen launched

Launched an isolated eight-run, two-seed, 5k-step screen under
`sweep_configs/phase3_basis_screen/` and
`model-output/position_bias_phase3_basis_screen/`. Canonical amplitude `0.3`,
linear relative-logit bias, coupling, mapper, and FlexAttention are held fixed.
At seeds `123` and `456`, the Q/K position input is:

- frozen Fourier;
- learned-temperature Fourier;
- learned-frequency Fourier;
- frozen Fourier plus normalized-position and log-position scalars.

All configs passed CPU dry runs and completed their first 500-step validation
after acquiring GPUs through `gpu-claim`. Position parameter counts are
2,390,080 for frozen, 2,390,088 for learned temperature, 2,390,464 for learned
frequency, and 2,402,368 for scalar augmentation. The small count differences
are intrinsic to the tested basis parameters/features rather than mapper or
geometry changes. The suite runs under non-autostarting one-shot supervisor
management; WandB group is `phase3-basis-screen-5k`.

### 5k basis-screen result

All eight runs completed cleanly. Final eval loss by seed:

- frozen Fourier: `4.4041` (seed `123`), `4.4018` (seed `456`);
- learned temperature: `4.4039`, `4.4025`;
- learned frequencies: `4.4041`, `4.4026`;
- normalized/log-position scalars: `4.3992`, `4.3896`.

Two-seed mean eval loss is `4.4030` for frozen Fourier, `4.4032` for learned
temperature, `4.4034` for learned frequencies, and `4.3944` for scalar
augmentation. Learning the Fourier temperature or individual frequencies
provides no measurable gain at this horizon. Adding normalized and logarithmic
absolute-position scalars improves both seeds and lowers mean loss by `0.0086`
relative to frozen Fourier, despite adding only 12,288 position parameters.
Scalar augmentation is the basis variant worth carrying into the next geometry
or conditioning screen.

---

## 2026-07-24 — Phase 3 scalar geometry-transfer screen launched

Launched an isolated eight-run, two-seed, 5k-step screen under
`sweep_configs/phase3_geometry_transfer/` and
`model-output/position_bias_phase3_geometry_transfer/`. Normalized-position and
log-position scalars, frozen Fourier inputs, linear relative-logit bias,
coupling, mapper, and FlexAttention are held fixed. At seeds `123` and `456`,
the Q/K geometry is:

- canonical additive amplitude+phase at amplitude `0.3`;
- rotary phase residual;
- projected-phase rotary;
- scaled-phase rotary.

All configs passed CPU dry runs, acquired GPUs through `gpu-claim`, and
completed their first 500-step validation. Canonical, projected-phase, and
scaled-phase runs each have 2,402,368 position parameters. The naturally
leaner phase-residual geometry has 1,806,400, so comparisons involving that
variant include a capacity difference; the other three are exactly matched.
The suite runs under non-autostarting one-shot supervisor management; WandB
group is `phase3-geometry-transfer-5k`.

### 5k scalar geometry-transfer result

All eight runs completed cleanly. Final eval loss by seed:

- canonical amplitude+phase: `4.3980` (seed `123`), `4.3895` (seed `456`);
- phase residual: `4.4866`, `4.4723`;
- projected phase: `4.4893`, `4.4731`;
- scaled phase: `4.4617`, `4.4569`.

Two-seed mean eval loss is `4.3938` for canonical, `4.4795` for phase residual,
`4.4812` for projected phase, and `4.4593` for scaled phase. Canonical therefore
retains a large advantage when scalar augmentation is held fixed; none of the
rotary alternatives is close enough to promote from this screen.

---

## 2026-07-24 — Phase 3 one-seed frontier screen launched

Launched seven new 5k-step runs at seed `123` under
`sweep_configs/phase3_frontier_screen/` and
`model-output/position_bias_phase3_frontier_screen/`. Canonical amplitude `0.3`,
normalized/log-position scalars, and frozen Fourier input are fixed. The new
mechanisms are:

- local-residual and content-gated Q/K conditioning, each with hidden width `32`;
- Inkling table and cosine-network relative-logit bias;
- zero-gated per-layer residual-stream position reinjection;
- zero-gated key-position and relative-offset attention-output writes.

The exact linear-logit anchor is not rerun: the completed geometry-transfer run
at this seed and horizon is reused (`eval_loss=4.3980`). All seven new configs
passed CPU dry runs. They were submitted under non-autostarting one-shot
supervisor management through the shared lifetime-locking `gpu-claim` queue;
four immediately acquired free GPUs and three are waiting without bypassing
foreign claims. This is a mechanism screen rather than a parameter-matched
capacity comparison. WandB group is `phase3-frontier-screen-5k`.

### 5k frontier-screen result

All seven new runs completed. The reused anchor is `4.3980`; final eval losses
are `4.3982` for key-position write, `4.3984` for relative-offset write,
`4.3985` for per-layer residual reinjection, `4.4030` for Inkling table, and
`4.4196` for Inkling cosnet. The first three learned gates remained small and
were empirically indistinguishable from the anchor; neither Inkling variant
improved it.

The original local-residual and content-gate Q/K conditioners failed
optimization at `6.0208` and `6.2365`. They completed without runtime errors or
non-finite values, but their positional addends reached RMS values in the
hundreds and maxima in the thousands. These are treated as unstable mechanisms,
not competitive results.

---

## 2026-07-24 — Bounded Q/K conditioning retry launched

Bounded the content-gate multiplier with `tanh` to `[0, 2]` and bounded each
local-residual latent correction to `[-1, 1]`; both still reproduce the
unconditioned model exactly at zero initialization and retain immediate
gradients. Added a focused extreme-logit bound test. All 23 playground tests and
25 position-channel regression tests pass.

Launched isolated seed-`123`, 5k-step retries for the two conditioners under
`sweep_configs/phase3_conditioning_retry/` and
`model-output/position_bias_phase3_conditioning_retry/`. Both configs passed CPU
dry runs and acquired GPUs through `gpu-claim`; WandB group is
`phase3-conditioning-bounded-retry-5k`.

### Bounded conditioning retry result

Bounding the direct conditioner outputs did not rescue either mechanism.
Content-gate conditioning was non-finite by the first 500-step validation.
Local-residual conditioning remained finite but moved only from `6.3530` at
step 500 to `6.1283` at step 5,000. Its underlying positional trunk still grew
to RMS values in the hundreds and maxima in the thousands. Both direct Q/K
content-conditioning mechanisms are retired from the current search.

---

## 2026-07-24 — Canonical coupling-transfer screen launched

Launched four new seed-`123`, 5k-step structural-sharing runs under
`sweep_configs/phase3_coupling_transfer/` and
`model-output/position_bias_phase3_coupling_transfer/`. Canonical amplitude
`0.3`, normalized/log-position scalars, frozen Fourier input, linear logit bias,
and all optimization settings are fixed. The structures are:

- fully shared Q/K with independent heads;
- fully separate Q/K with independent heads;
- separate Q/K readouts over a shared-head positional curve;
- separate Q/K readouts over a jointly mapped all-head curve.

The completed shared-trunk/separate-readout, independent-head run remains the
anchor (`eval_loss=4.3980`) and is not rerun. All four new configs passed CPU
dry runs and were submitted through `gpu-claim`; three immediately acquired
free GPUs and one is waiting behind active claims. WandB group is
`phase3-coupling-transfer-5k`.

### 5k canonical coupling-transfer result

All four runs completed cleanly. Against the reused `4.3980` anchor:

- fully shared Q/K: `4.3984`;
- fully separate Q/K: `4.3989`;
- separate readouts over a shared-head curve: `4.4116`;
- separate readouts over a joint-head curve: `4.3900`.

Q/K sharing itself has negligible effect at this horizon. Shared-head coupling
loses `0.0136`, while joint-head coupling gains `0.0080`; however, the joint-head
run has 6,531,136 position parameters versus 2,402,368 for the anchor because
its inferred Fourier basis expands from head width to model width. The gain
therefore requires a basis-width and generic-capacity control.

---

## 2026-07-24 — Structural/scalar/mapper follow-up launched

Launched eight targeted seed-`123`/`456`, 5k-step runs under
`sweep_configs/phase3_structural_followup/` and
`model-output/position_bias_phase3_structural_followup/`:

- joint-head coupling with explicit `basis_dim=96`, exactly matching the
  anchor's 2,402,368 position parameters;
- the wide joint-head and efficient shared-Q/K variants at seed `456`;
- the canonical anchor with FFN width `3328` as a generic-capacity control;
- normalized-position-only and log-position-only scalar ablations;
- non-residual rank-32 low-rank and hidden-128 MLP positional mappers.

The existing no-scalar and both-scalar runs provide the remaining scalar
anchors, so they are not repeated. All eight configs passed CPU dry runs and
were submitted through `gpu-claim`; five immediately acquired free GPUs and
three are waiting behind active claims. WandB group is
`phase3-structural-followup-5k`.

---

## 2026-07-25 — Factorized pair-aware logit screen launched

Added `pairwise_low_rank` relative-logit conditioning:

```text
b(i,j) = b_static(i-j)
       + g_h / sqrt(r) · Σ_m Q_m(q_i, φ(i)) K_m(k_j, φ(j)) D_m(φ(i-j))
```

The content and position factors are independently projected and RMS-normalized;
the unconstrained per-head outer gate initializes to zero. This preserves the
completed static linear-logit anchor exactly without using saturating
activations. FlexAttention contracts rank components directly in `score_mod`
without constructing a dense pairwise bias tensor.

Launched three seed-`123`, rank-`8`, 5k-step variants under
`sweep_configs/phase3_pairwise_logit/` and
`model-output/position_bias_phase3_pairwise_logit/`:

- relative-only Fourier offset (`φ(i-j)`), preserving translation invariance;
- query-absolute enrichment (`φ(i-j)` plus `φ(i)`);
- full-absolute enrichment (`φ(i-j)`, `φ(i)`, and `φ(j)`).

Canonical amplitude-`0.3` Q/K, normalized/log-position Q/K scalars, and the
static linear-logit base are held fixed. The completed geometry-transfer anchor
at this seed and horizon is reused (`eval_loss=4.3980`). All three configs passed
CPU dry runs; the full CUDA forward/backward smoke produced finite loss and a
nonzero zero-gate gradient. All 51 position tests pass. The jobs acquired three
GPUs through `gpu-claim` and reached steady training at roughly 2.25 steps/s.
WandB group is `phase3-pairwise-logit-5k`.

### 5k factorized pair-aware logit result

All three runs completed cleanly. Against the reused canonical static-logit
anchor at `4.3980`:

- relative-only: `4.3986`;
- query-absolute: `4.3917`;
- full-absolute: `4.3964`.

The relative-only interaction is a null result and full-absolute enrichment
provides only `0.0016`. Query-absolute is the screen winner by `0.0063` against
the anchor and merits another-seed confirmation, but the gap remains too small
for a single-seed conclusion.

---

## 2026-07-25 — AddRoPE component and residual-stream screens launched

Added independent `learn_amplitude` and `learn_phase` controls for canonical
AddRoPE. Disabled heads are not instantiated; disabling both produces a
parameter-free fixed Fourier Q/K carrier. Added top-level `use_rope=false`,
including a valid no-explicit-PE path and residual-only position models while
continuing to reject rotary Q/K channels without RoPE.

Launched three new seed-`123`, 5k-step AddRoPE component runs under
`sweep_configs/phase3_addrope_components/`:

- fixed amplitude-`0.3` Fourier carrier;
- learned amplitude with fixed phase;
- fixed amplitude with learned phase.

The completed learned-amplitude + learned-phase canonical run (`4.3980`) is the
fourth cell of the 2x2 design and is reused. Q/K scalar inputs and static linear
logit bias otherwise remain fixed.

Launched seven seed-`123`, 5k-step residual-sector runs under
`sweep_configs/phase3_residual_sector/`:

- standard RoPE control;
- no explicit positional encoding;
- sinusoidal residual input;
- learned absolute residual input;
- linear Fourier residual input;
- MLP Fourier residual input;
- zero-gated, layer-specific linear-Fourier reinjection.

Q/K and logit position channels are disabled throughout this sector; all but the
RoPE control use `use_rope=false`. The ten new configs passed CPU dry runs and
all 55 position tests pass. Eight runs immediately acquired GPUs through
`gpu-claim`; the two remaining residual runs are queued behind lifetime claims.
WandB groups are `phase3-addrope-components-5k` and
`phase3-residual-sector-5k`.

The fixed-carrier run reached step 500, then its diagnostics attempted to read a
basis module intentionally absent from the parameter-free carrier. Training was
finite (`eval_loss=5.7130`); this was a reporting-only failure. Fixed-carrier
diagnostics now derive frequencies directly from `rope_theta`, with a regression
test, and only that run was requeued from scratch through `gpu-claim`.

### 5k AddRoPE-component and residual-sector results

All corrected runs completed. AddRoPE component isolation, against the reused
combined amplitude+phase anchor at `4.3980`:

- learned amplitude only: `4.4015`;
- learned phase only: `4.4021`;
- fixed amplitude-`0.3` Fourier carrier: `4.4154`.

Amplitude-only and phase-only are indistinguishable at this resolution. Each
recovers most of the combined mechanism's gain over the fixed carrier, while
the combined model remains marginally best. The canonical stack therefore
retains both controls; further component ranking is shelved.

Residual-stream sole-mechanism results:

- RoPE control: `4.5166`;
- sinusoidal input: `4.7284`;
- linear Fourier input: `4.7896`;
- MLP Fourier input: `4.8028`;
- per-layer reinjection: `4.8598`;
- learned absolute input: `4.8930`;
- no explicit position: `4.9309`.

Residual position helps over no explicit PE, but even the strongest residual
variant remains `0.2118` behind RoPE. Residual-only PE is not promoted and is
not combined with the canonical attention-local stack.

---

## 2026-07-25 — Canonical stack frozen for final promotion

The active research default is now:

- canonical additive amplitude+phase AddRoPE, `amplitude_init=0.3`;
- frozen Fourier Q/K input plus normalized- and log-position scalars;
- linear Q/K mapper with shared trunk and separate Q/K readouts;
- per-head-independent position maps;
- static linear relative-logit bias under FlexAttention.

The existing 10k two-seed result established canonical AddRoPE plus linear
relative logits (`4.0508` mean versus RoPE `4.1417`). The two-seed 5k basis
screen independently established scalar augmentation (`0.0086` mean gain).
The next promotion isolates their union at 10k without changing coupling,
mapper, amplitude, or attention implementation.

The following axes are frozen from active expansion: learned rotary
phase/projected/scaled variants, learned Fourier temperatures/frequencies,
nonlinear and low-rank mapper searches, further Q/K/head-coupling sweeps,
residual-only position, attention-output writes, Inkling profile banks, direct
content-conditioned Q/K, and amplitude/component ranking. Additive
pair-normalized geometry, query-position writes, and learned base-RoPE
frequencies remain backlog rather than active experiments.

### Final-decision bundle

Launched the three pre-registered decision runs through `gpu-claim`:

- scalar-augmented canonical AddRoPE + static linear logits at 10k, seeds
  `123` and `456`;
- rank-8 query-absolute pairwise logits at 5k, seed `456`.

The 10k runs compare with the no-scalar canonical mean `4.0508`; the pairwise
run compares with the exact scalar/static seed-`456` anchor at `4.3895`.
Pairwise conditioning remains excluded from the 10k stack pending replication.

The scalar-promoted 10k runs completed at `4.0452` (seed `123`) and `4.0488`
(seed `456`), mean `4.0470`. This is a `0.0038` mean improvement over the
no-scalar canonical mean `4.0508`, below the pre-registered `~0.007` attribution
threshold. At 5k the existing FFN-width capacity control reached `4.3911`,
outperforming both the no-scalar (`4.4066`) and scalar (`4.3980`) seed-`123`
anchors. Scalars remain part of the consolidated model for the frozen
extrapolation comparison, but their small gain is not position-specific evidence.

The seed-`456` query-absolute pairwise confirmation finished at `4.3998`, which
is `0.0103` worse than its exact scalar/static anchor (`4.3895`). Combined with
the seed-`123` improvement of only `0.0063`, the two-seed result does not
replicate. Pairwise content conditioning is not promoted, and the gated
position-only query/offset surface is intentionally not implemented.

---

## 2026-07-25 — Length-extrapolation evaluation added

Training configuration now separates `training_length`,
`model_position_extent`, and `evaluation_lengths`. Longer validation examples
are contiguous rechunks of the existing tokenized validation stream, and one
evaluation record reports the in-distribution loss plus length-qualified losses.
A one-step CUDA smoke evaluated 1024, 1536, and 2048-token rows successfully.

The frozen comparison is limited to RoPE, free additive Q/K plus static linear
relative logits, and scalar-augmented canonical AddRoPE plus static linear
relative logits. Each will train at 1024 with position extent 2048 and be
evaluated at 1024, 1536, and 2048.

### Compact Q/K basis result

Against the native-width `basis_dim=96` scalar canonical anchor at `4.3980`:

- `basis_dim=16`: `4.4037`;
- `basis_dim=32`: `4.4040`;
- `basis_dim=64`: `4.4142`.

All compact cells remain close, and width 16 is the best compact result despite
being the smallest. It is the efficiency default for future exploratory work;
the native-width model remains the conservative quality default because this
screen is single-seed and retains a `0.0057` loss gap.

### Frozen-finalist extrapolation result

Final evaluation losses after training on length 1024 with position extent 2048:

- scalar canonical, initially normalized to model extent 2048:
  `4.0451` / `4.0378` / `4.0840` at
  1024 / 1536 / 2048;
- free additive + linear logit: `4.0607` / `4.0550` / `4.1040`;
- RoPE: `4.1284` / `4.1121` / `4.1514`.

The scalar canonical model is best at every evaluated length. Its 1024-to-2048
change is `+0.0388`, versus `+0.0434` for free additive and `+0.0230` for RoPE:
RoPE degrades least in relative terms, but its absolute 2048 loss remains
`0.0675` worse than canonical. The 1536 losses are slightly lower for every
model, so only the common-model comparisons and the 2048 tail should be treated
as extrapolation evidence rather than assuming monotonic loss with context.

The initial scalar run also exposed an important semantic distinction: scalar
features were normalized by the allocated model extent, making their training
range roughly `[0, 0.5]`. Added independent `scalar_normalization_extent`,
defaulting to training length, so positions beyond the training horizon produce
genuinely out-of-range scalar values. RoPE and free-additive results are
unaffected; only the canonical scalar finalist is rerun with normalization
extent 1024 before closing this comparison.

The corrected scalar-normalization run completed at `4.0444` / `4.0383` /
`4.0871` for 1024 / 1536 / 2048. It is effectively unchanged in-distribution
and remains best at all lengths; at 2048 it beats free additive by `0.0169` and
RoPE by `0.0643`. Its 1024-to-2048 change is `+0.0427`. These corrected values,
not the model-extent-normalized preliminary cell, close the frozen-finalist
comparison.

---

## 2026-07-25 — Post-position Q/K normalization screen launched

Added opt-in `post_position_qk_norm`, a parameter-free per-head RMS
normalization applied after all additive and rotary Q/K position operations.
The existing learned QK LayerNorm remains before position injection, giving the
controlled sequence `project -> LayerNorm -> position operation -> RMSNorm`.
RMSNorm was chosen for the second stage because it repairs magnitude without
LayerNorm mean subtraction changing positional direction.

Launched six seed-`123`, 5k-step runs:

- canonical scalar AddRoPE plus post-normalization;
- scalar free-additive control, with and without post-normalization;
- bounded local-residual conditioning plus post-normalization;
- bounded content-gate conditioning plus post-normalization;
- scaled rotary plus post-normalization.

Existing exact 5k cells are reused for the canonical, conditioning, and scaled
rotary comparisons. The free-additive pair is run together because no exact
scalar-augmented 5k control existed. All 60 position tests pass, all six configs
pass GPU dry runs, and all six jobs acquired GPUs through the shared
`gpu-claim` queue. WandB group is `phase4-post-qk-norm-5k`.

The completed post-normalization results were `4.3869` for canonical AddRoPE,
`4.4120 -> 4.4007` for the exact free-additive control pair, and `4.4732` for
scaled rotary versus its prior `4.4617` anchor. Local-residual and content-gate
conditioning remained collapsed at `5.9242` and `5.4448`. Post-normalization
kept attention finite but exposed position branches with RMS values in the
hundreds and maxima above 3,000: it fixed total Q/K magnitude without preserving
the content/position mixture.

Further review found that the bounded retries constrained only conditioner
corrections, not the underlying amplitude base. Conditioned diagnostics also
used zero content rather than a real validation example. These runs reject the
old parameterizations, not content conditioning in general.

---

## 2026-07-25 — Safe contribution and residual-content redesign launched

Added the following controls:

- positive `bounded_sigmoid` pair amplitudes;
- per-token/head RMS normalization of additive position branches followed by a
  bounded learned contribution gain;
- bounded-log rotary pair scales;
- `scaled_sigmoid` content gates and selectable `tanh`/`GELU`/linear local
  corrections;
- conditioning sourced directly from the block-normalized residual stream,
  with independent Q/K conditioner networks;
- exact unit-pair rotary residuals formed by normalizing
  `[cos(theta)+dx, sin(theta)+dy]`;
- real-validation-content diagnostics for branch/QK RMS ratios, p95 ratios,
  content-to-combined angles, and additive gains.

Also fixed `mapper.kind=linear, residual=true`, whose residual flag was
previously validated but not applied.

Launched eight seed-`123`, 5k-step runs under
`phase4-safe-conditioning-5k`: a safe unconditioned control; residual- and
QK-sourced sigmoid gates; residual-sourced tanh gate; residual-sourced GELU and
tanh local corrections; bounded scaled rotary; and unit-pair rotary. All 66
position tests pass, every config passes GPU dry-run validation, and CUDA
forward/backward smoke tests produced finite losses and real-content
diagnostics. Seven jobs acquired the currently free GPUs through `gpu-claim`;
the unit-pair rotary job is waiting behind the active external claim.

### Safe-conditioning result

All eight runs completed finitely:

- safe unconditioned AddRoPE: `4.4052`;
- Q/K-sourced sigmoid gate: `4.7263`;
- residual-sourced sigmoid gate: `4.9580`;
- residual-sourced GELU local correction: `4.9584`;
- residual-sourced tanh local correction: `4.9841`;
- residual-sourced tanh gate: `5.0100`;
- bounded scaled rotary: `4.4781`;
- unit-pair rotary: `4.4948`.

The safety controls worked mechanically: additive branch RMS maxima remained
`0.220–0.238`, p95 branch/content ratios remained below `0.259`, and the
minimum content-to-combined cosine remained above `0.973`. Thus the old
hundreds-RMS takeover is gone. Content conditioning nevertheless remains
substantially worse than the safe control. Q/K content beats residual content
by `0.2317` in the sigmoid-gate comparison; GELU beats tanh by `0.0257` for the
residual local correction, and sigmoid beats tanh by `0.0520` for the residual
gate. These activation effects are real within the screen but do not rescue
the mechanism.

The safe control is `0.0183` worse than the earlier unconstrained post-RMS
canonical result (`4.3869`), suggesting the full branch normalization/bounded
amplitude package also removes some useful freedom. Bounded scaled rotary and
unit-pair rotary do not improve the rotary sector.

Added `position_results.py` to make these comparisons repeatable. It selects
runs by glob/regex, filters arbitrary step intervals, emits summary or history
rows, provides core/QK-health/all post-processing presets, accepts extra metric
globs, and renders table, Markdown, CSV, JSON, or JSONL. Three focused tests
cover discovery, duplicate-step handling, interval filtering, health
aggregation, and machine-readable output.

---

## 2026-07-25 — Additive geometry cleanup launched

Implemented the final two additive geometry ideas:

- **true free residual:** the corrected linear mapper now computes
  `z(p) + Linear(z(p))`;
- **pair normalized:** an arbitrary split-half Cartesian output is normalized
  pairwise and multiplied by fixed radius `amplitude_init` before Q/K
  injection.

The screen contains three seed-`123`, 10k-step runs: free residual, pair
normalized at radius `0.3`, and pair normalized at radius `1.0`. All retain the
promotion screen's frozen Fourier basis, shared trunk with separate Q/K
readouts, per-head-independent maps, and static linear relative-logit bias.
Each has exactly `2,390,080` position parameters, matching the completed
free-direct anchor (`4.0615`). The existing canonical AddRoPE anchors are
`4.0525` at amplitude `0.3` and `4.0515` at amplitude `1.0`.

All 68 position tests pass. Focused eager and compiled CUDA forward/backward
smokes are finite for both new geometries. The three jobs acquired GPUs through
the shared lifetime `gpu-claim` queue under WandB group
`phase4-additive-geometry-10k`.

### Additive geometry cleanup result

At seed `123` and 10k steps:

- pair-normalized radius `0.3`: `4.0524`;
- pair-normalized radius `1.0`: `4.0618`;
- free residual: `4.0747`.

The radius-`0.3` model essentially ties canonical AddRoPE (`4.0525`) and beats
the exact free-direct anchor (`4.0615`) by `0.0091`. Radius `1.0` is neutral,
while the corrected residual skip regresses by `0.0132`. Real-content
diagnostics align with the ordering: the p95 addend/content ratio is `0.263`
at radius `0.3`, `0.924` at radius `1.0`, and `1.252` for free residual.

---

## 2026-07-26 — Geometry-preserving content phase screen launched

Added `conditioning.kind=phase_rotation` for additive `pair_normalized`
carriers. A local conditioner receives normalized Q/K or block-normalized
residual content and predicts only a bounded angular correction:

```text
delta = phase_bound * tanh(C(content))
e' = R(delta)e
```

It cannot alter pair radius or synthesize arbitrary Cartesian addends.
`target` selects Q, K, or both. `coupling` selects one shared output head or a
shared content trunk with separate zero-initialized Q/K phase readouts. The
zero initialization is exactly the unconditioned carrier and has live output
gradients.

Launched four seed-`123`, 10k-step variants over the radius-`0.3`
pair-normalized anchor: Q-only, K-only, both with a shared readout, and both
with separate readouts. Every conditioner uses block-normalized residual
content, no position input, hidden width `32`, and phase bound `0.25` radians.
The completed unconditioned anchor is `4.0524`.

All 71 position tests pass, all configs pass full-model dry runs, and eager
plus compiled CUDA forward/backward smoke tests are finite. All jobs acquired
GPUs through `gpu-claim`; WandB group is
`phase4-phase-conditioning-10k`.

---

## 2026-07-27 — Null-initialized dedicated-content screen launched

Refactored positional content conditioning around independent 64-dimensional
projections of the block-normalized residual. New Q/K conditioning and
content-aware relative logits no longer reuse projected attention Q/K. The
content projection may be shared or separate across Q/K and is RMS-normalized
before per-head actuator trunks.

Added `qk_norm_mode=method_aware_rms`, which applies exactly one per-head
RMSNorm:

```text
additive: project -> add carrier -> RMSNorm -> optional content gain
rotary:   project -> RMSNorm -> R(theta + content_delta)
```

Three explicit anchor-relative content actuators now use live trunks with
zero-initialized final projections and no second zero gate:

- `adaptive_gain`: `gain=exp(raw)`, initially exactly one;
- `additive_phase`: content rotates the established
  `a*cis(theta)` additive carrier, initially with zero phase delta;
- `rope_phase`: content modifies the actual RoPE rotation, initially standard
  RoPE.

Phase deltas are linear signed outputs rather than `tanh`-bounded values.
Canonical additive Fourier/AddRoPE remains
`q + a*cis(theta+delta)` and does not rotate Q itself.

Added a separate `attention_write.mode=query_position`:

```text
out_i = O(attn_i) + W_zero(position_i)
```

This mode does not append positional values to V. Its final projection starts
at zero and receives a live first-step gradient without a scalar gate.

The one-seed 5k attribution screen contains:

- method-aware RMS RoPE and fixed radius-`0.3` additive-Fourier anchors;
- adaptive gain, additive-carrier phase, and RoPE phase with shared versus
  separate Q/K readouts;
- the prior query-absolute pairwise relative-logit candidate rerun with
  dedicated content;
- the query-local position output write.

All ten configs passed full-model dry runs. The 51 position-playground tests
and result-collector tests pass. Eager and compiled CUDA forward/backward
smokes are finite for every new path. The first compiled dedicated-pairwise
smoke exposed a zero-stride expanded-content backward failure; materializing
the head-expanded low-rank content with `contiguous()` fixed it, and the
focused compiled retry passed.

The suite launched under supervisor and the shared lifetime `gpu-claim`
protocol as `mlprope_phase5_null_conditioning`. WandB group is
`phase5-null-conditioning-5k`; configs and outputs live under
`sweep_configs/phase5_null_conditioning/` and
`model-output/position_bias_phase5_null_conditioning/`.

## 2026-07-27 — Phase-5 results and breadth-before-scale freeze

All Phase-5 runs completed at 5k steps, seed 123. Two static anchors initially
failed because channels with `conditioning.kind=none` inherited
`source=dedicated` and requested a content projector they did not need. The
attention guard was corrected, regression-tested, and both anchors completed
on rerun.

Final losses:

| Variant | Eval loss |
| --- | ---: |
| additive phase, separate readouts | **4.3952** |
| additive phase, shared readout | **4.3959** |
| fixed radius-0.3 additive anchor + linear logit | 4.4098 |
| RoPE + method-aware RMS + linear logit | 4.4497 |
| adaptive gain, shared / separate | 4.4587 / 4.4590 |
| dedicated pairwise query-absolute logit | 4.4689 |
| true RoPE phase, separate / shared | 4.4792 / 4.4802 |
| query-local position write | 4.4994 |
| standard RoPE | 4.5165 |
| RoPE + method-aware RMS, no logit | 4.5210 |

Method-aware RMS alone was neutral (`4.5165 -> 4.5210`). The static linear
logit remained strongly useful (`4.5210 -> 4.4497`), and the fixed additive
carrier improved it again (`4.4497 -> 4.4098`). Dedicated-content additive
phase was the only new actuator to improve its matched anchor
(`4.4098 -> 4.3952`), while shared and separate readouts were effectively tied.
Its pair radius remained fixed: addend RMS `0.212`, maximum magnitude `0.3`,
and content-to-combined cosine about `0.973`. Phase RMS opened to roughly
`0.12–0.36` radians by layer without destabilizing Q/K.

The Phase-5 stack is not directly comparable to the promoted Phase-3 stack:
it used method-aware RMS, no scalar inputs, and a fixed identity-mapped
carrier. The next work therefore reconciles geometry and normalization on one
full scalar+linear-logit stack before promoting content phase.

The active portfolio is frozen as follows:

- active: canonical AddRoPE, radius-0.3 pair normalization, static linear
  logit, method-aware normalization, dedicated additive phase, and compact
  efficiency controls;
- controls only: RoPE, linear-logit-only, free additive, and wider FFNs;
- retired from active search: residual-stream PE, all attention writes,
  Inkling/pairwise content logits, direct Q/K residual/gate conditioning,
  rotary phase/scale alternatives, learned Fourier frequencies/temperature,
  and broad mapper/coupling sweeps.

Retirement removes mechanisms from future sweeps but keeps their code and
historical configs. Historical implementation failures remain explicitly
separate from clean negative hypothesis tests.

## 2026-07-27 — Phase-6 geometry and normalization reconciliation

Six seed-123 5k runs placed canonical amplitude+phase and radius-0.3
pair-normalized additive geometry on the same scalar-augmented, static
linear-logit stack:

| Variant | Eval loss |
| --- | ---: |
| canonical + method-aware RMS | **4.3851** |
| pair-normalized + method-aware RMS | 4.3971 |
| canonical + method-aware RMS + branch RMS | 4.3980 |
| canonical + legacy LayerNorm | 4.3984 |
| pair-normalized + legacy LayerNorm | 4.4046 |
| RoPE + linear logit + parameter-matched FFN-3168 | 4.4867 |

Canonical geometry with one method-aware RMS after the additive carrier is the
reconciled winner. It improves the exact legacy stack by `0.0133` and the
matched pair-normalized stack by `0.0120`. Separately normalizing the additive
branch erases that gain, so branch RMS is retired. Pair normalization remains
a sound geometry but is dominated on the full scalar stack. Generic FFN
capacity does not explain the Q/K positional gain.

## 2026-07-27 — Full-stack content-phase transfer stopped

The dedicated-content additive-phase actuator was transferred to the
reconciled canonical+scalar+linear-logit+method-aware-RMS stack with Q-only,
K-only, shared-both, and separate-both targets. All four runs showed the same
severe failure by step 3000 and were stopped:

- eval loss remained `5.13–5.24` versus the `4.3851` anchor;
- positional addend RMS reached `195–199`;
- addend maxima reached roughly `2,500`;
- p95 addend/content ratios exceeded `500`.

The content-predicted angular deltas themselves remained modest (usually
`0.01–0.08` radians RMS). The runaway quantity was the promoted carrier's
unbounded learned amplitude. Method-aware Q/K RMS kept final Q/K magnitudes
finite, but it could not prevent the enormous carrier from replacing their
direction. The Phase-5 fixed-radius actuator therefore remains a valid stable
test, while the union of dedicated phase conditioning with the unconstrained
learned amplitude+phase carrier is classified as a broken parameterization.
It is removed from promotion rather than rescued with another bounded-package
sweep.

## 2026-07-27 — Phase-6 compact efficiency screen

Three seed-123 5k efficiency variants used the reconciled canonical,
method-aware-RMS stack. The full-width/full-linear anchor is `4.3851` with
`2,402,368` positional parameters.

| Variant | Eval loss | Position parameters |
| --- | ---: | ---: |
| low-rank rank-32 linear logit | **4.3868** | 2,207,808 |
| Q/K Fourier basis 16 | 4.3932 | 1,910,848 |
| Q/K Fourier basis 32 | 4.3984 | 2,009,152 |

The corrected rank-32 logit is tied with the full linear channel while saving
`194,560` parameters and is promoted as the efficiency alternative. Basis 16
is within the predeclared `0.01` tie band while saving `491,520` parameters, so
it remains a compression option rather than a quality winner. Basis 32 is
strictly dominated by basis 16 (more parameters and worse loss) and is pruned.

## 2026-07-27 — h768/d8 10k promotion and extrapolation gate

The two promoted stacks and two controls trained for 10k steps and were
evaluated at 1024, 2048, and 4096 tokens:

| Variant | 1024 | 2048 | 4096 |
| --- | ---: | ---: | ---: |
| canonical + full linear logit | **4.0345** | 4.0634 | 4.1516 |
| canonical + rank-32 logit | 4.0462 | **4.0532** | **4.0556** |
| RoPE + linear logit + matched FFN | 4.0855 | 4.0964 | 4.0894 |
| standard RoPE | 4.1282 | 4.1515 | 4.2498 |

The full linear logit is the in-distribution quality winner. The low-rank
logit trails by `0.0117` at training length but generalizes much better:
its 4096 loss is essentially flat and beats the full channel by `0.0960`.
Both therefore pass the scale gate for different reasons. Standard RoPE and
the matched-FFN model remain controls; neither explains the positional-stack
gain.

## 2026-07-27 — h1024/d12 larger-model scale gate

The first h1024/d12 attempt used batch size 4 and was intentionally stopped
after its early metrics because that halved tokens per optimizer step relative
to h768/d8. Those stopped WandB attempts are invalid scale comparisons, not
model crashes. Their local partial metrics were removed before relaunch.

The corrected runs used batch size 8, 10k optimizer steps, and the same
1024-token training sequences. All four completed without OOM, traceback, NaN,
or queue failure:

| Variant | 1024 | 2048 | 4096 |
| --- | ---: | ---: | ---: |
| canonical + full linear logit | **3.9523** | **3.9566** | 3.9693 |
| canonical + rank-32 logit | 3.9545 | 3.9591 | **3.9601** |
| RoPE + linear logit + matched FFN | 4.0094 | 4.0281 | 4.0303 |
| standard RoPE | 4.0464 | 4.0836 | 4.2328 |

The two promoted stacks are tied in-distribution at larger scale (`0.0022`
apart). The rank-32 logit again has the better 4096 result, while the full
linear channel is marginally best at 1024 and 2048. Both beat the
parameter-matched FFN control by about `0.055–0.070`, confirming that the gain
is positional structure rather than generic parameter capacity.

The breadth-before-scale portfolio is therefore closed with two supported
endpoints:

- **quality default:** canonical scalar AddRoPE + method-aware RMS + full
  linear relative logit;
- **extrapolation/efficiency default:** the same Q/K stack with a corrected
  rank-32 linear relative logit.

## 2026-07-27 — 50k physical-batch and compile probe

The h1024/d12 full-linear candidate was profiled for 200 optimizer steps at
effective batch 32. Steady-state measurements exclude the first 20 compile and
allocator warmup steps:

| Physical batch / accumulation | Checkpointing | Compile mode | Tokens/s | Peak allocated | Peak reserved |
| --- | --- | --- | ---: | ---: | ---: |
| 32 / 1 | on | default | 31,210 | 23,710 MiB | 28,190 MiB |
| 32 / 1 | on | max-autotune-no-cudagraphs | 30,844 | 23,710 MiB | 27,770 MiB |
| 16 / 2 | on | default | 31,001 | 14,889 MiB | 17,180 MiB |
| **8 / 4** | **off** | **default** | **46,836** | **15,497 MiB** | **16,476 MiB** |

Physical batch 8 without activation checkpointing is about 50% faster than the
checkpointed configurations while retaining roughly 16 GiB of reservation
headroom. `reduce-overhead` was rejected because its CUDA-graph replay
overwrote a live compiled output tensor. The production choice is therefore
physical batch 8, accumulation 4, checkpointing off, and compile mode
`default`. This keeps effective batch 32 and processes 1,638,400,000 nominal
training tokens over 50,000 optimizer steps.

## 2026-07-27 — h1024/d12 50k promotion result

All six seed-123 runs completed 50,000 optimizer steps at sequence length 1024
and effective batch 32: `1,638,400,000` nominal training tokens per run.
Checkpoints were written every 5,000 steps, final weights were saved, every
queue child exited with code 0, and there were no OOMs, NaNs, tracebacks, or
checkpoint resumes.

| Variant | 1024 | 2048 | 4096 | Tokens/s |
| --- | ---: | ---: | ---: | ---: |
| **compact basis-16 AddRoPE + full logit** | **2.9835** | **3.0456** | 3.1142 | 47,298 |
| full-basis AddRoPE + full logit | 2.9839 | 3.0704 | 3.1855 | 47,463 |
| full-basis AddRoPE + rank-32 logit | 2.9875 | 3.1588 | 3.3595 | 46,950 |
| RoPE + full logit + matched FFN-4160 | 3.0048 | 3.0458 | **3.0578** | 47,214 |
| RoPE + full logit | 3.0132 | 3.0551 | 3.0658 | 48,017 |
| standard RoPE | 3.0337 | 3.0864 | 3.1489 | **90,299** |

This reverses two provisional 10k decisions:

- Basis 16 is no longer merely a compression option. It ties the full Q/K
  basis at training length, beats it by `0.0247` at 2048 and `0.0713` at 4096,
  uses about 590k fewer positional parameters, and has healthier Q/K
  diagnostics (`2.12` maximum addend RMS versus `2.72`). It is the practical
  additive quality/default endpoint.
- The rank-32 logit's earlier extrapolation advantage does not survive 50k.
  Its 4096 loss degrades to `3.3595`, worst in the pack, while no longer saving
  parameters at h1024. It is pruned from promotion.

RoPE + full relative logit is the strongest same-FFN extrapolation endpoint
(`3.0658` at 4096). The FFN-4160 control improves this to `3.0578`, a small
capacity effect, but remains a control rather than a distinct positional
method. Standard RoPE remains the throughput endpoint: fused SDPA reaches
about `1.88–1.92x` the throughput of all FlexAttention/logit-bias arms. The
extra cost is overwhelmingly the FlexAttention relative-logit path rather
than the additive Q/K carrier.

The final portfolio is therefore:

- **additive quality/default:** compact basis-16 canonical scalar AddRoPE +
  method-aware RMS + full linear relative logit;
- **extrapolation positional default:** RoPE + full linear relative logit;
- **throughput/control default:** standard RoPE with fused SDPA;
- **pruned:** rank-32 logit and full Q/K basis, both dominated at this horizon.

## 2026-07-27 — SDPA carrier-hypernetwork integration smoke

The anchor-relative carrier hypernetwork passed eager and compiled CUDA
forward/backward tests for additive AddRoPE and scaled rotary, then completed
three h768/d8 SDPA-only, seed-123 integration runs for 200 optimizer steps.
Every queue child exited with code 0; there were no OOMs, NaNs, non-finite
gradients, or tracebacks.

| Variant | Eval loss | Tokens/s | Peak reserved | Gain max | Phase p95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| compact basis-16 AddRoPE anchor | 6.4063 | 26,528 | 5,382 MiB | — | — |
| AddRoPE content+position SiLU hypernetwork | 6.5258 | 41,294 | 6,162 MiB | 1.173 | 0.0343 |
| scaled-RoPE content-linear hypernetwork | 6.2657 | 51,771 | 5,496 MiB | 2.431 | 0.1994 |

These short warmup-dominated runs validate execution and metric plumbing only;
they are not a ranking result. The additive hypernetwork remained tightly
bounded at step 200 (`delta_log_gain` p95 `0.0347`, phase p95 `0.0343`).
The rotary arm moved more aggressively (`delta_log_gain` p95 `0.447`,
effective gain max `2.43`), but remained finite. Longer screens should monitor
that radial scale before promotion.

## 2026-07-28 — AddRoPE gauge correction

The 5k carrier-hypernetwork screen exposed a major additive confound. The
generalized AddRoPE mapper learned position-dependent amplitude/phase while a
second hypernetwork learned multiplicative log-gain and phase deltas. Those
branches have an amplitude/phase gauge, amplified by post-addition Q/K RMS:
effective gain maxima reached `122–132x` and one addend RMS reached `519`.

The v2 channel now distinguishes:

- `output.parameter_source=direct`: canonical per-head/per-frequency AddRoPE
  parameters with no position mapper;
- mapped amplitude/phase: the existing generalized position-functional model;
- `conditioning.components=amplitude_phase`: a gauge-free dynamic replacement
  requiring static amplitude/phase learning to be disabled and using direct
  softplus amplitude around `amplitude_init`.

Zero hypernetwork heads recover exactly
`amplitude_init * cis(omega * position)`. Unit tests verify exact anchors,
separate Q/K gradients, absence of the static mapper, and schema rejection of
mixed static/dynamic controls. Eager and compiled CUDA forward/backward smokes
passed for direct canonical and dynamic softplus AddRoPE.

A six-arm seed-123, 5k SDPA screen is prepared under
`phase8_addrope_clean`: standard RoPE, the generalized mapped anchor, direct
canonical AddRoPE, fixed AddRoPE, content-only dynamic replacement, and
content+position dynamic replacement.

## 2026-07-28 — Unit-anchor HyperAddRoPE screen

The gauge-free dynamic AddRoPE path now also accepts signed amplitude. With
`amplitude_init=1`, disabled static amplitude/phase learning, and zeroed
hypernetwork readouts, its exact initial carrier is:

```text
(1 + predicted_scale) * cis(omega*p + predicted_phase)
```

Both predictions begin at zero. This preserves the old mapped, softplus, and
log-gain paths while providing a raw-polar unit anchor with no learned static
amplitude/phase gauge.

The `phase9_unit_hyper` family contains ten seed-123, 5k-step SDPA cells:
standard RoPE; mapped-0.3 AddRoPE; direct unit AddRoPE; position, content, and
content+position inputs crossed with linear and SiLU trunks; and one
content+position SwiGLU arm. Hypernetwork cells fix per-head outputs, one shared
normalized content projection, and shared-trunk/separate-QK readouts. Every
configuration passed `train_gpt.py --dry_run`; direct unit and
content+position-SiLU variants passed eager and compiled CUDA
forward/backward smoke tests.

The completed screen promoted the content+position SiLU unit hypernetwork
(`4.2899`), with content+position linear and content-only SiLU within the
predeclared `0.01` tie threshold. Six of seven hypernetwork cells beat the
mapped-0.3 AddRoPE control (`4.3403`); direct unit AddRoPE was worst (`4.4419`),
showing that dynamic conditioning rather than the unit carrier itself supplied
the gain.

A six-cell `phase9_carrier_followup` screen now isolates the next content-aware
axes: shared versus separate Q/K content projections, dynamic-Q/static-K,
static-Q/dynamic-K, and phase-only HyperRoPE with shared versus separate
content projections. “Static” means directly learned canonical AddRoPE on the
inactive branch, while the active branch has only the dynamic hypernetwork.
The mixed branch implementation and phase-only rotary analogue passed eager
and compiled CUDA forward/backward tests.

The carrier follow-up retained symmetric HyperAddRoPE with shared content:

- dynamic-both/shared-content: `4.2922`;
- dynamic-both/separate-content: `4.3021`;
- static-Q/dynamic-K: `4.3305`;
- dynamic-Q/static-K: `4.3499`;
- phase-only HyperRoPE: `4.4231–4.4345`.

The h768/d8 30k gate confirmed the mechanism at training length. Content+
position SiLU reached `3.5710`, content+position linear `3.5732`, mapped
AddRoPE `3.5855`, and standard RoPE `3.6272`. Checkpoint saving is disabled by
default for subsequent screens, and evaluation defaults to the 1024 training
length unless extrapolation is explicitly requested.

A matched 2x2 Q/K independence screen then separated content-projection sharing
from hypernetwork-trunk sharing. Shared content plus a shared trunk led at
`4.2878`; separate content/shared trunk reached `4.3002`, shared
content/separate trunks `4.3040`, and fully separate conditioning `4.3062`.
The learning curves showed no late convergence trend by step 5k. The active
default is therefore one normalized content projection and one shared trunk,
with distinct Q/K final readouts.

The next `phase9_hyper_capacity` screen keeps that default as its control and
tests: one shared Q/K readout, one conditioner shared across heads, content
dimension 128, SiLU trunk widths 128 and 256, and content-128 combinations with
both wider trunks. All cells remain 5k, SDPA, 1024-only, and checkpoint-free.

The capacity screen completed under Supervisor after two terminal-managed
launchers were externally terminated. Sharing the final Q/K readout (`4.3135`)
or sharing one conditioner across heads (`4.3065`) was clearly worse than the
separate-readout/per-head control (`4.2905`). Wider conditioner cells were all
within the `0.01` tie threshold:

- content-128/trunk-256: `4.2810`;
- content-64/trunk-256: `4.2835`;
- content-128/trunk-64: `4.2843`;
- content-128/trunk-128: `4.2890`;
- content-64/trunk-128: `4.2897`.

The apparent capacity gain is not free. At h768/d8, content-128/trunk-256 uses
`6.35M` positional parameters versus `1.53M` for content-64/trunk-64 and
`1.31M` for mapped AddRoPE. Its sub-`0.01` advantage therefore requires a
parameter-matched wider-FFN control before promotion. The structural default
remains shared content and trunk with per-head conditioning and separate Q/K
readouts.

## 2026-07-30 — Phase-10 HyperAddRoPE normalization/output-geometry screen launched

The `phase10_hyper_geometry` family tests normalization and carrier geometry
around the shared-content/shared-trunk, per-head, separate-Q/K-readout
HyperAddRoPE default. All HyperAddRoPE cells use `content_dim=128`, trunk width
64, SDPA, method-aware add-then-RMS Q/K normalization, seed 123, 5,000 steps,
and 1024-only evaluation. Checkpoint and final-model saving remain disabled.

The 11 cells are standard RoPE, mapped-0.3 AddRoPE, the signed-polar
HyperAddRoPE control, modality-wise input RMS with and without learned
content/position scalar gains, exact-one softplus amplitude, Cartesian complex
residual `(1+u)+iv`, dynamic amplitude+phase+frequency, static learned amplitude
with dynamic frequency+phase, amplitude-only, and phase-only. The frequency
arms use the unwrapped float32 composition
`cis((1+delta_frequency)*(omega*p+delta_phase))`. New dynamic output heads are
zero-initialized and recover exactly `cis(omega*p)`; schema validation prevents
a dynamic amplitude or phase from overlapping a learned static parameter for
the same component.

Validation before launch:

- all 98 CPU position tests passed (one expected claimed-CUDA-only skip);
- all 11 h768/d8 configs passed `train_gpt.py --dry_run` through `gpu-claim`;
- modality-RMS/gain, Cartesian, and frequency cases passed eager and compiled
  CUDA forward/backward smoke tests;
- positional parameter counts are 2.187M for matched two-component hypernetwork
  cells, 2.587M for amplitude+phase+frequency, 1.788M for one-component
  isolation cells, 2.193M for static-amplitude+frequency+phase, and 1.309M for
  mapped AddRoPE.

The sweep launched at 2026-07-30 23:13 UTC under Supervisor program
`mlprope-phase10-hyper-geometry`, with output root
`model-output/position_bias_phase10_hyper_geometry/`. Eight cells acquired
lifetime GPU claims immediately and three entered the cooperative queue. W&B
upload was not enabled for this launch; local output metrics are authoritative.

## 2026-07-30 — Phase-10 normalization/output-geometry result

All 11 seed-123 cells completed 5,000 steps without OOM, NaN, non-finite
gradients, traceback, or queue failure. Final 1024-token evaluation losses:

| Variant | Eval loss | Position params |
| --- | ---: | ---: |
| modality RMS + learned modality gains | **4.2816** | 2.187M |
| modality RMS | 4.2831 | 2.187M |
| signed-polar control | 4.2840 | 2.187M |
| Cartesian complex residual | 4.2852 | 2.187M |
| exact-one softplus polar | 4.2872 | 2.187M |
| amplitude-only polar | 4.3024 | 1.788M |
| mapped-0.3 AddRoPE | 4.3396 | 1.309M |
| standard RoPE | 4.4118 | 0 |
| phase-only polar | 4.4348 | 1.788M |
| amplitude+phase+frequency polar | 4.5369 | 2.587M |
| static amplitude + dynamic frequency/phase | 4.7788 | 2.193M |

The five matched two-component geometries are inside a `0.0057` band, well
within the established `0.01` tie threshold. Modality RMS with gains is only
`0.0024` ahead of the signed control at one seed. Its learned content gains
ended in `1.060–1.145` and position gains in `1.042–1.184` across layers, so
the normalization path was healthy rather than inert or runaway. It was also
slower in this screen (about 135k tokens/s versus 149k for signed polar;
RMS without gains was about 143k). This does not justify changing the default.
The project therefore retains raw signed `1+s` polar HyperAddRoPE; modality RMS
remains an optional tied variant, not a promoted mechanism. Softplus and
Cartesian geometry add no quality evidence and are not promoted.

The isolation result is informative: amplitude-only remains strong and beats
mapped AddRoPE by `0.0372`, but trails full amplitude+phase conditioning by
`0.0185`. Phase-only is worse than mapped AddRoPE and standard RoPE. Dynamic
phase is therefore complementary to dynamic amplitude in the joint actuator,
not independently useful in this parameterization.

Both raw frequency-multiplier arms are clean negative results. The fully
dynamic arm trails the signed control by `0.2529`; static amplitude with dynamic
frequency+phase trails by `0.4948`. They remained finite, but learned frequency
multipliers crossed through zero and spread widely (`-0.81–3.37` for the full
arm and `-1.64–3.55` for the static-amplitude arm), allowing local frequency
reversal. The unbounded `1+predicted_frequency` parameterization is pruned.
No bounded-frequency rescue is scheduled without new evidence.

## 2026-07-31 — Zero-GPU probes: logit-curve structure and closed-form fits

Two read-only analyses over saved `position_profiles/step_*.pt` (`[heads, extent]`
relative-bias curves for layers 0 / mid / last). Full derivations and the
retraction below are recorded in `CONCAT_QK_POSITION.md`.

**Concatenated Q/K reformulation.** A relative bias that factors as an inner
product of a query-side and a key-side vector can be folded into the attention
dot product as extra Q/K dimensions, requiring no `score_mod` and running on
fused SDPA. An `R`-frequency cosine series is exactly `2R` extra dims; a free
rank-`r` non-Toeplitz bias is `r` extra dims. The existing `[heads, extent]`
curve is low-parameter but **full-rank**, so the two constraints
(translation-invariance and low rank) are orthogonal rather than nested.

A first probe SVD'd the causal Toeplitz matrix with hard zeros above the
diagonal and reported rank 64–250 for 99% energy. **That bound is invalid** and
is retracted: attention applies the causal mask itself, so entries above the
diagonal are free, and forcing them to zero measures the triangular cutoff
rather than the curve. The valid measurement is DCT truncation error on
`d >= 0`: `R=16` leaves 18–26% of curve range, `R=64` still leaves 9–12%. The
Fourier form is therefore lossy at affordable `R`, not the exact drop-in first
claimed. Materializing an `[H, L, L]` bias for SDPA `attn_mask` was also
rejected: it scales as `H*L^2` (137 GB at h=64, L=32k).

**The learned curves are log-shaped per-head decay profiles.** Sampled values
decay by a roughly constant amount per octave. Pair-frequency-weighted least
squares (weight `L-d`, since distance `d` occurs `L-d` times per sequence):

| Model | Layer | linear (ALiBi) | log | log+tau (3p) | log+linear (3p) |
| --- | --- | ---: | ---: | ---: | ---: |
| h768/d8 | 0 | 0.742 | 0.895 | **0.915** | 0.910 |
| h768/d8 | 4 | 0.651 | 0.793 | 0.872 | **0.889** |
| h768/d8 | 7 | 0.749 | 0.874 | **0.933** | 0.918 |
| h1024/d12 | 0 | 0.184 | 0.529 | 0.528 | **0.688** |
| h1024/d12 | 6 | 0.208 | 0.550 | 0.553 | **0.679** |
| h1024/d12 | 11 | 0.124 | 0.418 | 0.416 | **0.584** |

Linear-in-`d` is the wrong functional family. `tau` grows with depth
(`7.2 -> 12.7 -> 37.9` at h768), so locality relaxes monotonically with depth.
The fit degrades at h1024/d12 (`tau` pins at the search floor), meaning the
larger model uses more of the curve's freedom, not less. Log shapes also
explain the slow DCT convergence: strong curvature at the origin spreads
spectral energy.

Both probes measure *distillation* of a freely learned curve, not what a
rank- or parameter-limited channel could reach when trained from scratch. They
set `R`, they do not veto the arm.

**Reconciliation with `phase10`.** The relative logit bias is a learned per-head
attention decay profile, and `phase10` showed the Q/K carrier gain lives almost
entirely in the amplitude branch. Both of the project's strongest directions
reduce to **per-head control of how fast attention decays with distance**.

## 2026-07-31 — Phase-11 narrow spectral carrier readouts

`phase10` established an ordering by how badly a modulation breaks translation
invariance: amplitude (preserved) `4.3024`, phase (bounded violation) `4.4348`
— worse than standard RoPE `4.4118` — and frequency (violation growing with
absolute position) `4.5369`. Content-dependent `omega` makes the logit depend on
absolute position with drift `m*p`, so the axis is dead for content
conditioning at any parameterization: an `epsilon`-bounded multiplier needs
`epsilon < 1e-4` to keep drift under 0.1 rad at L=1024.

Dividing the multiplier by `p` removes exactly the growing factor and yields a
**new relativity-preserving axis**. With `omega_i = omega*(1 + m_i/i)`,

```text
omega_i*i - omega_j*j = omega*((i + m_i) - (j + m_j))
```

so the mechanism is a content-dependent shift of *effective position*. This is
not the failed phase-only arm: free per-frequency phase decoheres the spectrum,
whereas an offset ties phase to `omega_r` and stays coherent across it.

Two narrow readouts were added to `CarrierHypernetwork`, both anchored at exact
`cis(omega*p)` with zeroed readouts:

- `amplitude_slope` (2 scalars/head): `amplitude = 1 + gain + slope*tilt_r`,
  where `tilt_r` is z-scored `log(omega_r)`. A locality / decay-rate control.
- `position_offset` (1 scalar/head): `phase_r = omega_r * bound * tanh(m)`,
  with `offset_bound` defaulting to 8 tokens.
- `slope_offset` (3 scalars/head) composes both.

Schema additions: `conditioning.components` accepts the three new modes,
`conditioning.offset_bound` is a new positive key, and the new modes are
registered in the additive amplitude/phase overlap checks so dynamic components
still cannot coexist with learned static ones.

Six CPU tests cover readout width, the exact zero-readout anchor and its
content-independence, the identity `phase = omega*m` equals evaluating the
carrier at `p + m`, tilt normalization and monotonicity, distinct Q/K gradients,
and four schema rejections. All 104 CPU tests pass. Eager and compiled CUDA
forward/backward smokes pass for all three modes.

The `phase11_spectral` family is six seed-123, 5k-step, SDPA, 1024-only cells at
`content_dim=128` / trunk 64 with `method_aware_rms`, matching `phase10`
conditions so deltas are measured under identical settings. Free per-frequency
controls are re-run in-family rather than borrowed across families. Positional
parameter counts:

| Cell | Position parameters |
| --- | ---: |
| free amplitude+phase (control) | 2,187,264 |
| free amplitude-only | 1,787,904 |
| slope+offset | 1,413,504 |
| amplitude-slope | 1,405,184 |
| position-offset | 1,396,864 |
| standard RoPE | 0 |

The narrow arms remove about `780k` positional parameters, matching the readout
arithmetic (`8*64*96*2*8` collapsing to a few thousand). Single seed by
intent: this screen looks for a large enough delta to be worth pursuing, not a
tiebreak inside the noise band.

### Phase-11 spectral readout result

All six seed-123 cells completed 5,000 steps with `rc=0`; no OOMs, NaNs, or
tracebacks. The three re-run controls replicate `phase10` to within `0.0008`
(control `4.28477` vs `4.28397`; amplitude-only `4.30306` vs `4.30243`; standard
RoPE `4.41137` vs `4.41176`), confirming that same-seed runs are effectively
deterministic and that cross-family comparison is sound.

| Cell | Eval loss | Position parameters | vs RoPE | vs free control | Tokens/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| free amplitude+phase (control) | **4.28477** | 2,187,264 | -0.1266 | — | 139,060 |
| **slope+offset (3 scalars/head)** | **4.29601** | **1,413,504** | -0.1154 | +0.0112 | 124,116 |
| free amplitude-only (48/head) | 4.30306 | 1,787,904 | -0.1083 | +0.0183 | 141,543 |
| amplitude-slope (2 scalars/head) | 4.32392 | 1,405,184 | -0.0875 | +0.0392 | 125,205 |
| standard RoPE | 4.41137 | 0 | — | +0.1266 | 178,581 |
| position-offset (1 scalar/head) | 4.45335 | 1,396,864 | **+0.0420** | +0.1686 | 112,752 |

**Three scalars per head beat forty-eight free amplitudes.** `slope+offset`
reaches `4.29601` against free amplitude-only's `4.30306` while using `374,400`
fewer positional parameters, so it **strictly dominates** that arm on both axes.
It trails the full free amplitude+phase readout by only `0.0112` while removing
`773,760` positional parameters (35%). This is an efficiency result, not a
quality win: the full free readout remains the best loss.

**The amplitude tilt alone is insufficient.** `amplitude-slope` (`4.32392`) is
`0.0209` worse than free per-frequency amplitude, so a single log-frequency
slope captures most but not all of what the free amplitude envelope does. It
still recovers 69% of the total gain over RoPE using two outputs per head.

**Position offset alone fails, and this refutes a prediction.** The
relativity-preserving argument predicted it should help; instead it is `0.0420`
*worse than standard RoPE*, the second-worst cell in the screen. Translation
invariance is therefore **necessary but not sufficient** — the monotone
"how badly is relativity broken" ordering from `phase10` does not by itself
predict a gain.

**But offset is strongly complementary to amplitude.** Adding it on top of the
slope improves `4.32392 -> 4.29601`, a gain of `0.0279`. This exactly replicates
the `phase10` non-additivity: free phase alone hurt (`4.4348` vs RoPE `4.4118`)
yet added `0.0184` on top of free amplitude. Across two independent screens the
pattern is the same — **amplitude conditioning is load-bearing, and angular
conditioning (phase or position offset) is only useful in its presence.** A
plausible reading is that an angular perturbation distorts the decay profile and
the model needs amplitude freedom to compensate; without it the perturbation is
pure noise on the most valuable short-range signal. `offset_bound=8` tokens was
not tuned and may be too permissive for the isolated arm.

**Narrow readouts buy parameters, not compute.** Throughput is unchanged or
slightly worse (`112–125k` tokens/s versus `139–142k` for the free readouts),
because the trunk and the carrier synthesis are unchanged while the broadcast
plus `tanh` add elementwise work. These are single uninstrumented measurements
that include warmup and should not be over-read. The durable throughput fact is
that the carrier hypernetwork costs roughly 22–30% against standard RoPE
(`178,581` tokens/s), which is the figure any iso-wallclock comparison needs.

## 2026-07-31 — Phase-12 offset parameterization, decomposition, per-head Q/K norm

Seven seed-123, 5k, SDPA, 1024-only cells at `content_dim=128` / trunk 64,
matching `phase11`. All completed `rc=0`. `conditioning.offset_parameterization`
was added (`raw` default, `softplus`, `tanh`) along with a `qk_norm_per_head`
model flag backed by a new `PerHeadRMSNorm` (`[heads, head_dim]` gains, ones
init). All 104 CPU tests pass.

| Cell | Eval loss | phase11 reference | Delta |
| --- | ---: | ---: | ---: |
| per-head QK norm + free carrier | 4.28728 | free control `4.28477` | +0.0025 |
| slope+offset, raw | 4.29998 | slope+offset tanh `4.29601` | +0.0040 |
| slope+offset, softplus | 4.30248 | slope+offset tanh `4.29601` | +0.0065 |
| per-head QK norm + standard RoPE | 4.41878 | standard RoPE `4.41137` | +0.0074 |
| offset only, raw | 4.45747 | offset only tanh `4.45335` | +0.0041 |
| offset only, softplus | 4.45789 | offset only tanh `4.45335` | +0.0045 |
| position-only warp (offset) | 4.46848 | standard RoPE `4.41137` | +0.0571 |

**Every arm is a negative.** Nothing in this batch improved on `phase11`.

**Offset parameterization does not help, and bounded tanh was mildly best.**
Both `raw` (unbounded signed) and `softplus` are worse than the original
`8 * tanh(z)` in both the isolated and combined arms, by `0.0040-0.0065`. That
is above the `0.0006` same-seed determinism floor but small. The theoretical
objections to tanh — saturation, poor gradients at integer token scales, a
handcrafted bound — are sound but do not show up empirically at this horizon;
if anything the bound is doing useful work. The axis is closed as a null.

**A deterministic position warp does not work.** Conditioning the offset on
position only (`p -> p + m(p)`, a rank-1 constraint on the phase table shared
across frequencies) reaches `4.46848`, worse than standard RoPE by `0.0571`,
using only `86,144` positional parameters. Combined with the isolated-offset
failure, angular reparameterization of the position axis is unproductive in
every form tested so far.

**Per-head Q/K norm hurts in both settings, with one caveat.** On the free
carrier, where both sides use RMSNorm, the comparison is clean and per-head
gains cost `0.0025`. On standard RoPE the `+0.0074` is **confounded**:
`PerHeadRMSNorm` is always RMS, while the shared-gain baseline for that cell
used `legacy_layernorm`, so the comparison mixes per-head gains with a
LayerNorm-to-RMSNorm change. A clean shared-RMSNorm control is needed before
that number is quoted. The direction is consistent across both cells, so the
provisional read is that extra per-head normalization capacity is not useful
here — which also argues the carrier's amplitude branch is *not* merely
compensating for an over-constrained shared Q/K gain.

**The phase11 complementarity result survives.** `slope+offset` at `4.29998`
remains far better than offset alone (`4.45747`) and better than
`amplitude_slope` alone (`4.32392`), independent of parameterization. Amplitude
conditioning is load-bearing; angular conditioning only helps in its presence.

Not run: the compression decomposition (free amplitude + offset, and slope +
free phase). Those need mixed-width component modes — one branch pairwise, the
other spectral — which the current `component_width` is a single scalar for.

## 2026-07-31 — Phase-13 de-confounded per-head Q/K norm and compression decomposition

Four seed-123, 5k, SDPA, 1024-only cells, all with `qk_norm_mode
=method_aware_rms` so no cell uses LayerNorm. Two mixed-width carrier component
modes were added (`amplitude_offset`, `slope_phase`), each narrowing exactly one
branch; `CarrierHypernetwork` now carries per-component widths instead of one
shared width. All 104 CPU tests pass, eager CUDA smoke passes.

### Per-head Q/K norm, de-confounded

| Cell | Eval loss |
| --- | ---: |
| standard RoPE, shared RMSNorm gain | **4.41610** |
| standard RoPE, per-head RMSNorm gains | 4.41934 |

The clean gap is `+0.0032`, not the `+0.0074` reported in `phase12`; the
LayerNorm-to-RMSNorm change accounted for most of that. It matches the clean
free-carrier comparison (`+0.0025`), so **per-head Q/K gains are a small,
replicated negative** and the axis is closed. This also argues the carrier's
amplitude branch is not merely compensating for an over-constrained shared Q/K
gain, since adding that capacity directly does not help.

Side observation: `legacy_layernorm` standard RoPE reached `4.41137` in
`phase10`/`phase11`/`phase12` versus `4.41610` for shared RMSNorm here, so
LayerNorm is `0.0047` *better* than RMSNorm for plain RoPE at this horizon.
Single seed and small, but it runs against the usual preference for RMSNorm on
Q/K and is worth remembering when choosing a control.

### Compression decomposition

| Arm | Amplitude readout | Angular readout | Eval loss | Position parameters |
| --- | --- | --- | ---: | ---: |
| free control | 48 | 48 | **4.28477** | 2,187,264 |
| slope + free phase | **2** | 48 | 4.28827 | 1,804,544 |
| slope + offset | **2** | **1** | 4.29601 | 1,413,504 |
| free amplitude only | 48 | — | 4.30306 | 1,787,904 |
| free amplitude + offset | 48 | **1** | 4.30665 | 1,796,224 |

**Compressing amplitude is nearly free; compressing the angular branch is not.**
Narrowing amplitude from 48 free values to 2 scalars costs `0.0035`; narrowing
the angular branch from 48 to 1 costs `0.0219`, six times more. This inverts the
naive reading: the amplitude branch carries the mechanism (phase10/phase11) yet
is the one that compresses almost losslessly, while the angular branch is
useless alone yet needs its full width.

`slope + free phase` is the best compressed variant found: `0.0035` behind the
full control while removing `382,720` positional parameters, and it strictly
dominates both `free amplitude only` and `free amplitude + offset` on loss and
parameters simultaneously.

One non-monotonicity worth noting: with a narrow angular branch, *adding*
amplitude capacity hurts (`slope + offset` `4.29601` beats `free amplitude +
offset` `4.30665` by `0.0107`, well above the `0.0006` determinism floor). The
slope tilt and the coherent position offset appear to be a matched pair, whereas
free per-frequency amplitude can partly mimic locality itself and then conflicts
with a single coherent offset.

Nothing in phases 11-13 beats the free control on quality. The result is an
efficiency frontier and a mechanism decomposition, not a better model.

## 2026-07-31 — Phase-14 angular rank sweep and trained-readout spectra

Six seed-123, 5k cells. A `slope_phase_lowrank` component mode factorizes the
phase readout through `angular_rank` (readout emits `r` values, expanded to
`pair_dim` by a learned `[groups, r, pair_dim]` basis). The rank-8 and
free-phase cells saved final weights.

| Cell | Eval loss | Position parameters |
| --- | ---: | ---: |
| free phase (rank 48) | **4.28957** | 1,804,544 |
| rank 32 | 4.29471 | 1,769,728 |
| rank 16 | 4.30735 | 1,587,456 |
| rank 2 | 4.31816 | 1,427,968 |
| rank 8 | 4.32556 | 1,496,320 |
| rank 4 | 4.33308 | 1,450,752 |

**The sweep is non-monotone and partly confounded.** Rank 2 beats rank 8 and
rank 4 by `0.007-0.015`. The factorization is not a clean capacity ladder: the
expansion basis is randomly initialized (the readout keeps the zero anchor, so
the basis cannot also be zero), its Xavier scale depends on rank
(`std = sqrt(2/(r + pair_dim))`), and a two-matrix path has different gradient
dynamics from the single zero-init matrix used by the free readout. Differences
among the low ranks reflect those artifacts more than capacity.

**Revised replication floor.** `phase13`'s `slope + free phase` (`4.28827`) and
`phase14`'s free-phase cell (`4.28957`) normalize to identical configs yet
differ by `0.0013`. Earlier replications were `0.0004-0.0008`. The honest
run-to-run floor is therefore about `0.0015`, not `0.0006`, and gaps below that
should not be ranked.

### Why: the trained phase readout is genuinely mid-rank

Singular spectra of the trained free-phase readout (`[heads, 64, 48]` per
branch, so full rank is 48):

| Layer / branch | rank @90% energy | rank @99% | top singular energy |
| --- | ---: | ---: | ---: |
| L0 q / k | 17.6 / 18.0 | 35.4 / 35.6 | 0.32 / 0.28 |
| L4 q / k | 9.4 / 11.6 | 28.2 / 30.2 | 0.46 / 0.40 |
| L7 q / k | 8.8 / 11.9 | 27.9 / 30.5 | 0.51 / 0.38 |

This makes the loss table coherent. The angular branch really does use
**roughly rank 30**, so rank 32 landing within `0.005` of free is expected,
while every cell at rank `<= 16` is far under-ranked and their ordering is
noise. **There is no cheap compression available in the angular branch** — its
`+0.0219` cost at rank 1 is a genuine capacity limit, not a parameterization
accident.

**Effective rank falls with depth** (`~18` at layer 0 to `~9-12` at layer 7).
That mirrors the independent finding from the relative-logit curves, where the
fitted `tau` grew with depth (`7.2 -> 12.7 -> 37.9`): deeper layers are flatter,
longer-range, and need less fine angular structure. Two unrelated analyses
agreeing on a depth trend is the strongest structural signal so far.

Per-output-unit, the narrow components carry *more* weight than the free ones
(`gain/slope` `0.565/sqrt(2) = 0.40` versus phase `1.996/sqrt(48) = 0.29` at
layer 0), consistent with the slope being an efficient summary of the amplitude
envelope while the phase readout spreads its work across many directions.

**Standing frontier after phases 11-14** (5k, h768/d8, seed 123):

| Variant | Loss | Position parameters |
| --- | ---: | ---: |
| free amplitude + free phase | **4.28477** | 2,187,264 |
| slope + free phase | 4.28827 | 1,804,544 |
| slope + offset | 4.29601 | 1,413,504 |
| standard RoPE | 4.41137 | 0 |

Nothing beats the free control on quality; the result remains an efficiency
frontier plus a mechanism decomposition (amplitude is 2-dimensional and
compresses freely, angular is ~rank-30 and does not).

## 2026-07-31 — Phase-15 weight decay on anchored parameters, and cross-head readout mixing

Six seed-123, 5k cells. Two changes shipped: `exclude_position_from_decay`
(explicit exemption list for `qk_position` / `logit_bias_position` /
`position_content` / `carrier_hypernetwork`, since the existing `no_decay` rule
matched only `bias`/`norm`), and `conditioning.readout_head_mixing`, a dense
`[heads*hidden -> heads*out]` carrier readout replacing the block-diagonal one.
106 CPU tests pass; eager and compiled CUDA smokes pass.

| Cell | Eval loss | Position parameters | Reference | Delta |
| --- | ---: | ---: | ---: | ---: |
| **head-mixed readout, no decay** | **4.26708** | 7,692,288 | free control `4.28477` | **-0.0177** |
| **head-mixed readout** | **4.26859** | 7,692,288 | free control `4.28477` | **-0.0162** |
| wide trunk 256 control | 4.28106 | 6,352,896 | phase9 `4.2810` | +0.0000 |
| free control, no decay | 4.28454 | 2,187,264 | decayed `4.28477` | -0.0002 |
| slope+offset, no decay | 4.29533 | 1,413,504 | decayed `4.29601` | -0.0007 |
| mapped AddRoPE, no decay | 4.34031 | 1,308,672 | decayed `4.33964` | +0.0007 |

### Weight decay on zero-anchored parameters: no measurable effect

The concern was structurally real -- the carrier readouts are zero-initialized
so the channel starts at exactly `cis(omega*p)`, so decaying them toward zero is
a prior against using the mechanism rather than a shrinkage prior on large
weights -- but it does not matter empirically. All three exempted cells land
within `0.0007` of their decayed counterparts, inside the `~0.0015` replication
floor. The earlier worry that this might have biased phases 9-14 toward
"conditioning does not help" is **not supported**; those results stand. The flag
is kept because the exemption is still the correct default for anchored
parameters, but it changes nothing at this horizon.

### Cross-head readout mixing: the first arm to beat the free control

`readout_head_mixing` improves the free control from `4.28477` to `4.26859`, a
gain of `0.0162` -- more than ten times the replication floor, and the first
result in phases 11-15 that is a **quality** win rather than an efficiency one.

The trunk's grouping is free: its input (broadcast content plus shared position
features) is identical across heads, so a grouped `[heads, in, hidden]` map is
exactly equivalent to a dense `[in, heads*hidden]` map that is then split. The
readout is where grouping bites, because after the nonlinearity each head holds
a *different* nonlinear feature set computed from the same input, and a
block-diagonal readout forbids head `h` from using any of them. Effectively each
head had a hidden width of 64 rather than 512.

**It is not merely capacity.** The wide-trunk-256 control uses `6.35M`
positional parameters against head-mixing's `7.69M` (21% fewer) and reaches only
`4.28106`; head-mixing beats it by `0.0125`. For scale, widening the per-head
trunk from 64 to 256 -- a 4x parameter increase -- bought only `0.0037`. Sharing
features across heads is doing something that adding per-head width does not.
The wide-trunk control also replicates `phase9_hyper_capacity`'s content-128 /
trunk-256 cell (`4.2810`) to four decimals, confirming the comparison.

This reframes the `phase9_hyper_capacity` null. That screen concluded capacity
was not binding, and it was right -- but it only ever varied *per-head* width
and sharing of whole modules (shared readout, shared head, both worse). It never
tested partial sharing, where heads keep independent readouts but may read each
other's features.

Cost: `3.5x` the free control's positional parameters. Open questions are
whether a cheaper form retains the gain (low-rank cross-head mixing, mixing at a
subset of layers, or a shared bottleneck) and whether it survives to 30k.

### Standing frontier after phases 11-15 (5k, h768/d8, seed 123)

| Variant | Loss | Position parameters |
| --- | ---: | ---: |
| **head-mixed readout** | **4.26859** | 7,692,288 |
| wide trunk 256 | 4.28106 | 6,352,896 |
| free amplitude + free phase | 4.28477 | 2,187,264 |
| slope + free phase | 4.28827 | 1,804,544 |
| slope + offset | 4.29601 | 1,413,504 |
| standard RoPE | 4.41137 | 0 |

## 2026-07-31 — Phase-16 cheap forms of cross-head readout mixing

Six seed-123, 5k cells. `readout_head_mixing` became an enum
(`none`/`dense`/`lowrank`, with the legacy boolean normalized to
`none`/`dense`), plus `readout_mix_rank` and `readout_mix_alpha`. The `lowrank`
mode keeps the per-head readout and adds a rank-`r` cross-head residual,
initialized LoRA-style: down random with a **rank-independent** fan-in, up zero,
output scaled by `alpha/rank`. That scaling is the fix for the confound that made
phase14's angular sweep unreadable -- ranks now differ in capacity rather than in
effective learning rate. 108 CPU tests pass; eager and compiled CUDA smokes pass.

| Cell | Eval loss | Position parameters | vs free control |
| --- | ---: | ---: | ---: |
| dense mixing | **4.26756** | 7,692,288 | **-0.0170** |
| low-rank r16 | 4.27945 | 2,514,944 | -0.0051 |
| low-rank r64 | 4.27971 | 3,497,984 | -0.0048 |
| low-rank r8 | 4.28024 | 2,351,104 | -0.0043 |
| low-rank r32 | 4.28089 | 2,842,624 | -0.0036 |
| free control | 4.28451 | 2,187,264 | — |

Dense mixing replicates phase15 (`4.26756` vs `4.26859`, inside the `~0.0015`
floor), and the free control replicates twice over (`4.28451` / `4.28454` /
`4.28477`).

**The cheap forms do not work.** Low-rank cross-head residuals recover only
about a quarter of the dense gain (`~0.005` of `0.0170`), and the result is
**flat in rank**: `r8` through `r64` span `0.0014`, inside the replication
floor. Since the rank sweep is now properly scaled, that flatness is a real
finding rather than an artifact -- adding cross-head rank does not buy anything,
so whatever dense mixing provides is not a low-rank correction to the
block-diagonal readout.

This narrows the mechanism. The dense `[groups*hidden -> groups*out]` map is not
usefully approximated by `block-diagonal + low-rank`, which means its advantage
is either genuinely high-rank in the cross-head direction, or is an optimization
property of one large matrix rather than an expressivity property at all --
mirroring the shared-versus-per-head Q/K gain result from phase13, where merging
parameters that see more gradient beat splitting them.

Dense cross-head mixing therefore stands as a real but **expensive** win: the
best 5k loss in the project's SDPA line at `3.5x` the control's positional
parameters, with no cheaper form found. Untested alternatives that are sparsity
rather than low rank: mixing within head groups (block size 2 or 4) instead of
all 8, and dense mixing at only a subset of layers.

### Standing frontier after phases 11-16 (5k, h768/d8, seed 123)

| Variant | Loss | Position parameters |
| --- | ---: | ---: |
| **dense cross-head readout mixing** | **4.26756** | 7,692,288 |
| low-rank cross-head mixing (r16) | 4.27945 | 2,514,944 |
| wide trunk 256 | 4.28106 | 6,352,896 |
| free amplitude + free phase | 4.28451 | 2,187,264 |
| slope + free phase | 4.28827 | 1,804,544 |
| slope + offset | 4.29601 | 1,413,504 |
| standard RoPE | 4.41137 | 0 |

## 2026-07-31 — Phase-17 30k horizon gate

Six seed-123 cells at 30,000 steps, all on `method_aware_rms` so standard RoPE
differs from the rest only in the position channel. All completed `rc=0`.
Nothing in phases 11-16 had been run past 5k.

| Cell | 30k | 5k | 30k vs free control | 5k vs free control | Tokens/s | Position params |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| wide trunk 256 | **3.55872** | 4.28106 | **-0.0161** | -0.0035 | 140,679 | 6,352,896 |
| dense head mixing | 3.56074 | 4.26756 | -0.0141 | **-0.0170** | 153,100 | 7,692,288 |
| slope + free phase | 3.57243 | 4.28827 | -0.0024 | +0.0038 | 151,630 | 1,804,544 |
| free control | 3.57480 | 4.28451 | — | — | 154,745 | 2,187,264 |
| position-only | 3.57865 | ~4.304 | +0.0039 | ~+0.014 | 169,405 | 876,544 |
| standard RoPE | 3.62539 | 4.41137 | +0.0506 | +0.1269 | 180,198 | 0 |

### The phase-15 "it is not capacity" conclusion is refuted

At 5k, dense cross-head mixing beat the parameter-matched wide-trunk control by
`0.0125`, which is why phase15 concluded the gain was feature sharing rather
than capacity. At 30k the ordering **reverses**: wide trunk reaches `3.55872`
against head mixing's `3.56074`, and does so with 17% fewer positional
parameters. Capacity simply needed longer to pay off. The two are now within
`0.002`, close to the `~0.0015` replication floor, so they are effectively tied.

This is the third 5k-to-longer-horizon reversal in the project, after the
rank-32 logit and the Q/K basis size. Head mixing survives as a real effect
(`-0.0141` against the free control, ten times the floor) but the *mechanistic*
claim made for it does not.

### Content conditioning contributes almost nothing

Position-only conditioning lands within `0.0039` of the full content+position
control, down from about `0.014` at 5k. It uses `876,544` positional parameters
against `2,187,264` and runs 9% faster. The gap is shrinking with training, so
the "content-conditioned positional encoding" framing is not what this line of
work has actually been demonstrating; the durable object is a **learned per-head
positional profile**, with content conditioning a small and diminishing
correction.

### The RoPE gap keeps halving

`0.127` at 5k, `0.0506` at 30k -- consistent with the earlier observation of
roughly halving per 6x tokens (`0.122 -> 0.056` on the phase9 line). Naive
extension gives `~0.020` at 180k and `~0.008` at 1M steps.

### Iso-wallclock

The local slope at 30k is `0.250` loss per `ln(step)` (from the 25k and 30k eval
points), **not** the `0.44` average across 5k-30k; using the average
overstates what extra RoPE steps buy. Granting standard RoPE the extra steps its
throughput advantage affords:

| Arm | 30k loss | RoPE iso-steps | RoPE loss there | Margin |
| --- | ---: | ---: | ---: | ---: |
| wide trunk 256 | 3.5587 | 38,428 | 3.5636 | +0.0049 |
| dense head mixing | 3.5607 | 35,310 | 3.5847 | +0.0240 |
| slope + free phase | 3.5724 | 35,652 | 3.5823 | +0.0099 |
| free control | 3.5748 | 34,934 | 3.5874 | +0.0126 |
| **position-only** | 3.5787 | 31,911 | 3.6100 | **+0.0313** |

Every conditioned arm still beats RoPE at equal wall clock, but the margins are
much smaller than the raw losses suggest, and wide-trunk's advantage nearly
vanishes (`+0.0049`) once its 22% throughput cost is charged.

**Position-only is the best throughput-adjusted variant in the project.** It
costs only 6% throughput against RoPE, uses the fewest positional parameters of
any conditioned arm, and its iso-wallclock margin (`+0.0313`) is the largest.
The entire content-conditioning apparatus costs 12% more throughput and `1.3M`
more parameters to buy `0.0039` of loss.

### Standing conclusions

- Nothing measured within the conditioned group at 5k survived unchanged to 30k;
  5k orderings inside `~0.02` should be treated as unreliable, not merely noisy.
- The efficiency and throughput-adjusted default is **position-only conditioning**.
- The best raw loss is wide trunk 256 / dense head mixing, effectively tied, at
  3-3.5x the positional parameters and 15-22% throughput.
- Standard RoPE remains the throughput endpoint and its deficit continues to
  shrink with training.

## 2026-08-02 — Phase-18 h1024/d12 scale gate

Four seed-123 cells at 30,000 steps, h1024/d12 with **8 heads** (head_dim 128,
pair_dim 64) so the carrier has the same number of groups as h768/d8, lr `4e-4`,
`beta1=0.95`, `beta2=0.999`, batch 8. Betas are family-scoped in the launcher so
earlier families keep the settings their results were produced under. An initial
launch using the phase6 recipe (16 heads, lr `2.5e-4`, betas `0.9/0.98`) was
stopped and discarded before completion.

| Cell | h1024/d12 | h768/d8 | Gap vs RoPE (h1024) | Gap vs RoPE (h768) | Erosion | Tokens/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| wide trunk 256 | **3.29126** | 3.55872 | 0.0573 | 0.0667 | 14% | 71,075 |
| free control | 3.30178 | 3.57480 | 0.0467 | 0.0506 | 8% | 75,008 |
| position-only | 3.30639 | 3.57865 | 0.0421 | 0.0467 | 10% | 85,285 |
| standard RoPE | 3.34852 | 3.62539 | — | — | — | 89,536 |

### Width barely erodes the gap; tokens do

Going from h768/d8 to h1024/d12 -- about 2.7x the non-embedding parameters --
costs the positional advantage only `8-14%` of its size. Compare the token axis,
where 6x more tokens **halved** it (`0.127` at 5k to `0.0506` at 30k). This
independently reproduces the phase6 observation that the logit stack held
`0.094` over RoPE at both h768 and h1024 while the token axis eroded it.

**The practical consequence is that the shrinking-gap concern is specifically
about training duration, not model size.** A positional method that looks good
at a given token budget should keep most of its advantage as the model grows,
but should be expected to lose roughly half per 6x tokens.

This is also the first ordering in phases 11-18 to survive a scale change
unchanged: wide trunk < free control < position-only < RoPE at both sizes.

### Content conditioning stays negligible at scale

Position-only's deficit to full content+position conditioning is `0.0046` at
h1024 against `0.0038` at h768 -- essentially flat, and near the replication
floor. Content conditioning does not become more useful with model size. It
costs `2.36M` extra positional parameters and 12% throughput to buy `0.005`.

### Capacity matters less at larger width

Wide trunk's advantage over the free control shrank from `0.0161` at h768 to
`0.0105` at h1024. Extra positional capacity is worth less when the model itself
is wider, which is the expected direction and further weakens the case for the
high-parameter arms.

### Iso-wallclock: position-only is the only arm that survives

Local slope at 30k is `0.272` loss per `ln(step)` (25k/30k eval points).
Granting RoPE the extra steps its throughput affords:

| Arm | h1024 loss | Tokens/s | Iso-wallclock margin vs RoPE |
| --- | ---: | ---: | ---: |
| **position-only** | 3.30639 | 85,285 | **+0.0289** |
| free control | 3.30178 | 75,008 | -0.0014 |
| wide trunk 256 | 3.29126 | 71,075 | -0.0055 |

At h768 every conditioned arm beat RoPE at equal wall clock. At h1024 only
**position-only** does. The heavier arms are now *worse than RoPE* per unit of
compute: their 16-21% throughput cost exceeds the loss they buy. Position-only
costs only 4.7% throughput, which is why it survives.

### Caveat

lr and betas differ from phase17 (`4e-4` / `0.95` / `0.999` versus `3e-4` /
`0.9` / `0.98`), so the cross-scale gap comparison carries an optimizer
confound; some of the `8-14%` erosion could be optimizer rather than width.
Within-family comparisons at h1024 are clean. The consistency with phase6's
independent h768-versus-h1024 result argues the direction is real. Closing this
would take one h768 run at the new recipe.

### Standing conclusion

**Position-only conditioning is the recommended method.** It is the only arm
that beats standard RoPE per unit of compute at h1024/d12, uses the fewest
positional parameters (`1.71M`), and its `0.0421` raw advantage erodes only
slowly with width. Content conditioning and extra positional capacity are both
dominated once throughput is charged.

## 2026-08-02 — Documentation checkpoint and handoff

`HANDOFF.md` rewritten for an incoming agent with no prior context. It now leads
with the headline claim, the current standing at both model sizes, and the three
scaling facts, and contains an **audit brief** intended to invite scrutiny rather
than transfer confidence: the four errors actually made during this work (the
invalid causal-Toeplitz rank bound, the phase15 mechanism claim refuted at 30k,
the near-miss on average-versus-local slope in the iso-wallclock analysis, and
the silently ignored `angular_rank` config key), seven specific things to verify,
and three framing questions worth an independent view.

`POSITION_CONFIG.md` gained a recommended-default section naming position-only
conditioning, with the content-conditioning cost stated in both loss and
throughput terms.

The weakest links in the current headline, in order: the phase18 optimizer
confound (lr and betas differ from phase17, so some of the `8-14%` width erosion
may be optimizer); single-seed everywhere, with the position-only versus free
control gap of `0.0046` below what one seed resolves; and throughput figures
taken from a single logged value that includes warmup, on which the entire
compute-adjusted argument rests.

## 2026-08-05 — Independent audit reconciliation and phase-19 disclosure

An independent reviewer followed `INDEPENDENT_REVIEW_BRIEF.md` and inspected the
implementation, configs, raw per-example evaluations, checkpoints, and logs.
No comparison-invalidating code bug was found. In particular, the reviewer
independently verified the fp32 RoPE path, exact frequency/controller anchors,
controller locality and Q/K sharing, paired per-name initialization, optimizer
grouping, and the phase-20 through phase-23 analysis arithmetic. This raises
confidence that the recent static- and dynamic-frequency nulls are real results
of the tested formulations rather than silent implementation failures.

The audit found a material process omission: phase 19 had already locked and
launched the paired h1024 confirmation described in
`CONFIRMATION_PROTOCOL.md`, but its interrupted status was absent from this
journal and `HANDOFF.md`. Artifact inspection gives the exact inventory:

- 15 locked configs: 5 arms x seeds `123/456/789`;
- one completed run: `position-only/seed123`;
- three interrupted `content-position` jobs ending at training steps
  29,350 / 29,491 / 29,988;
- interrupted `standard-rope/seed456` at 21,449 and
  `mapped-addrope/seed456` at 1,307;
- nine jobs that were queued but never began (zero-byte logs);
- no intermediate checkpoints, so interrupted runs must restart rather than
  resume.

All live logs stopped around 2026-08-02 11:28 UTC. The surviving artifacts do
not establish why the parent launch was terminated, so the journal does not
attribute a cause.

The completed phase-19 position-only model reached development loss `3.30964`,
within `0.00325` of phase 18's `3.30639` on the same 25-batch window. Its new
disjoint 1,024-example holdout loss is `3.43016`; there is not yet a paired RoPE
model on that holdout, so this absolute value cannot confirm the headline gap.
The same-window replication supports the sign/stability of the candidate but
does not repair the missing paired multi-seed comparison.

The h1024 headline is therefore restated as a strong screening result, not a
confirmed `0.0421` effect. Phase-17/18 endpoints were the last observations on
a repeatedly used 25-batch development window; there was no early stopping, so
comparisons remain internally consistent, but the values are not disjoint
holdout estimates. Phase-18 arms also used independent rather than paired base
initialization. The claim that content adds only `0.004-0.005` is unresolved at
that precision.

The throughput arithmetic itself checks out, but the inputs are single-run
measurements including evaluation/profiling/tracking on a shared box. Same fixed
RoPE configs measured across phases varied by about `2.8%`, comparable to the
reported `4.7%` position-only cost and larger than the margin deciding the
free-control/wide-trunk iso-wallclock negatives. Position-only's `0.0289` raw
iso-wallclock margin is plausibly robust; the exact throughput cost and close
negative rankings require a same-GPU steady-state benchmark.

The audit also sharpened the frequency interpretation. Phase-22 additive
frequencies did not numerically collapse, but `11-12%` became negative and p95
extra phase at position 1024 reached `42-45` radians, so "finite spectra"
undersells substantial winding and sign reversal. Phase-23 low-rank controllers
were active but mostly operated inside tanh's near-linear region; rare extrema
approached the one-radian bound. This makes free-vs-bounded output a legitimate
conceptual question but a low-priority training experiment.

Revised order of work:

1. complete the decisive phase-19 position-only vs RoPE paired three-seed gate;
2. run a controlled steady-state h1024 throughput benchmark;
3. run phase-23 checkpoint-only zero/mean/shuffle ablations if the mechanism
   wording is worth resolving;
4. de-confound h768 vs h1024 optimizer settings;
5. only then consider a genuinely distinct dynamic mechanism, such as a causal
   cumulative content clock with explicitly positive increments, rather than
   another bounded-output map.

The checkpoint ablations require GPU inference but no retraining. They measure
endpoint reliance on the learned controller and sensitivity to token alignment;
they do not alone identify the causal source of any training-time gain.

## 2026-08-17 — Instance migration, artifact loss, and literature review

The project now runs on a fresh Vast instance (8x RTX 5090, torch
2.12.0+cu130, Python 3.12). **Nothing outside git survived the previous box.**
Lost: all of `model-output/` (every run's metrics, checkpoints, per-example
final evaluations, and the four `*_RESULTS.md` analysis files), the tokenized
OpenWebText cache, `gpu-claim`, and `/workspace/GPU_QUEUEING.md`. The journal
preserved the numbers; the raw artifacts backing them are gone. `/workspace`
on this instance is also **not** a persistent volume, so off-box sync of small
artifacts (git) and weights (external storage) must precede new training.

Consequences for the standing plan:

- Phase-19 cannot reuse `position-only/seed123`; all 15 locked runs restart
  fresh. This permits enabling `checkpointing_steps` uniformly across arms
  without breaking reuse — to be recorded as a protocol revision in
  `CONFIRMATION_PROTOCOL.md` at launch. Old absolute losses are reference
  points only: new hardware, new torch, and a rebuilt dataset cache (datasets
  5.0.1) mean the tokenized data may not be byte-identical. Internal pairing
  is unaffected.
- The phase-23 checkpoint ablations (consolidated plan, work package C) are
  impossible as specified — the rank-32 SiLU checkpoints no longer exist.
  The local dynamic branch is closed on the existing `-0.002` sub-gate result;
  retraining for the mechanism diagnostic is not currently justified.
- All prior throughput measurements are hardware-obsolete. Work package B is
  to be re-run on this box; the `+0.0289` iso-wallclock margin should not be
  quoted against new numbers.

Environment verification on this box: 120/121 CPU tests pass (1 known skip)
after installing `datasets`/`transformers`/`accelerate`/`wandb` into
`/venv/main`; all 15 phase-19 configs load and `--dry_run` cleanly; SDPA,
FlexAttention, and compiled forward/backward verified on the 5090s. One stale
artifact: `scripts/position_v2_cuda_smoke.py` still constructs the removed
`amplitude_phase_frequency` mode and crashes at import (fix before relying on
the smoke gate). Launchers hard-require `gpu-claim`, which needs a shim here.

An independent code re-audit of the load-bearing position-only claim
confirmed the content path is severed *structurally* in that mode
(`PositionContentProjection` is never constructed; the only content references
on the carrier path are shape/dtype metadata reads). Two caveats recorded:
`paired_initialization_seed` does not cover `_GroupedHyperTrunk.weight`
(raw parameter, xavier init without a per-name generator, and construction
order lets arm-specific modules shift the global RNG stream — harmless for the
phase-19 contrasts, but pairing should not be assumed for carrier trunks), and
there is no behavioral content-invariance test — the guarantee is only that
the projector is `None`. A test feeding two different token batches and
asserting identical position addends would close that gap cheaply.

`LITERATURE_REVIEW.md` added (four web sweeps, 2026-08-16): the field's
in-distribution evidence matches our effect sizes; cumulative content signals
beat local bounded ones everywhere; decay beats phase wherever separated;
closest neighbors to the carrier are Goat (arXiv:2601.15380) and, for the
backlog clock, Selective RoPE (arXiv:2511.17388). The novelty framing, the
must-cite list, the LeRoPE reconciliation obligation, and seven ranked
experiment candidates are in that file.

## 2026-08-19 — Phase-19 confirmation complete: headline confirmed, hypernetwork unnecessary, content increment revised upward

All 15 locked runs completed on the third box (8x RTX 5090). Fourteen ran
clean; `mapped-addrope-a03/seed456` failed at step 20,000 when a checkpoint
write hit a full disk (the suite's own checkpoints had flooded it), and was
resumed from `step_15000` under the 2026-08-17 protocol revision — the first
practical payoff of resumable checkpoints. Checkpoints were pruned after
completion (~295 GB reclaimed); final weights, per-example evaluations, and
`confirmation_analysis.json` are snapshotted under `results/phase19_confirmation/`
in git.

Locked analysis on the disjoint 1,024-example holdout (candidate minus
reference; negative favors candidate):

| Contrast | Seed 123 | Seed 456 | Seed 789 | Mean | All seeds? |
| --- | ---: | ---: | ---: | ---: | --- |
| position-only vs standard RoPE | -0.046123 | -0.060674 | -0.046192 | **-0.050997** | yes |
| position-only vs mapped AddRoPE | -0.004087 | -0.004372 | +0.001532 | -0.002309 | no |
| position-only vs matched-FFN RoPE | -0.044198 | -0.054200 | -0.051717 | -0.050038 | yes |
| content+position vs position-only | -0.011832 | -0.011798 | -0.008806 | -0.010812 | yes |

Reading against the locked decision rules:

1. **The headline is confirmed.** Position-only beats standard RoPE in all
   three paired seeds with mean `-0.051` on the holdout, five times the `0.01`
   gate. The confirmed effect is *larger* than the phase-18 development-window
   screen (`0.0421`), on new hardware, a new torch build, and a rebuilt
   dataset.
2. **The hypernetwork-specific mechanism is not supported.** Position-only vs
   mapped AddRoPE is `-0.0023` with mixed signs — inside the unresolved band.
   The simple mapped additive carrier (amplitude anchor 0.3) captures
   essentially the entire gain. The durable object is therefore even simpler
   than "learned per-head positional profile via hypernetwork": **a mapped
   additive Q/K carrier suffices.** The hypernetwork apparatus is not the
   recommended default going forward; it is a capacity elaboration that buys
   nothing resolvable at this scale.
3. **The gain is positional, not generic capacity.** Matched-FFN RoPE tracks
   standard RoPE (position-only beats it by `-0.050`), closing the capacity
   confound.
4. **The content-conditioning story reverses.** The screening narrative said
   content adds `~0.004` and shrinking, near the replication floor. On the
   disjoint holdout with paired seeds it adds `-0.0108` mean, favorable in all
   three seeds, with per-seed paired-example CI95s excluding zero (e.g. seed
   123: `[-0.0132, -0.0105]`; seed-level std `0.0017`). Two seeds clear the
   `0.01` materiality line, one sits at `-0.0088`. By the letter of the locked
   rule (mean at least `0.01`, favorable signs, decisive intervals) this is a
   material increment. The 25-batch development window systematically
   understated it. HANDOFF's "content conditioning contributes almost nothing"
   must not be quoted without this correction.

Combined holdout ladder: RoPE ≈ matched-FFN < mapped AddRoPE ≈ position-only
< content+position, with gaps of ~0.049 / ~0.002 / ~0.011.

Environment notes for provenance: runs executed via the box-shared `gpu-claim`
(python reimplementation) on GPUs 0-5 with six other active projects; the
2026-08-19 disk crisis (90%) was resolved by checkpoint pruning and the OWT
cache consolidation (all four OWT projects now read
`/workspace/data/tokenized/openwebtext_gpt2_bs1024`; ngpt's fat cache and the
redundant 512 cache were deleted after read-path verification; raw HF cache
deleted — re-tokenization requires re-download).

Next, in order: (1) work package B steady-state throughput benchmark on this
hardware — the mapped-AddRoPE equivalence makes this more interesting, since
mapped AddRoPE is cheaper than the hypernetwork and may beat it iso-wallclock;
(2) journal-driven write-up per the consolidated plan's section 11, with the
content-increment revision and the mapped-carrier simplification as first-class
results; (3) sync final weights off-box before trusting them (git holds the
small artifacts as of this entry).

## 2026-08-19 — Post-confirmation design-space prune

With phase-19 settled, the playground was pruned to the surviving hypotheses
(244 insertions, 7,591 deletions incl. 155 archived configs; git history
retains everything). Removed: conditioning sources `qk` and `residual`
(content now comes only from the dedicated norm_x projection — full-rank
input, no back-to-back linears, no coupling into W_q/W_k); the relative
logit-bias channel and Inkling variants (closed on SDPA cost; `logit_bias:
{enabled: false}` remains parseable, `enabled: true` errors pointing to
CONCAT_QK_POSITION.md); rotary geometries `projected_phase`/`unit_pair`/
`scaled_phase` (plain `rotary/phase` kept as the one RoPE-modification
control — modifications inside the rotation never paid); learned-frequency
input kinds (closed by phases 20-22); `log_gain_phase`, `pairwise_low_rank`,
per-head QKNorm, and `offset_parameterization` raw/softplus (tanh only).
`attn_impl=flex` survives as a raw backend. Phase-19 locked configs verified
byte-identical; suite now 105 tests + 1 skip (17 tests covered removed
features). Note: the deleted phase12/13 archive included four per-head-QKNorm
controls; recover from git history if ever needed.

Design principle recorded from this round of discussion: interventions sort
by whether they preserve RoPE's shared sequence-wide phase frame. Shared-frame
(static spectra, position-only profiles) is safe; per-token frame-breaking is
null-to-catastrophic; the cumulative clock (monotone shared time-warp) is the
only known content-dependent form that preserves cross-sequence alignment —
next content-conditioned screens should be cumulative-first.

## 2026-08-20 — Content-mechanism probe on the confirmed content+position models

Inference-only ablations of the dedicated content path on the three confirmed
phase-19 `content-position` models, evaluated on the locked disjoint holdout.
Correctness gate: `native` reproduces the recorded holdout losses to within
`1.8e-5` on all three seeds, so the reconstruction path is faithful.

All modes read content only from positions `<= t`. An earlier draft used a
within-sequence permutation and a whole-sequence mean; both let content from
*future* tokens reach `q_t`/`k_t`, and since `q_t` predicts token `t+1` that
leaks the target into its own prediction, with the leak and the intended
damage pushing loss in opposite directions. They were replaced by causal
equivalents and the script now refuses any mode that fails a preflight
causality check.

| Mode | Seed 123 | Seed 456 | Seed 789 | Mean |
| --- | ---: | ---: | ---: | ---: |
| zero | +1.405959 | +1.066780 | +1.258394 | **+1.243711** |
| prefix_mean | +0.365693 | +0.240327 | +0.294711 | +0.300243 |
| lag1 | +0.422861 | +0.414384 | +0.290954 | +0.376067 |
| lag4 | +0.713237 | +0.585823 | +0.647702 | +0.648921 |
| lag16 | +0.710643 | +0.605254 | +0.647472 | +0.654456 |
| lag64 | +0.726364 | +0.650143 | +0.655766 | +0.677424 |

Findings:

1. **Endpoint reliance is enormous and is not the same quantity as
   contribution.** Zeroing content costs `+1.24`, about 115x the `-0.011` that
   content conditioning actually buys over position-only in the paired
   confirmation. The trained network has woven the content path deep into its
   computation, but a model trained without it lands within `0.011`. Ablation
   cost measures coadaptation at the endpoint, not what the mechanism
   contributed during training; this is a clean quantitative example of that
   gap and belongs in the write-up as a methodological point.

2. **The lag curve saturates by 4 tokens.** Cost rises `+0.376` (lag1) ->
   `+0.649` (lag4) and is then flat through lag16 (`+0.654`) and lag64
   (`+0.677`). Content from four tokens ago is already as useless as content
   from sixty-four. The useful signal is local, with no long-range component.

3. **A causal running mean is cheaper than a sharply misaligned token.**
   `prefix_mean` (`+0.300`) costs less than `lag1` (`+0.376`), and less than
   every lag. Per-seed differences are `-0.057`, `-0.174`, `+0.004`: favorable
   in two seeds and effectively tied in the third, so this is directionally
   consistent but does not meet the project's all-seeds standard and is
   reported as suggestive only. It does argue against a purely token-identity
   reading: if the carrier needed *this* token specifically, a prefix mean
   dominated by distant history should hurt more than one adjacent token, not
   less. A smooth causal summary preserves more of what the carrier wants than
   a precise but wrong token does.

Taken together the content path behaves like a locally-varying contextual
modulation rather than a token-identity lookup. This does not close the
cumulative-clock direction; if anything (3) is mildly encouraging for
cumulative formulations, while (2) says any such mechanism should have a short
effective horizon. It also does not raise the value of content conditioning:
the honest end-to-end number remains `-0.011`.

Artifacts: `results/content_mechanism_probe.json`, produced by
`scripts/content_mechanism_probe.py`.

Reproducibility note discovered here: saved `training_config.json` files have
**never** been round-trippable. `load_config` derives `pos_variant="custom"`
and writes it, but has never accepted it as input; saved configs also store
fully normalized channel blocks (so inactive channels carry now-removed
defaults) and the run pads the vocabulary to a multiple of 64. The probe
reconstructs these explicitly and requires an exact state-dict match rather
than relaxing `load_config`'s unknown-key rejection, which is a safety
property that previously caught the `angular_rank` bug. Worth fixing properly
at the source later: write a schema version alongside each saved config, and
record the derived variant tag in a separate field from `pos_variant`.

## 2026-08-21 — External review of the coherence framing; Claim 2 refuted

An external reviewer assessed `COHERENCE_REVIEW_BRIEF.md`. Corrections are
recorded in that file's section 8. Summary of what changed:

- **The "pairwise coherence" formalization was wrong.** `B_mn = U_m^T U_n` is
  cycle-consistent for *any* per-position orthogonal transform applied to both
  q and k, so the property is automatic and separates nothing. Verified
  numerically here (cycle consistency to `1.8e-7` for deliberately arbitrary
  per-pair phases). The meaningful restriction in a shared warp is **spectral
  locking** — all frequency planes advancing on one scalar coordinate — plus
  order preservation with bounded local speed.
- **Claim 1 stands** (relativity-preserving phase edits are affine, mod 2pi,
  per-pair slopes allowed, no cross-pair rescue) but does **not** close the
  in-rotation direction, because fixed-context LM is not translation-stationary.
- **The catastrophic content-frequency result is better explained by gradient
  amplification** (`d theta / d g = omega * p * exp(g)`) than by manifold
  departure. Combined with the bounded-offset null, this reads as evidence that
  phase is simply not a useful channel at this scale. The monotone warp is
  worth one decisive run for mechanistic value, at a weak prior.
- **Amplitude is not temperature.** It controls two content-to-carrier cross
  terms and a carrier Gram kernel. For multiplicative gains, query factor =
  row temperature, key factor = token salience.

Verified locally in response: our pipeline concatenates documents with no
separator and chunks at fixed offsets, so there is no BOS artifact and document
boundaries are uncorrelated with position — but every block starts mid-document,
so early positions carry systematically truncated context and a position-only
profile can exploit that.

**Reprioritized.** The next experiment is no longer the warp but the
gain/salience decomposition: query-only, key-only, and both, driven by the same
position network as the carrier amplitude, exact-null at init, fused SDPA. This
asks whether the confirmed `-0.051` is positional geometry or adaptive
attention allocation. The matched-FFN control did not test this. Phase-24
(RoPE-embed basis) continues in the background; 3 of 12 runs complete.

## 2026-08-22 — Phase 24 complete; dynamic primitives refactored

Phase 24 completed all 12 planned h768/d8, 5k cells. The disjoint 256-example
context-1024 means recovered from launcher logs are: fixed RoPE `4.579567`,
compact basis anchor 0.3 `4.472233`, compact basis anchor 1.0 `4.428000`, and
full native RoPE basis anchor 1.0 `4.442033`. The anchor-1 compact basis beats
anchor-0.3 by `-0.044233` and the full native basis by `-0.014033`, favorable
in all three seeds. Thus amplitude scale was the largest tested lever and the
full native basis did not explain the additive-carrier gain. This remains a 5k
screen. Model-output directories were intentionally removed, so only aggregate
losses survive and per-example confidence intervals cannot be regenerated.
`analyze_rope_embed_basis_screen.py` now reconstructs the durable report from
logs under `results/phase24_rope_embed_basis/`.

Repository cleanup kept provenance separate from active design. The obsolete
`transformer_old.py` and checkpoints stay deleted. Archived schema loading and
the phase-23 frequency controller remain because configs and tests still depend
on them, but the controller is explicitly historical. Saved configs carrying
the derived `pos_variant="custom"` label now round-trip through `load_config`
without relaxing unknown-key rejection.

Two untrained mechanisms were added as isolated first-stage interventions:

1. `qk_preprojection`: add a full-width frozen Fourier vector only before
   `W_q/W_k`, leaving V and the residual stream untouched. By linearity this is
   exactly a tied projected additive carrier.
2. `rotary_clock`: predict bounded positive local speed, exclusive-cumsum it to
   a monotone coordinate, and multiply that one coordinate by the fixed RoPE
   spectrum. This is spectrally locked and removes the direct
   `position * learned_frequency_error` actuator.

The clock supports pointwise and short depthwise causal-convolution controllers
with full/incremental parity. EMA/linear-RNN support is deferred behind the
temporal-controller boundary until a stable differentiable scan or custom
kernel has parity, compile, and throughput evidence. The CPU suite is now 123
passing tests plus one existing CUDA-only skip, including prefix-causality,
exact-anchor, gradient, fp32-frequency, config, and streaming-state checks.

## 2026-09-03 — Phases 26-30: attention-local survivors

Phase 26 screened ten mechanisms at 5k steps, seed 123. The useful results were
AddRoPE amplitude 1.0 (`-0.1673` versus RoPE), the pre-Q/K sinusoid followed by
RoPE (`-0.0815`), and position-only Q/K gain (`-0.0371`). The pointwise and
causal-convolution rotary clocks were only `-0.0043/-0.0047` and stayed inside
the unresolved region. Pre-Q/K injection without RoPE greatly improved NoPE
but remained worse than fixed RoPE (`4.6248` versus `4.5981`). This last
comparison is too short to answer whether RoPE remains necessary.

Phase 27 replicated pre-Q/K+RoPE and position gain across seeds 123/456/789 at
5k. Their mean deltas versus fresh fixed RoPE were `-0.069122` and `-0.024215`,
respectively, favorable in every seed. Position gain explains part, but not
most, of the pre-Q/K advantage.

Phase 28 promoted pre-Q/K+RoPE to 30k across all three seeds. It beat fixed
RoPE by `-0.065235` mean held-out loss, with per-seed deltas
`-0.073489/-0.043428/-0.078788`. The development gap remained favorable at
every 5k checkpoint through 30k. Median throughput was effectively unchanged
(`192,507` versus `192,714` tokens/s).

Phases 29 and 30 tested the qk-preprojection x AddRoPE factorial at 5k and 15k,
seed 123. The combination was worse than AddRoPE alone by `+0.020261` at 5k
and `+0.004934` at 15k. The interaction was strongly sub-additive. The two
mechanisms therefore appear to overlap, but the 15k endpoint is not a
mature-model result.

Durable reports and paired analyses live under `results/phase26_*` through
`results/phase30_*`.

## 2026-09-03 — Phases 31-32: EMA result and decision not to promote

Phase 31 compared rotary clocks and AddRoPE content conditioning at 15k,
seed 123. Pointwise and EMA rotary clocks were both null versus RoPE
(`-0.000360/-0.000362`), and EMA changed the pointwise clock by only
`-0.000001`. On AddRoPE, pointwise content improved over position-only by
`-0.014836`, while per-dimension content EMA improved over pointwise by another
`-0.011100`.

Phase 32 isolated the EMA coefficient granularity. Scalar, per-head, and
per-dimension EMA improved over the pointwise content controller by
`-0.010626/-0.011146/-0.010853`. Differences among the EMA forms were under
`0.0006` with intervals crossing zero. The learned scalar decays were about
`0.856-0.895`, corresponding to effective windows of roughly 7-10 tokens.

The scalar EMA retained nearly full throughput (`147,961` versus `152,866`
tokens/s), while per-head EMA fell to `112,790` tokens/s and doubled peak
memory. Using the local pointwise learning-curve slope to estimate equal
wall-clock progress consumes almost the entire scalar-EMA loss gain: estimated
iso-wall-clock delta is about `-0.0015`. This is an estimate, not a directly
trained endpoint.

Decision: EMA is a valid small step-matched result but is not promoted. It does
not rescue rotary clocks, coefficient granularity adds nothing, and its likely
compute-adjusted value does not justify expanding the recurrent-controller
surface. Preserve the compact result reports, then remove EMA from the active
runtime during repository consolidation.

## 2026-09-03 — Long-horizon and repository consolidation decision

The h768/d8 baseline has approximately 153.4M parameters. With batch 8 and
context 1024, the 5k/15k/30k experiments consume only 41M/123M/246M tokens, or
about 0.27/0.80/1.60 tokens per parameter. The completed results primarily
measure early optimization. In particular, the question of whether a pre-Q/K
sinusoid needs RoPE remains unresolved.

The active next mechanism is a static constrained adapter applied only to the
pre-Q/K positional carrier:

```text
A_i^q = a_i^q R(phi_i^q)
A_i^k = a_i^k R(phi_i^k)
q = W_q(x + A_q z(p))
k = W_k(x + A_k z(p)).
```

The nested ladder is tied scalar, separate Q/K scalars, separate Q/K pairwise
amplitudes, and separate pairwise amplitude+phase. All variants initialize to
the current carrier at `a=1, phi=0`. A per-head axis is intentionally excluded
because the intervention occurs before `W_q/W_k` create heads.

The consolidation screen will use one paired seed, six arms, and a common
200k-step learning-rate horizon from step zero. Milestones are
10k/30k/60k/100k/150k/200k; only finalists receive other seeds. Short runs
whose linear scheduler already reached zero cannot be resumed as equivalent
prefixes of this protocol.

Repository policy: keep fixed RoPE, AddRoPE, Fourier utilities, and the pre-Q/K
carrier; remove learned/dynamic multiplicative RoPE, rotary phase special
cases, clocks, EMA, retired residual/write channels, and the completed
position-gain control from the active runtime after the Phase 29-32 state is
committed. The exact plan is `CONSOLIDATION_PLAN.md`.

## 2026-09-03 — Static adapter and Phase 33 launch gate complete

The static pre-Q/K adapter now supports four nested per-layer modes: a tied
global gain, separate Q/K gains, separate zero-sum pairwise log-amplitudes,
and separate pairwise log-amplitudes plus phases. Pair amplitudes use `P-1`
coordinates in an orthonormal zero-sum basis, so their geometric-mean factor
is exactly one and cannot duplicate the global gain. Every mode has the old
carrier as its exact `g=1, delta=0, phi=0` anchor. The implementation and
tests are commit `58c4e9d`.

Long-run safeguards and the frozen Phase 33 protocol are commit `6aa859d`.
Checkpoints receive a completion marker only after Accelerate finishes writing
state; pruning happens afterward and keeps the newest state plus explicit
milestones. A policy file makes automatic resume ignore partial directories.
Every launch now appends its exact resolved config, Git commit/dirty state,
package and CUDA versions, GPU identity, dataset manifests/fingerprints, and
parameter counts to `run_provenance.json`. Development evaluations now retain
paired per-example losses at periodic milestones as well as at final holdout.

All 184 retained and new configs load, and the CPU suite passes 109 tests with
one CUDA-gated skip. The canonical cache validated at 8,372,843 train and
443,501 validation blocks. All six separate 50-step h768/d8/context-1024
preflights completed through `gpu-claim` from a clean tree on RTX 5090 GPUs.
Post-warmup throughput ranged from 185,639 to 190,790 nominal tokens/s, and
reserved memory ranged from 5,076 to 5,360 MiB. The pair-polar endpoint was
2.70% slower than fixed RoPE in this short measurement. At the six-arm mean,
200k steps are about 2.41 compute hours per arm before validation, compile,
and checkpoint overhead.

A real Accelerate integration smoke saved steps 1/2/3 with keep-latest=1,
left only marked step 3, and then restored model, optimizer, scheduler,
sampler, and RNG state in a fresh process. All preflights emitted the same
compile-time bf16-input/fp32-LayerNorm-weight fusion warning; training and
compilation succeeded, so it is a possible common kernel optimization rather
than an arm-specific blocker. The 50-step losses are health checks only.
Compact evidence is under `results/phase33_static_qkpre_preflight/`.

## 2026-09-04 — Phase 35 smooth carrier shape

After removing learned-RoPE and EMA machinery from the active runtime, Phase 35
tested a narrow carrier-only question. Each layer retained the established
learned tied scalar gate and optionally added four low-order, unit-RMS DCT
modes over log-frequency index for zero-mean log amplitude, phase, or separate
Q/K polar transforms. Standard RoPE was either fixed and present or absent;
none of these parameters modified RoPE itself. All eight variants shared the
same seed, initialization, data order, and 20k linear schedule.

On the disjoint 1,024-example final holdout, smooth amplitude improved the
NoPE carrier by `-0.010644`, paired-example 95% CI
`[-0.011700,-0.009589]`. It improved the RoPE carrier by a smaller `-0.002604`,
CI `[-0.003194,-0.002013]`, below the predeclared `0.003` practical gate.
Phase added `+0.000141/+0.000142` under RoPE/NoPE. Splitting Q/K added
`-0.000264/-0.001506`; neither cleared its gate. Smooth amplitude recovered
17.8% of the RoPE-versus-NoPE gap, while matched amplitude+RoPE still beat
amplitude+NoPE by `0.037168`.

All sparse optimizer traces were finite and unclipped, every adapter produced
nonzero carrier-function movement, and most sampled updates aligned with the
current descent direction. The learned spectra moved substantially, especially
without RoPE. Thus phase and Q/K-untying are well-optimized null/subthreshold
results rather than inactive coordinates. No automatic seed or 200k expansion
is planned; NoPE amplitude remains conditional on a specifically RoPE-free
research objective. Durable results are in
`results/phase35_smooth_carrier_20k/`.

## 2026-09-04 — Post-Phase-35 pre-Q/K runtime consolidation

The active pre-Q/K adapter was reduced to the two modes supported by the
completed evidence: `tied_scalar` and `tied_smooth_amplitude`. The historical
`split_scalar`, `split_pair_amplitude`, `split_pair_polar`,
`tied_smooth_polar`, and `split_smooth_polar` implementations were removed.
This deletes separate Q/K gates, phase coordinates, full per-pair tables, and
their phase diagnostics while retaining the shared rank-4, zero-mean,
unit-RMS DCT amplitude profile.

Enabled configs naming a removed mode now fail with a migration message that
points to the retained alternatives and git history. Disabled archival blocks
canonicalize to `tied_scalar`, since their mode has no model effect. Historical
sweep configs, analysis scripts, and durable result reports were not deleted.

The four-module CPU suite passes 121 tests with one intentional CUDA-only skip.
The ten-case retained matrix also passes two optimizer steps in both eager and
compiled bf16 through `gpu-claim`, including scalar and smooth carriers with
fixed RoPE and NoPE and the supported AddRoPE combination. All losses and
gradients were finite, and the GPU claim was released normally.

## 2026-09-05 — Phase 36 direct amplitude/frequency screen

Phase 36 replaced transform-mediated amplitude/frequency coordinates with
direct, nonsaturating ones while preserving the scalar carrier as an exact
initial anchor. Rank-4 amplitude used `1 + Bc`; global and rank-4 hybrid
frequency used fixed gains chosen to control endpoint-phase derivatives. The
carrier remained tied across Q/K, fixed RoPE stayed enabled, and method-aware
QKNorm diagnostics measured the actual content/position mixture.

Eight 512-step calibration runs selected amplitude LR1, global-frequency LR4,
and hybrid-frequency LR1, with LR4 hybrid retained as a speed sensitivity.
Seven h768/d8/context-1024 arms then completed 20k steps from clean commit
`d7991d9`. Direct amplitude improved its scalar parent by `-0.003661`, paired
CI `[-0.004247,-0.003074]`, and was still improving over the late development
window. The amplitude+frequency arm reached `-0.004118` versus scalar, but
frequency added only `-0.000457` beyond amplitude.

Global frequency was null (`+0.000253`). Hybrid LR1 was `-0.000941`; hybrid
LR4 was `-0.002258` but produced `6.27%` adjacent-order violations. All
frequency optimization traces were finite and unclipped. Direct amplitude is
therefore the only Phase-36 component promoted to longer confirmation; the
frequency paths are retained only for reproducibility pending later cleanup.

## 2026-09-05 — Phase 37 long-horizon matrix frozen

The Phase-36 direct-amplitude result met its screen gate, so the next run is a
narrow 200k confirmation with the scalar, exponential rank-4, and direct
rank-4 carrier-amplitude arms. All three share fixed RoPE, method-aware QKNorm,
the same seed/data order/schedule, and no positional weight decay. This makes
direct versus exponential a paired parameterization comparison and omits the
unpromoted frequency variants. The primary long-horizon gate is a `0.002`-nat
direct-over-scalar improvement with a below-zero paired interval, a
non-collapsing late curve, and healthy carrier diagnostics. The frozen protocol
is `DIRECT_AMPLITUDE_CONFIRMATION_PLAN.md`.

## 2026-09-05 — Phase 37 completed

All three Phase-37 arms completed 200k steps. The initial interactive launcher
exited after step 70k; complete checkpoints were present for every arm, and a
supervisor-managed launch restored model, optimizer, scheduler, sampler, and
RNG state. Both launches recorded the same clean commit `781a8ae` and canonical
dataset fingerprints.

On the primary disjoint 1,024-example holdout, direct amplitude was `+0.000111`
versus scalar, paired CI `[-0.001041,+0.001263]`. Exponential amplitude was
`-0.000363`, CI `[-0.001559,+0.000833]`; direct versus exponential was
`+0.000474`, CI `[-0.000589,+0.001537]`. None passed the `0.002` gate.

The repeated 128-example development slice favored direct by `-0.002427` on
average over 150k--200k, but its per-step uncertainty was large and the larger
frozen holdout did not reproduce it. Direct factors reached `0.004--2.664` and
exponential factors `0.290--4.368`; all optimization traces were finite and
active. Smooth amplitude is therefore closed as a mature-model refinement,
not dismissed for lack of optimization. No seed or scale expansion is queued.

## 2026-09-05 — Scalar-carrier consolidation and Phase 38 freeze

The closed carrier-shape branch was removed from the active runtime. Learned
carrier frequency, exponential/direct smooth pre-Q/K amplitude, their
frequency-specific optimizer/clipping/diagnostic machinery, and phase-35--37
launch/preparation scripts were deleted. The pre-Q/K implementation now has
only one active mode: a per-layer scalar gate multiplying a fixed, tied
full-width Fourier carrier. Disabled historical configs canonicalize to this
form; enabled removed modes fail explicitly. Static AddRoPE and its replicated
pointwise content-conditioned reference remain active.

Superseded root protocols and design briefs were removed after verifying that
their decisions and numerical conclusions are present in this journal,
`CURRENT_STATUS.md`, and compact phase reports. Git remains the source archive.
The CPU suite passes 108 tests with one expected CUDA-only skip, and all eight
remaining carrier/backbone smoke cases pass eager and compiled bf16.

Storage was audited independently. Fourteen completed step-200k checkpoints
all had completion markers, standalone final weights, summaries, metrics,
provenance, and 21 context-1024 evaluations. Those resume states plus one
checkpoint smoke artifact were deleted, reclaiming about 24 GiB; final weights
and evidence remain.

Phase 38 is a fixed-method evidence program rather than another design sweep.
It contains four paired fixed-RoPE versus scalar-pre-Q/K comparisons: h768/d8
seeds 456 and 789 at 200k, h768/d8 without QKNorm at seed 123 and 200k, and an
h1024/d12 scale-up at seed 123 and 200k. The two added h768 seeds join the
existing Phase-33 seed-123 result for mature three-seed evidence. Predeclared
gates and the complete matrix are in `NEXT_EXPERIMENT_ROADMAP.md`.

All four unique full-size operational preflights completed 20 compiled bf16
steps through `gpu-claim`. Peak reserved memory was 4.89--5.02 GiB for the
no-QKNorm h768 pair and 9.34--9.59 GiB for the h1024 pair. The 20-step design
intentionally consumed the complete throughput warmup window, so it validates
execution and memory rather than providing a throughput estimate.
