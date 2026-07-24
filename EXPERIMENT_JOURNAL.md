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
