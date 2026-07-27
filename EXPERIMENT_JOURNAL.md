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
