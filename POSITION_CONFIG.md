# Position playground configuration (schema v2)

MLPRope resolves every position configuration to JSON-safe schema v2 before
constructing the model. Legacy Phase 1/1b/1c dictionaries still upgrade
strictly and retain their historical tags and parameter counts.

The playground separates six independent sectors:

1. position input basis;
2. Q/K application and output geometry;
3. Q/K and head coupling;
4. local content conditioning;
5. residual-stream and attention-output writes;
6. static or content-conditioned relative-logit bias.

Defaults preserve the post-refactor RoPE baseline. New residual/write and
content-conditioned paths are zero-gated where possible.

## Package map

| Module | Responsibility |
| --- | --- |
| `position/config.py` | defaults, strict validation, v1 upgrade, tags |
| `position/basis.py` | frozen/learned Fourier bases and scalar features |
| `position/mappers.py` | grouped identity, affine, linear, low-rank, MLP maps |
| `position/channels.py` | Q/K, logit, residual-stream, attended-position channels |
| `position/rotary.py` | split-half RoPE, phase composition, radial scaling |

`transformer.Attention` retains projections and SDPA/Flex dispatch.

## Full top-level shape

```yaml
position_schema_version: 2
use_rope: true
post_position_qk_norm: false
qk_norm_mode: legacy_layernorm   # legacy_layernorm | method_aware_rms
position_content_dim: 64
position_content_coupling: separate  # shared | separate

qk: { ... }
logit_bias: { ... }
residual_stream: { ... }
attention_write: { ... }
```

`qk_norm_mode=method_aware_rms` uses one learned per-head RMSNorm at the
geometry-appropriate site: additive channels use
`project -> add position -> RMSNorm`, while rotary channels use
`project -> RMSNorm -> rotate`. New null-conditioning experiments use this
mode. `post_position_qk_norm` is a legacy control and cannot be combined with
`method_aware_rms`.

`post_position_qk_norm: true` applies a parameter-free per-head RMS
normalization after all additive/rotary Q/K position operations. The existing
learned QK LayerNorm remains before position injection. This isolates positional
direction from magnitude changes without adding capacity or applying another
mean-subtracting LayerNorm.

`position_source_schema` is written by the trainer. It is `1` only when all
active channels came from legacy v1; otherwise it is `2`.

## Q/K channel

```yaml
qk:
  enabled: false
  application: rotary
  geometry: phase
  input:
    kind: frozen_fourier
    basis_dim: null
    theta: null
    scalars: []
  mapper:
    kind: identity
    residual: false
    rank: 32
    hidden_dim: 128
  output:
    parameter_source: mapped # mapped | direct
    amplitude_init: 0.1
    amplitude_max: 1.0
    amplitude_parameterization: signed
    learn_amplitude: true
    learn_phase: true
    phase_scale: 1.0
    additive_normalization: none
    additive_gain_init: 0.1
    additive_gain_max: 1.0
    learn_additive_gain: true
    scale_init: 1.0
    scale_max: 4.0
    scale_parameterization: exp
  conditioning:
    kind: none
    source: dedicated
    activation: tanh
    hidden_dim: 64
    gate_init: 0.0
  qk_coupling: shared
  head_coupling: per_head_independent
```

### Applications and geometries

| Application | Geometry | Output and operation |
| --- | --- | --- |
| `additive` | `free` | free `[H,L,D]` addend; multiplicative RoPE is skipped |
| `additive` | `pair_normalized` | free split-half pair coordinates projected to radius `amplitude_init` |
| `additive` | `amplitude_phase` | canonical `a·cis(ωp+δ)` addend |
| `rotary` | `phase` | strict `R(θ+δ)` |
| `rotary` | `projected_phase` | predict a free pair-vector, project onto the RoPE tangent |
| `rotary` | `unit_pair` | add a zero-init Cartesian pair residual, normalize to a unit pair, then rotate |
| `rotary` | `scaled_phase` | `s·R(θ+δ)`, with positive/exact-one initialization |

Invalid application/geometry pairs fail during config loading.

`pair_normalized` first constructs the same arbitrary `D`-dimensional output as
`free`, interprets its first and second halves as paired Cartesian coordinates,
and normalizes every pair before multiplying by the fixed
`output.amplitude_init`. Unlike `amplitude_phase`, it does not preserve or
predict an explicit base Fourier phase.

### Canonical amplitude+phase AddRoPE

For each pair:

```text
e_q(p) = a_q(p) · [cos(ωp + δ_q(p)), sin(ωp + δ_q(p))]
q'(p)  = q(p) + e_q(p)
```

Despite the historical name, this mode does **not** apply RoPE to Q/K. The
phase delta rotates the Fourier addend itself and multiplicative RoPE is
disabled. A rotary channel instead computes `q'=R(θ+δ)q`. Thus:

- additive Fourier/AddRoPE: `q + a·cis(θ+δ)`;
- modified RoPE: `R(θ+δ)q`.

K has its own amplitude/phase readouts when coupling permits. The default
amplitude is `0.1`, deliberately below the poor fixed-unit Phase-1c control.

For amplitude+phase geometry, `parameter_source=mapped` predicts amplitude and
phase from the configured position features. `parameter_source=direct` instead
uses only per-head/per-frequency parameters, matching canonical AddRoPE without
a position-feature mapper. `learn_amplitude` and `learn_phase`
independently control the two output heads:

- both `false`: fixed Fourier carrier at `amplitude_init`;
- amplitude only: learned radial magnitude with zero learned phase delta;
- phase only: fixed radial magnitude with learned phase delta;
- both `true`: existing combined AddRoPE behavior.

Disabled component heads and the fully fixed carrier trunk are not instantiated,
so parameter counts reflect the active mechanism rather than dead parameters.
`signed` amplitude is unconstrained around that initialization; `softplus`
keeps it positive. `bounded_sigmoid` maps into `[0, amplitude_max]` and
initializes exactly at `amplitude_init`.

For additive channels, `additive_normalization=rms` normalizes each token/head
position branch before applying a bounded gain in
`[0, additive_gain_max]`. The gain begins at `additive_gain_init` and may be
fixed with `learn_additive_gain=false`. This controls branch contribution
before the optional top-level post-position Q/K RMS normalization.

This is distinct from historical v1 `feature_map=add_rope`, which upgrades to
`euclidean_affine` over Fourier coordinates.

### Projected and scaled rotary

`projected_phase` predicts `[dx,dy]` and keeps only the tangent component:

```text
δ = <-sin(θ), cos(θ)> · [dx,dy]
```

The zero output remains exact RoPE. `scaled_phase` predicts phase plus pairwise
radial scale. `scale_parameterization=exp` uses `scale_init*exp(raw)`;
`linear` uses `scale_init+raw`. `bounded_log` uses
`scale_init*exp(tanh(raw)*log(scale_max))`, restricting pair scales to
`[scale_init/scale_max, scale_init*scale_max]`.

Learned exponential parameterizations use the exact exponential in the forward
pass and a straight-through identity derivative in the backward pass. This
applies to rotary scales, adaptive gains, learned Fourier temperatures and
frequencies, and Inkling CosNet frequencies. It avoids multiplying gradients
by the current exponential value; `bounded_log` still retains the derivative
of its `tanh` bound.

`unit_pair` constructs and normalizes
`[cos(θ)+dx, sin(θ)+dy]`, then converts the valid unit pair back to a relative
phase for the shared rotary kernel. Zero-initialized `[dx,dy]` is exact RoPE;
the nonzero base carrier avoids zero-vector normalization.

## Position inputs

```yaml
input:
  kind: frozen_fourier
  basis_dim: null
  theta: null
  scalars: []
```

Kinds:

- `frozen_fourier`: cached RoPE schedule;
- `learned_temperature_fourier`: one learned positive global frequency scale;
- `learned_frequency_fourier`: independent positive log-frequency residuals.

Both learned kinds initialize exactly to frozen Fourier values.

Fourier layout is interleaved:

```text
[cos_0, sin_0, cos_1, sin_1, ...]
```

This is not split-half Q/K RoPE pairing. Optional scalars are:

- `position`;
- `normalized_position`;
- `log_position`.

Scalars append to the mapper input. Identity, Euclidean-affine, and residual
mappers still require input/output widths to match, so use a non-residual
linear/low-rank/MLP mapper when adding scalars or reducing `basis_dim`.

`basis_dim=null` resolves to `head_dim`, or `model_dim` under
`per_head_joint`. Explicit smaller even widths are allowed.

## Mappers

| Kind | Formula / initialization |
| --- | --- |
| `identity` | passthrough |
| `euclidean_affine` | `(1+scale)·x + offset`, zeros |
| `linear` | full affine, Xavier/zero; adds its input when `residual=true` |
| `low_rank` | linear `down→up`; residual optional |
| `bottleneck_mlp` | `down→GELU→up`; rank width |
| `mlp` | `down→GELU→up`; independent hidden width |

Residual branches use zero-initialized up projections and therefore begin as
identity. Non-residual low-rank/MLP branches Xavier-initialize both projections
so a downstream zero gate still receives a gradient.

## Coupling

Q/K coupling:

| Value | Semantics |
| --- | --- |
| `shared` | one pipeline/readout (v1 parity) |
| `shared_trunk_separate_readouts` | shared position trunk, independent equal-init Q/K heads |
| `separate` | deep-copied independent trunks and heads |

Head coupling:

| Value | Semantics |
| --- | --- |
| `shared_head` | one group broadcast over heads |
| `per_head_independent` | one mapper/readout group per head |
| `per_head_joint` | joint model-dimension mapper, reshaped into heads |

Recent Phase-2 results favor `shared_trunk_separate_readouts` for additive
linear Q/K.

## Dedicated positional content and Q/K conditioning

Content-conditioned Q/K and relative-logit mechanisms receive independent
low-rank projections of the block-normalized residual:

```text
c_q = RMS(P_q(norm_x))
c_k = RMS(P_k(norm_x))
```

The default width is `64`. `position_content_coupling=shared` reuses one
projection; `separate` gives Q and K independent projections. New configs do
not derive conditioning content from attention Q/K. Legacy `source=qk` and
`source=residual` values remain loadable only to reproduce historical runs.

```yaml
conditioning:
  kind: none               # also carrier_hypernetwork and legacy conditioners
  source: dedicated
  activation: tanh         # tanh | gelu | linear | scaled_sigmoid
  hidden_dim: 64
  input_mode: content      # content | position | content_position
  input_normalization: none # none | modality_rms
  learnable_input_gains: false
  network: linear          # linear | silu_mlp | swiglu_mlp
  components: phase        # see carrier component modes below
  head_coupling: per_head_independent # shared_head | per_head_independent
  gate_init: 0.0
  target: both             # q | k | both
  coupling: shared_trunk_separate_readouts
  static_complement: false # learned direct AddRoPE on inactive q/k branch
  phase_bound: 0.25
```

- `local_residual`: `base + activation(up(GELU(down([content,base]))))`,
  zero-init up;
- `content_gate`: a zero-init content projection followed by either legacy
  `1+tanh(raw)` or `2*sigmoid(raw+bias)` scaling. For `scaled_sigmoid`,
  `gate_init` is the initial multiplier and must lie in `(0,2)`.
- `phase_rotation`: only valid for additive `pair_normalized`. A local content
  network predicts `phase_bound*raw` radians and rotates each
  already-normalized Cartesian pair. It cannot alter pair radius. `target`
  selects Q, K, or both; `coupling=shared` reuses one output head, while
  `shared_trunk_separate_readouts` gives Q and K distinct zero-initialized
  phase heads over one content trunk.

- `adaptive_gain`: a zero-initialized scalar head gives
  `gain=exp(raw)=1` and scales each token/head after Q/K RMSNorm;
- `additive_phase`: a zero-initialized `D/2` content head changes only the
  phase of the established additive Fourier carrier;
- `rope_phase`: a zero-initialized `D/2` content head changes the actual RoPE
  rotation after Q/K RMSNorm.

These new actuators are token-local and KV-cache compatible. Their final
projections initialize to zero and have no second zero gate.
For canonical AddRoPE, conditioning acts on amplitude and phase latents before
cosine/sine synthesis; it never perturbs pair coordinates independently.

### Anchor-relative carrier hypernetwork

`carrier_hypernetwork` modulates either additive `amplitude_phase` AddRoPE or
rotary `phase`/`scaled_phase` while remaining on the Q/K path used by fused
SDPA. It consumes normalized dedicated content, the raw configured
Fourier/scalar position basis, or their concatenation. `linear`, `silu_mlp`,
and `swiglu_mlp` networks are available. With
`input_normalization=modality_rms`, content and position are independently
scaled to unit RMS over their feature dimensions before concatenation.
`learnable_input_gains=true` then applies one learned scalar per present
modality, initialized to one; it is valid only with modality-wise RMS.

Every final projection weight and bias is zero-initialized. There is no output
RMSNorm on the predicted deltas, because normalization would magnify a tiny
departure and defeat the exact null. Additive channels retain method-aware
RMSNorm after carrier addition; rotary channels retain Q/K RMSNorm before
rotation.

The legacy additive modulation mode uses `components=log_gain_phase` on an
established additive anchor `(a, phi)`:

```text
a'   = a * exp_ste(delta_log_gain)
phi' = phi + delta_phase
```

The gauge-free dynamic replacement uses `components=amplitude_phase`,
`learn_amplitude=false`, and `learn_phase=false`. It supports two amplitude
parameterizations. The positive softplus form is:

```text
a(x,p)   = softplus(inv_softplus(amplitude_init) + raw_amplitude(x,p))
phi(x,p) = raw_phase(x,p)
```

The raw signed form is:

```text
a(x,p)   = amplitude_init + raw_scale(x,p)
phi(x,p) = raw_phase(x,p)
```

The static mapper is absent. Zero-initialized final projections therefore
recover exactly `amplitude_init * cis(omega*p)`, while the hypernetwork becomes
the sole learned source of amplitude and phase. The unit-anchor configuration
sets `amplitude_init=1`, `amplitude_parameterization=signed`, and composes

```text
addend(x,p) = (1 + predicted_scale(x,p))
              * cis(omega*p + predicted_phase(x,p))
```

Both predicted terms start at exactly zero. This is a raw polar actuator around
a unit AddRoPE carrier: it is not the mapped-0.3 model (whose position mapper
learns amplitude and phase), and it is not the positive softplus replacement
(whose amplitude cannot cross zero). Setting `amplitude_init=1` with softplus
also preserves the exact unit anchor, but changes the local amplitude derivative
and enforces positivity.

Additional additive component modes retain zero-initialized exact anchors:

- `amplitude` and `phase` isolate one dynamic polar component while the other
  remains fixed;
- `cartesian` predicts a complex residual `(1+u)+iv` and multiplies it by the
  base carrier, so `u=v=0` is exactly `cis(omega*p)`;
- `frequency_phase` composes
  `cis((1+delta_frequency)*(omega*p+delta_phase))`; it may pair those dynamic
  angular components with a directly learned static amplitude vector;
- `amplitude_phase_frequency` makes all three polar components dynamic using
  the same frequency-multiplier formula.

Dynamic amplitude or phase cannot overlap a learned static parameter for the
same component. This preserves the gauge correction while allowing a static
amplitude vector to be compared with dynamic frequency and phase.

The first matched unit-anchor screen keeps per-head outputs, a shared trunk with
separate Q/K readouts, one shared normalized content projection, SDPA, and
method-aware add-then-RMS normalization fixed. Its ten cells are:

- standard RoPE, mapped-0.3 AddRoPE, and direct unit AddRoPE controls;
- position, content, and content+position inputs with a linear hypernetwork;
- the same three inputs with a SiLU MLP;
- content+position with a SwiGLU MLP.

For asymmetric additive experiments, `target=q|k` together with
`static_complement=true` gives the inactive branch its own directly learned
canonical AddRoPE amplitude and phase. The dynamic branch has no static
amplitude/phase parameters, and the static branch has no hypernetwork readout,
so each branch retains exactly one parameter source and no gauge is introduced.

For rotary channels, `components=phase` predicts only `delta_phase`.
`components=log_gain_phase` additionally applies pairwise radial scale
`exp_ste(delta_log_gain)`, intentionally making the operation scaled rotary
rather than strictly norm-preserving RoPE. `exp_ste` is exponential in the
forward pass and has identity derivative in the backward pass.

`target=q|k` leaves the other branch exactly on its static anchor.
`coupling=shared` reuses one complete network and, for `target=both`, requires
`position_content_coupling=shared`.
`shared_trunk_separate_readouts` shares nonlinear features but has independent
zero heads. `separate` builds independent Q/K networks. Hypernetwork head
coupling is independent of the carrier mapper and may broadcast one output
over heads or learn per-head outputs.

Evaluation diagnostics use one real validation sequence for conditioned
channels and report branch/QK RMS ratios, p95 ratios, content-to-combined
cosines, and bounded additive gains. Static zero-content summaries remain
available when no example is supplied.

## Local result collection

`position_results.py` scans one or more output roots for `metrics.jsonl`.
Runs may be selected with repeatable `--run-glob`, `--run-regex`, and
`--exclude-glob` filters. `--step-min`, `--step-max`, and `--every` select a
history interval; omit `--history` to reduce each run to its final and best
retained evaluations.

Post-processing presets are:

- `core`: evaluation losses and perplexity;
- `qk-health`: core metrics plus cross-layer branch maxima, contribution ratios,
  content/combined cosine, rotary scale, and additive gain summaries;
- `hyper-health`: Q/K health plus cross-layer gain/phase delta RMS and p95
  magnitudes and the maximum effective exponential gain;
- `all`: every final numeric metric.

Additional metric globs may be supplied with repeatable `--metric`. Output
formats are terminal table, Markdown, CSV, JSON, and JSONL; `--output` writes
the rendered result to a file.

```bash
# Final Q/K-health summary for one family.
python position_results.py model-output/position_bias_phase4_safe_conditioning \
  --run-glob 'phase4-safe-*' --preset qk-health

# Gate trajectories from steps 1000 through 3000 as CSV.
python position_results.py model-output \
  --run-glob '*gate*' --history --step-min 1000 --step-max 3000 \
  --every 1000 --metric 'position/*/qk/additive_gain_*' --format csv
```

## Relative logit channel and Inkling

Static logit config uses the same input/mapper/head-coupling fields:

```yaml
logit_bias:
  enabled: true
  application: logit_bias
  geometry: scalar_curve
  input: {kind: frozen_fourier, basis_dim: null, theta: null, scalars: []}
  mapper: {kind: linear, residual: false, rank: 32, hidden_dim: 128}
  conditioning:
    kind: none
    pair_rank: 16
    position_mode: relative_only
    num_profiles: 8
    router_hidden_dim: 64
    profile_init_std: 0.02
    num_frequencies: 16
    gate_init: 0.0
  head_coupling: per_head_independent
```

Conditioning values:

- `none`: existing position-only `[H,R]` curve;
- `inkling_table`: query-routed bank of bounded learned table profiles;
- `inkling_cosnet`: query-routed bank of bounded learned cosine functions;
- `pairwise_low_rank`: factorized query-content, key-content, and relative-offset
  interaction, optionally enriched with absolute Fourier position.

Inkling returns `[B,H,Q,R]`; Flex score modification indexes the active query
and distance without materializing `[B,H,Q,K]`. The profile gate initializes to
zero while profiles initialize with small symmetry-breaking values. This keeps
the baseline exact and gives the gate an immediate gradient. Diagnostics report
routing entropy, maximum routing probability, and gate magnitude.

Convenience presets `inkling_table` and `inkling_cosnet` now work and force
FlexAttention.

Pairwise low-rank conditioning computes

```text
b(i,j) = b_static(i-j)
       + g_h / sqrt(r) · Σ_m Q_m(q_i, φ(i)) K_m(k_j, φ(j)) D_m(φ(i-j))
```

`Q`, `K`, and `D` are independently projected and RMS-normalized factor
vectors. The per-head outer gate `g_h` is unconstrained and initializes to zero:
the static linear-logit anchor is therefore exact, while the gate immediately
receives a gradient. No `tanh` is applied to the factors.
The content factors consume the dedicated `c_q` and `c_k` projections rather
than the attention Q/K vectors.

`position_mode` controls which absolute terms are available:

- `relative_only`: content plus `φ(i-j)` only; shifting the same content pattern
  preserves the interaction;
- `query_absolute`: additionally adds `φ(i)` to the query factor;
- `full_absolute`: additionally adds both `φ(i)` and `φ(j)`.

Here Fourier features are not inherently relative or absolute: `φ(i-j)` is a
translation-invariant relative representation, while `φ(i)` and `φ(j)` encode
absolute indices. The factorization is evaluated directly inside FlexAttention
without materializing a dense `[B,H,Q,K]` bias tensor.

## Residual-stream position

```yaml
residual_stream:
  enabled: false
  placement: input        # input | per_layer | both
  source: position_basis  # position_basis | learned_absolute
  input: {kind: frozen_fourier, basis_dim: null, theta: null, scalars: []}
  mapper: {kind: identity, residual: false, rank: 32, hidden_dim: 128}
  gate_init: 0.0
  layer_shared: false
```

Controls:

- standard sinusoidal: `position_basis + identity + gate_init=1`;
- functional sinusoidal: smaller `basis_dim` + linear/MLP mapper;
- learned absolute: `source=learned_absolute`;
- zero-init reinjection: `placement=per_layer`, `gate_init=0`;
- shared or layer-specific reinjection modules.

Writes occur in model dimension after `in_proj` and/or before each Transformer
block.

Set top-level `use_rope: false` to test these as the sole explicit positional
mechanism. With all position channels disabled, this gives the no-explicit-PE
control; the causal mask still exposes visible-context length. Rotary Q/K
channels remain invalid when RoPE is disabled, while additive Q/K and
logit-only configurations are allowed.

## Attention-output position writes

```yaml
attention_write:
  enabled: false
  mode: key_position      # key_position | relative_offset | query_position
  input: {kind: frozen_fourier, basis_dim: null, theta: null, scalars: []}
  mapper: {kind: identity, residual: false, rank: 32, hidden_dim: 128}
  head_coupling: per_head_independent
  gate_init: 0.0
```

Positional values are appended to V, so SDPA/Flex computes their attended
summary with the exact same weights as content. `key_position` writes where
retrieved values came from. `relative_offset` uses the Fourier difference
identity after attention:

```text
cos(i-j) = cos(i)cos(j) + sin(i)sin(j)
sin(i-j) = sin(i)cos(j) - cos(i)sin(j)
```

Relative mode therefore requires pure paired Fourier values (no scalar inputs).
The mapped summary is merged to model dimension and zero-gated into the
attention output.

`query_position` is deliberately different: it does not append anything to V
or use attention weights. It writes the literal query index after `to_out`:

```text
out_i = O(attn_i) + W_zero(position_i)
```

The final projection initializes to zero and has no scalar gate, so the write
is an exact null while the projection receives a gradient on the first step.

## Parameter-matched controls

`ff_hidden_dim` overrides `dim*ff_mult`. Dry runs print a suggested aligned
GeGLU width that spends the active position-module parameter budget in the
baseline FFN.

## Legacy mapping

| v1 | canonical v2 |
| --- | --- |
| `apply=add` | `additive/free` |
| `apply=phase_residual` | `rotary/phase` |
| `feature_map=identity` | `mapper=identity`, non-residual |
| `feature_map=add_rope` | `mapper=euclidean_affine`, non-residual |
| `feature_map=linear` | `mapper=linear`, non-residual |
| `feature_map=low_rank` | `mapper=low_rank`, residual |
| `feature_map=bottleneck_mlp` | `mapper=bottleneck_mlp`, residual |
| `feature_map=mlp` | `mapper=mlp`, residual |
| `sharing=shared_head` | `head_coupling=shared_head` |
| `sharing=per_head` | `head_coupling=per_head_independent` |
| `sharing=full_dim` | `head_coupling=per_head_joint` |
| implicit Q/K coupling | `shared` |

Legacy state-dict paths migrate for shared coupling. Incompatible shapes fail.
Optimizer state across renamed v1/v2 parameters is not guaranteed.

## Training and extrapolation lengths

The training configuration separates three length concepts:

- `training_length`: tokenized training-row length; `block_size` remains its
  backward-compatible alias;
- `model_position_extent`: the maximum sequence length allocated by positional
  channels and RoPE caches;
- `evaluation_lengths`: context lengths evaluated from the same validation
  token stream;
- `scalar_normalization_extent`: denominator horizon for normalized/log-position
  scalar features; defaults to `training_length`.

The training length is always inserted into `evaluation_lengths`.
`model_position_extent` and an explicit `rel_extent` must cover every requested
evaluation length. Longer validation examples are formed by contiguous
rechunking of the existing tokenized validation stream, so extrapolation does
not require another dataset download or tokenization cache. Keeping scalar
normalization tied to training length makes values beyond that length genuinely
out of distribution instead of renormalizing them into `[0, 1]`.

Native-v2 automatic run names end with a stable ten-hex hash over all canonical
Q/K, logit, residual-stream, and attention-write settings plus positional model
context (heads, width, theta, and extent). This prevents behaviorally different
experiments from sharing an output directory. Explicit `run_name` values remain
the caller's responsibility.

## Flex and compilation invariants

- Any logit channel forces Flex.
- Q/K, residual-stream, and attention-write-only runs may use SDPA.
- Flex remains behind `@torch.compiler.disable`.
- `attn_impl=flex` with `compile_fullgraph=true` is rejected by the trainer.
- The score-mod closure stays stable and reads a mutable static `[H,R]`,
  Inkling `[B,H,Q,R]`, or factorized pairwise bias representation.
- Pinned kernel options and outer `fullgraph=false` behavior remain unchanged.

## Experiment helpers

`launch_position_bias.sh` retains historical emitters and adds:

- `emit_v2_playground_variant`;
- `v2_qk_playground_json`;
- `v2_inkling_json`;
- `v2_pairwise_logit_json`.

They do not emit a sweep unless called by a future family. Existing completed
JSON and output artifacts are not rewritten.
