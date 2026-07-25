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

qk: { ... }
logit_bias: { ... }
residual_stream: { ... }
attention_write: { ... }
```

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
    amplitude_init: 0.1
    amplitude_parameterization: signed
    learn_amplitude: true
    learn_phase: true
    phase_scale: 1.0
    scale_init: 1.0
    scale_parameterization: exp
  conditioning:
    kind: none
    hidden_dim: 64
    gate_init: 0.0
  qk_coupling: shared
  head_coupling: per_head_independent
```

### Applications and geometries

| Application | Geometry | Output and operation |
| --- | --- | --- |
| `additive` | `free` | free `[H,L,D]` addend; multiplicative RoPE is skipped |
| `additive` | `amplitude_phase` | canonical `a·cis(ωp+δ)` addend |
| `rotary` | `phase` | strict `R(θ+δ)` |
| `rotary` | `projected_phase` | predict a free pair-vector, project onto the RoPE tangent |
| `rotary` | `scaled_phase` | `s·R(θ+δ)`, with positive/exact-one initialization |

Invalid application/geometry pairs fail during config loading.

### Canonical amplitude+phase AddRoPE

For each pair:

```text
e_q(p) = a_q(p) · [cos(ωp + δ_q(p)), sin(ωp + δ_q(p))]
q'(p)  = q(p) + e_q(p)
```

K has its own amplitude/phase readouts when coupling permits. The default
amplitude is `0.1`, deliberately below the poor fixed-unit Phase-1c control.

For canonical amplitude+phase geometry, `learn_amplitude` and `learn_phase`
independently control the two output heads:

- both `false`: fixed Fourier carrier at `amplitude_init`;
- amplitude only: learned radial magnitude with zero learned phase delta;
- phase only: fixed radial magnitude with learned phase delta;
- both `true`: existing combined AddRoPE behavior.

Disabled component heads and the fully fixed carrier trunk are not instantiated,
so parameter counts reflect the active mechanism rather than dead parameters.
`signed` amplitude is unconstrained around that initialization; `softplus`
keeps it positive.

This is distinct from historical v1 `feature_map=add_rope`, which upgrades to
`euclidean_affine` over Fourier coordinates.

### Projected and scaled rotary

`projected_phase` predicts `[dx,dy]` and keeps only the tangent component:

```text
δ = <-sin(θ), cos(θ)> · [dx,dy]
```

The zero output remains exact RoPE. `scaled_phase` predicts phase plus pairwise
radial scale. `scale_parameterization=exp` uses `scale_init*exp(raw)`;
`linear` uses `scale_init+raw`.

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
| `linear` | full affine, Xavier/zero |
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

## Local Q/K content conditioning

```yaml
conditioning:
  kind: none               # none | local_residual | content_gate
  hidden_dim: 64
  gate_init: 0.0
```

- `local_residual`: `base + up(GELU(down([content,base])))`, zero-init up;
- `content_gate`: `base * (1 + gate_init + Linear(content))`, zero-init linear.

Conditioning consumes normalized Q/K locally at the same token. It does not
look across sequence positions, so the Q/K path remains KV-cache compatible.
For canonical AddRoPE, conditioning acts on amplitude and phase latents before
cosine/sine synthesis; it never perturbs pair coordinates independently.

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
  mode: key_position      # key_position | relative_offset
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
