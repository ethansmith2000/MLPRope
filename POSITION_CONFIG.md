# Position configuration (v2)

Canonical schema for MLPRope position channels after the foundation refactor.
Trainer and model code always resolve configs to **schema version 2** before
constructing modules. Legacy v1 JSON (Phase 1/1b/1c sweep files) still loads
through a strict upgrader.

See also: `position_embedding_experiments.md` (research design + historical
results), `EXPERIMENT_JOURNAL.md` (run log).

## Package layout

| Module | Role |
| --- | --- |
| `position/config.py` | v2 schema, validation, v1 upgrade, run tags |
| `position/basis.py` | Frozen interleaved Fourier basis |
| `position/mappers.py` | Identity / Euclidean affine / linear / low-rank / MLP |
| `position/channels.py` | Head pipeline, Q/K coupling, logit channel, builders |
| `position/rotary.py` | Split-half RoPE cache, phase compose, rotate |

`transformer.Attention` keeps projections, SDPA/Flex dispatch, and compile
guards. It consumes typed channel outputs (`QKPositionOutput`) rather than
mapper name strings.

## Canonical v2 schema

Top-level training config always stores:

- `position_schema_version: 2`
- `position_source_schema: 1|2` — whether the loaded JSON/preset was legacy

```yaml
qk:
  enabled: false
  application: rotary          # additive | rotary
  geometry: phase              # free | phase
  input:
    kind: frozen_fourier       # only shipped kind
    basis_dim: null            # resolved from head_coupling
    theta: null                # null inherits rope_theta
    scalars: []                # must stay empty for now
  mapper:
    kind: identity             # identity | euclidean_affine | linear |
                               # low_rank | bottleneck_mlp | mlp
    residual: false
    rank: 32
    hidden_dim: 128
  qk_coupling: shared          # shared | shared_trunk_separate_readouts | separate
  head_coupling: per_head_independent  # shared_head | per_head_independent | per_head_joint

logit_bias:
  enabled: false
  application: logit_bias
  geometry: scalar_curve
  input: { kind: frozen_fourier, basis_dim: null, theta: null, scalars: [] }
  mapper: { kind: identity, residual: false, rank: 32, hidden_dim: 128 }
  head_coupling: per_head_independent
```

### Shipped combinations

| Channel | Allowed `(application, geometry)` |
| --- | --- |
| Q/K | `(additive, free)`, `(rotary, phase)` |
| Logit | `(logit_bias, scalar_curve)` only |

Unsupported future modes (`learned_fourier`, projected-phase, content-aware,
canonical amplitude+phase AddRoPE, residual-stream PE, …) **fail explicitly**.

Mixing v1 keys (`feature_map`, `sharing`, `apply`, …) with v2 axis keys in one
channel dict is rejected.

## Tensor shapes and layouts

- Attention tensors remain `[batch, heads, sequence, head_dim]`.
- **Fourier feature basis** is interleaved `[cos_0, sin_0, cos_1, sin_1, …]`.
  This is **not** the same pairing as split-half RoPE (first half ↔ second half
  of each head).
- Q/K absolute basis extent: `max_seq_len` / `block_size`.
- Logit relative basis extent: `rel_extent` (defaults to `block_size`).
- Additive Q/K output: `[heads, seq, head_dim]` addends.
- Rotary/phase output: `[heads, seq, head_dim/2]` phase deltas; composed as
  `R(θ + δ)`.
- Logit output: `[heads, rel_extent]` scalar curves.

`basis_dim` resolves as:

- `shared_head` / `per_head_independent` → `head_dim`
- `per_head_joint` → `model_dim`

## Mapper initialization

| Mapper | Residual | Init |
| --- | --- | --- |
| `identity` | no | passthrough |
| `euclidean_affine` | no | `1+scale`, offset; both zero (legacy `add_rope`) |
| `linear` | no | Xavier weight, zero bias |
| `low_rank` / `bottleneck_mlp` / `mlp` | yes | Xavier down, zero up → identity at init |

`euclidean_affine` is the blog-style coordinate affine on Fourier features. It
is **not** canonical angular AddRoPE (`q + a·cis(ωp+δ)`).

## Q/K coupling

| Mode | Behavior at init |
| --- | --- |
| `shared` | One pipeline; Q and K receive the same tensor (v1 parity) |
| `shared_trunk_separate_readouts` | Shared trunk; additive uses identity-init dual `D→D` readouts; rotary uses dual zero `D→D/2` phase heads |
| `separate` | Deep-copied twin pipelines (no shared storage); equal outputs at init, independent grads |

`Attention` uses `qk_position.uses_multiplicative_rope` (true for `rotary`).

## Head coupling

| v2 | Legacy v1 `sharing` | Mapper layout |
| --- | --- | --- |
| `shared_head` | `shared_head` | one group, broadcast |
| `per_head_independent` | `per_head` | `H` independent groups |
| `per_head_joint` | `full_dim` | joint `model_dim` map, then reshape; phase/logit readouts still per-head |

## Exact v1 → v2 mapping

| v1 | v2 |
| --- | --- |
| `apply=add` | `application=additive`, `geometry=free` |
| `apply=phase_residual` | `application=rotary`, `geometry=phase` |
| `feature_map=identity` | `mapper.kind=identity`, `residual=false` |
| `feature_map=add_rope` | `mapper.kind=euclidean_affine`, `residual=false` |
| `feature_map=linear` | `mapper.kind=linear`, `residual=false` |
| `feature_map=low_rank` | `mapper.kind=low_rank`, `residual=true` |
| `feature_map=bottleneck_mlp` | `mapper.kind=bottleneck_mlp`, `residual=true` |
| `feature_map=mlp` | `mapper.kind=mlp`, `residual=true` |
| `sharing=shared_head` | `head_coupling=shared_head` |
| `sharing=per_head` | `head_coupling=per_head_independent` |
| `sharing=full_dim` | `head_coupling=per_head_joint` |
| `rank` / `mlp_hidden` | `mapper.rank` / `mapper.hidden_dim` |
| (implicit) | `qk_coupling=shared` |

`pos_variant` presets still expand to **logit-only** channels and force Flex.
Inkling presets raise `NotImplementedError`.

Source-schema v1 configs without an explicit `run_name` keep **legacy auto tags**
(`qk-add-identity-per_head-…`). Native v2 tags include application, geometry,
mapper, coupling, and head coupling.

## Checkpoint compatibility

`Transformer.load_state_dict` runs `adapt_legacy_position_state_dict`:

- `.features.*` → `.pipeline.mapper.*`
- Q/K `.output_weight|bias` → `.phase_head.weight|bias`
- Logit `.readout|_bias` → `.scalar_head.weight|bias`

A warning lists remapped keys. Incompatible shapes fail rather than silently
skip. **Optimizer-state resume across the v1→v2 parameter rename is not
guaranteed**; model-weight resume for shared coupling is supported. Prefer
fresh optimizer state when loading a renamed checkpoint into v2 modules.

## Attention / Flex invariants

- Logit bias requires `attn_impl=flex` (auto-forced by the trainer).
- Q/K-only channels may use SDPA.
- FlexAttention stays behind `@torch.compiler.disable` with pinned kernel
  options; do not fold it into the outer compiled graph.

## Examples

Legacy (still valid in JSON):

```json
{
  "qk": {
    "enabled": true,
    "feature_map": "identity",
    "sharing": "per_head",
    "apply": "add",
    "rank": 32,
    "mlp_hidden": 128
  },
  "logit_bias": { "enabled": false }
}
```

Native v2 with separate Q/K trunks:

```json
{
  "qk": {
    "enabled": true,
    "application": "rotary",
    "geometry": "phase",
    "input": {
      "kind": "frozen_fourier",
      "basis_dim": null,
      "theta": null,
      "scalars": []
    },
    "mapper": {
      "kind": "mlp",
      "residual": true,
      "rank": 32,
      "hidden_dim": 128
    },
    "qk_coupling": "separate",
    "head_coupling": "per_head_independent"
  },
  "logit_bias": { "enabled": false }
}
```

Launcher helper for future families: `emit_v2_channel_variant` in
`launch_position_bias.sh`. Do not rewrite completed Phase 1/1b/1c JSON files.
