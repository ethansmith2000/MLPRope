# Position configuration

MLPRope resolves positional settings to JSON-safe schema v2 before model
construction. The active runtime deliberately supports three core families:

1. fixed standard RoPE, or a NoPE control;
2. additive Fourier features on projected Q/K (AddRoPE);
3. a sinusoid added immediately before the Q/K projections; and
4. a one-shot input sinusoid with a narrow set of static adapter controls.

Learned AddRoPE amplitude/phase acts only on its additive carrier. The promoted
pre-Q/K carrier exposes only a scalar gate; isolated static low-rank pathways
remain available for development comparisons. RoPE is always standard and
fixed when enabled; otherwise the backbone is NoPE. The active design contract is
[`SINUSOID_INTERVENTION_POLICY.md`](SINUSOID_INTERVENTION_POLICY.md).

The research rationale and experiment plan are in
[`CONSOLIDATION_PLAN.md`](CONSOLIDATION_PLAN.md). Historical implementations
remain recoverable from git; their compact result reports remain in `results/`.

## Top-level configuration

```yaml
position_schema_version: 2
use_rope: true
post_position_qk_norm: false
qk_norm_mode: legacy_layernorm     # legacy_layernorm | method_aware_rms
position_content_dim: 64
position_content_coupling: separate # shared | separate

qk_preprojection: {enabled: false}
input_sinusoid: {enabled: false}
qk: {enabled: false}
logit_bias: {enabled: false}
```

`use_rope=true` always applies fixed split-half RoPE. Carrier interventions are
orthogonal: an enabled additive `qk` channel no longer disables RoPE.
`use_rope=false` selects NoPE whether or not a carrier is active. This behavior
differs from historical AddRoPE configs, which implicitly replaced RoPE.

`qk_norm_mode=method_aware_rms` composes an additive Q/K position signal with
the raw projections and then applies one learned per-head RMSNorm:

```text
q = RMSNorm(W_q x + e_q)
k = RMSNorm(W_k x + e_k)
```

The legacy mode normalizes the projections first and then adds position.
`post_position_qk_norm=true` is a parameter-free unit-RMS control and cannot be
combined with `method_aware_rms`.

## Fixed RoPE

Fixed RoPE uses the canonical inverse-frequency schedule

```text
omega_i = theta^(-i / (D/2))
q'_i(p) = R(omega_i p) q_i(p)
```

RoPE angles and cached sine/cosine tables remain fp32 even when model weights
are converted to bf16/fp16. There are no trainable frequency, phase-residual,
scale, clock, or warp parameters in this path. Historical learned-RoPE forms
raise a clear migration error.

## Pre-Q/K sinusoidal injection

```yaml
qk_preprojection:
  enabled: false
  mode: tied_scalar       # promoted; static development modes are listed below
  basis_dim: null       # resolves to model width; other widths are rejected
  theta: null           # resolves to rope_theta
  gate_init: 1.0
  learnable_gate: true
  rank: 32
  readout_lr_multiplier: 1.0
  compensate_readout_weight_decay: true
```

For the normalized block input `x_p` and frozen full-width Fourier vector
`z_p`, the original tied mode computes

```text
q_p = W_q(x_p + alpha z_p)
k_p = W_k(x_p + alpha z_p)
v_p = W_v x_p
```

Thus Q and K learn different reads through their existing projections, while V
and the residual stream are untouched. It may be used alone, with fixed RoPE,
or together with AddRoPE. The latter combination is supported for controlled
factorials even though Phase 30 found the two additive routes sub-additive.

`tied_scalar` is the promoted mode. The development-only
`low_rank_premap`, `low_rank_qk_replace`, `low_rank_qk_residual`, and
`low_rank_qk_shared_residual` modes test a shared position bottleneck with
either a model-space residual or shared/separate projected-space output maps.
The scalar gain and frozen Fourier table remain fp32 under module-wide
bf16/fp16 conversion. The completed carrier is cast to the activation dtype
before addition to `x`.

Phase-45 breadth controls add `dense_premap_residual`,
`dense_qk_shared_residual`, `dense_qk_residual`, and
`low_rank_qk_mlp_residual`. They distinguish a dense map before the existing
Q/K projections from shared versus separate native projected-space maps and a
nonlinear shared bottleneck. Every residual output is zero-initialized, so all
four begin exactly at `tied_scalar`; `rank` is the hidden width for the MLP
mode.

`readout_lr_multiplier` applies only to `shared_up`, `q_up`, and `k_up`; it
does not change the bottleneck or scalar-gate LR. When
`compensate_readout_weight_decay=true`, the readout group's AdamW coefficient
is divided by the same multiplier, preserving the per-step `lr * weight_decay`
shrinkage. Phase 47 used `sqrt(model_dim/rank)` to compare low-rank widths on a
dimension-aware function-update scale. It improved both endpoints but left a
`1.300x` median rank-128/rank-32 function-step ratio, narrowly missing the
predeclared `1.25x` ceiling; the setting is therefore a useful default rather
than an exact rank normalization. Phase 48's one-shot empirical rank-32
multiplier `6.367487` brought the ratio to `1.090x` and is the frozen efficient
candidate setting; no further local LR tuning is planned.

## One-shot input sinusoid control

```yaml
input_sinusoid:
  enabled: false
  mode: tied_scalar     # tied_scalar | low_rank_linear_residual |
                        # dense_linear_residual | residual_mlp |
                        # per_pair_amplitude
  basis_dim: null       # resolves to model width
  theta: null           # resolves to rope_theta
  gate_init: 1.0
  learnable_gate: true
  rank: 32
  hidden_dim: null      # resolves to model width for residual_mlp
```

The scalar control computes

```text
x_0 = in_proj(token_embedding) + beta z(p)
```

once, before the first Transformer block. The static adapter modes instead use

```text
low_rank_linear_residual: x_0 = in_proj(token) + beta z(p) + U V z(p)
dense_linear_residual:    x_0 = in_proj(token) + beta z(p) + D z(p)
residual_mlp:              x_0 = in_proj(token) + beta z(p) + W2 GELU(W1 z(p))
per_pair_amplitude:       x_0 = in_proj(token) + A z(p)
```

where the low-rank `U`, dense `D`, and MLP `W2` output maps are
zero-initialized, so their residuals begin exactly at the scalar parent, and
`A` has one signed coefficient shared by each cosine/sine frequency pair and
begins at identity. It may be combined with standard RoPE.
This remains a one-shot, position-only intervention and does not restore the
former generic residual/per-layer framework. Gates, pair amplitudes, and the
carrier table retain fp32 master values. Evidence and the distinction between
low-rank residual and replacement maps are summarized in
[`INPUT_SINUSOID_DESIGN.md`](INPUT_SINUSOID_DESIGN.md).

## Additive Q/K channel

```yaml
qk:
  enabled: false
  application: additive
  geometry: amplitude_phase # free | pair_normalized | amplitude_phase
  placement: before_rope    # before_rope | after_rope
  input:
    kind: frozen_fourier
    basis_dim: null
    theta: null
    scalars: []              # position | normalized_position | log_position
    normalization_extent: null
  mapper:
    kind: identity           # identity | euclidean_affine | linear |
                             # low_rank | bottleneck_mlp | mlp
    residual: false
    rank: 32
    hidden_dim: 128
  output:
    parameter_source: mapped # mapped | direct
    amplitude_init: 1.0
    amplitude_max: 1.5
    amplitude_parameterization: signed # signed | softplus | bounded_sigmoid
    learn_amplitude: true
    learn_phase: true
    phase_scale: 1.0
    additive_normalization: none       # none | rms
    additive_gain_init: 0.1
    additive_gain_max: 1.0
    learn_additive_gain: true
  conditioning:
    kind: none
    source: dedicated
    hidden_dim: 64
    target: both
    coupling: shared_trunk_separate_readouts
  qk_coupling: shared_trunk_separate_readouts
  head_coupling: per_head_independent
```

All enabled Q/K channels are additive. The geometries are:

| Geometry | Operation |
| --- | --- |
| `free` | arbitrary `[H,L,D]` positional addend |
| `pair_normalized` | arbitrary paired coordinates normalized to a fixed radius |
| `amplitude_phase` | `a_i(p)[cos(omega_i p+phi_i), sin(omega_i p+phi_i)]` |

For `amplitude_phase`, `parameter_source=mapped` predicts static position-only
amplitude and phase from the configured basis. `parameter_source=direct` uses
only per-head/per-frequency parameters. `learn_amplitude` and `learn_phase`
independently select which static components are trainable. Setting both false
produces the exact fixed AddRoPE carrier at `amplitude_init`.

`placement=before_rope` is the historical runtime ordering:

```text
q_p = R_p RMSNorm(W_q x_p + e_q(p)).
```

`placement=after_rope` instead computes:

```text
q_p = RMSNorm(R_p W_q x_p + e_q(p)).
```

It requires standard RoPE and `method_aware_rms`. The scalar RMS denominator
commutes with an orthogonal rotation, but the learned coordinatewise RMSNorm
gain generally does not: `R_p Gamma != Gamma R_p` unless the gain is equal
within every rotary pair. The comparison therefore changes both whether RoPE
rotates the additive carrier and whether the learned gain occurs before or
after rotation. Applying `R_p` to a canonical matched-frequency carrier does
advance that carrier from phase `omega*p` to `2*omega*p`; this remains a useful
idealized account of one effect, not a complete isolation of ordering.

`additive_normalization=rms` normalizes the position branch per token/head and
then applies a bounded learned gain. It controls branch magnitude without
changing the carrier phase alignment.

### Inputs, mappers, and coupling

The only basis kind is `frozen_fourier`, with interleaved layout

```text
[cos_0, sin_0, cos_1, sin_1, ...].
```

This differs from split-half RoPE pairing. Optional scalar features append to
the mapper input. Identity, Euclidean-affine, and residual mappers require
matching input/output widths.

Q/K coupling values are `shared`, `shared_trunk_separate_readouts`, and
`separate`. Head coupling values are `shared_head`,
`per_head_independent`, and `per_head_joint`.

### Pointwise content conditioning

Conditioned AddRoPE reads a dedicated low-rank, unit-RMS projection of the
block-normalized residual. Q and K content projections may be shared or
separate. The conditioning is token-local, so it does not read future tokens.

Supported conditioning kinds are:

- `local_residual`: zero-initialized local residual on the positional output;
- `content_gate`: tokenwise bounded scaling;
- `phase_rotation`: content-dependent rotation of `pair_normalized` pairs;
- `additive_phase`: content-dependent phase of an additive carrier; and
- `carrier_hypernetwork`: anchor-relative additive carrier deltas.

The carrier hypernetwork is valid only for additive `amplitude_phase`. Its
`input_mode` is `content`, `position`, or `content_position`; its network is
`linear`, `silu_mlp`, or `swiglu_mlp`. Final readouts initialize to zero, so
the configured static carrier is an exact anchor. Common component sets include
`amplitude`, `phase`, `amplitude_phase`, and `cartesian`; the complete validated
set lives in `position/config.py`.

EMA and other recurrent/scan conditioning were removed. Content-conditioned
frequency multipliers were also removed because their phase sensitivity grows
with absolute position. The retained dynamic path changes additive carrier
amplitude/phase without changing fixed RoPE itself.

## Intervention optimizer diagnostics

`position_lr_multiplier` scales the learning rate of position-specific
parameters relative to the base optimizer LR; it defaults to `1.0`. Two fields
control sparse read-only optimizer-health sampling:

```yaml
intervention_optimizer_warmup_steps: [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
intervention_optimizer_log_every: 1000  # null disables periodic samples
```

When a learned carrier intervention is active, samples are appended to
`intervention_optimization.jsonl` and sent to the configured tracker. Metrics
are separated into `pre_qk_sinusoid_adapter`, `additive_qk_sinusoid`, and
`position_content_projection`. They include raw/clipped gradient statistics,
Adam moment diagnostics, realized parameter movement, update/gradient
alignment, and carrier-function movement.
Sampling does not alter the optimizer or forward pass.

## Legacy and removed fields

Legacy v1 additive configs still upgrade (`apply=add`). Enabled
`apply=phase_residual` configs now fail with a migration message. Historical
rotary output-scale keys remain parseable inside disabled archival config
shapes but do not affect fixed RoPE.

Historical top-level `rope_frequency` and `rope_frequency_mode` are accepted
only when fixed, then removed from the resolved configuration. Any learned form
fails with a migration message.

Historical pre-Q/K frequency, smooth-amplitude, split-Q/K, and phase modes were
removed after the Phase 33--37 confirmations. An enabled block using one raises
a migration error; a disabled archival block is accepted and canonicalized to
`tied_scalar` because it has no model effect. Compatibility-only
`smooth_rank`, `frequency`, and `frequency_lr_multiplier` fields are discarded
from the resolved configuration. Historical configs and reports remain in the
repository, while implementations remain in git history.

The former `conditioning.kind=adaptive_gain` is also rejected: it multiplied
the complete Q/K tensors rather than transforming the sinusoidal carrier.

The following historical top-level blocks accept only their disabled form:

```yaml
logit_bias: {enabled: false}
residual_stream: {enabled: false}
attention_write: {enabled: false}
rotary_clock: {enabled: false}
position_gain: {enabled: false}
```

This keeps old resolved configs understandable without retaining dormant model
machinery.

## Training and extrapolation lengths

- `training_length` is the tokenized training-row length (`block_size` remains
  its alias).
- `model_position_extent` allocates RoPE and position caches.
- `evaluation_lengths` selects validation context lengths.
- `scalar_normalization_extent` is the normalization horizon for scalar
  position features.

The model extent and explicit relative extent must cover every requested
evaluation length. Longer validation examples are formed by contiguous
rechunking of the cached validation token stream.

## Long-run storage and provenance

The canonical OpenWebText cache is
`/workspace/data/tokenized/openwebtext_gpt2_bs1024`. The loader accepts and
strictly checks both the historical MLPRope fingerprint filename and the
shared `.tokenized-cache-manifest.json` filename. Every run copies all
available cache manifests, their SHA-256 hashes, split counts, and dataset
fingerprints into `run_provenance.json`.

Runs default to `checkpointing_steps: null` and `save_final_model: false`.
Enable recovery state only when interruption recovery is worth its storage
cost; a typical explicitly resumable long run uses:

```yaml
checkpointing_steps: 5000
checkpoint_keep_latest: 1
checkpoint_milestones: []
resume_from_checkpoint: auto
save_evaluation_details: true
```

A checkpoint becomes resumable only after `accelerator.save_state` completes
and `CHECKPOINT_COMPLETE.json` is written. Only then may older states be
pruned. `checkpoint_keep_latest` retains that many newest complete states;
steps named in `checkpoint_milestones` are retained in addition. A policy file
makes automatic resume ignore a newer partial directory after an interrupted
save. Null `checkpoint_keep_latest` preserves the historical keep-all policy.
The active default is `1`, so merely enabling checkpointing cannot silently
accumulate every periodic state. Delete the final recovery directory after a
successful run and its declared downstream analyses unless resume remains a
named need. Save final weights only for a specified evaluation or intervention;
configs, metrics, evaluation details, provenance, logs, and summaries remain
the durable evidence layer.

`run_provenance.json` appends one launch record on every restart. It includes
the exact resolved config and its hash, source commit and dirty-tree listing,
parameter counts, focused Python/package/CUDA versions, visible GPU identity,
and dataset identity. Periodic development evaluations save their per-example
losses separately from the final holdout details, enabling paired confidence
intervals at every evaluation milestone.

## Verification

```bash
/venv/main/bin/python -m unittest \
  test_position_channels test_position_dynamics \
  test_position_playground test_position_results

gpu-claim run --owner mlprope --job position-consolidated-smoke --wait -- \
  /venv/main/bin/python -u scripts/position_v2_cuda_smoke.py
```

FlexAttention remains an optional raw backend. Its compiled helper is isolated
from the outer model compile; `attn_impl=flex` with
`compile_fullgraph=true` is rejected.
