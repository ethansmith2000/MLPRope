# Position configuration

MLPRope resolves positional settings to JSON-safe schema v2 before model
construction. The active runtime deliberately supports three families:

1. fixed standard RoPE, or a NoPE control;
2. additive Fourier features on projected Q/K (AddRoPE); and
3. a frozen sinusoid added immediately before the Q/K projections.

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
qk: {enabled: false}
logit_bias: {enabled: false}
```

`use_rope=true` applies fixed split-half RoPE unless an additive `qk` channel
is enabled. AddRoPE replaces multiplicative RoPE; this preserves the
historical experiment semantics. `use_rope=false` with both additive paths
disabled is the NoPE control.

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
scale, clock, or warp parameters in this path.

For compatibility with old resolved configs, the loader accepts only
`rope_frequency: {mode: fixed}`, `rotary_clock: {enabled: false}`, and
`position_gain: {enabled: false}`. Any active historical form raises a clear
migration error.

## Pre-Q/K sinusoidal injection

```yaml
qk_preprojection:
  enabled: false
  mode: tied_scalar     # tied_scalar | split_scalar |
                        # split_pair_amplitude | split_pair_polar
  basis_dim: null       # resolves to model width; other widths are rejected
  theta: null           # resolves to rope_theta
  gate_init: 1.0
  learnable_gate: true
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

The modes form a nested static adapter ladder:

| Mode | Global gains | Pair amplitudes | Pair phases |
| --- | ---: | ---: | ---: |
| `tied_scalar` | one shared | fixed | fixed |
| `split_scalar` | separate Q/K | fixed | fixed |
| `split_pair_amplitude` | separate Q/K | separate Q/K | fixed |
| `split_pair_polar` | separate Q/K | separate Q/K | separate Q/K |

For branch `b` and Fourier pair `i`, the pairwise modes apply

```text
A_i^b = g_b exp(delta_i^b) R(phi_i^b).
```

The spectral log-amplitude vector is represented with `P-1` coordinates in an
orthonormal zero-sum basis. Consequently
`sum_i delta_i=0` and the spectral factor has geometric mean one: `g_b`
controls global carrier strength without a scale gauge, while `delta_i`
redistributes it across frequencies. All modes initialize at `g=gate_init`,
`delta=0`, and `phi=0`; with the default `gate_init=1`, every rung is exactly
the original carrier.

The small gain, log-amplitude, phase, and zero-sum-basis state remains fp32
under module-wide bf16/fp16 conversion. The completed carrier is cast to the
activation dtype before addition to `x`. `learnable_gate=false` freezes only
the global gain; pair amplitude/phase parameters implied by the selected mode
remain trainable.

## Additive Q/K channel

```yaml
qk:
  enabled: false
  application: additive
  geometry: amplitude_phase # free | pair_normalized | amplitude_phase
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
- `adaptive_gain`: tokenwise Q/K gain initialized to one;
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

## Legacy and removed fields

Legacy v1 additive configs still upgrade (`apply=add`). Enabled
`apply=phase_residual` configs now fail with a migration message. Historical
rotary output-scale keys remain parseable inside disabled archival config
shapes but do not affect fixed RoPE.

The following top-level blocks accept only their disabled form:

```yaml
logit_bias: {enabled: false}
residual_stream: {enabled: false}
attention_write: {enabled: false}
rotary_clock: {enabled: false}
position_gain: {enabled: false}
rope_frequency: {mode: fixed}
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

Long runs should use:

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
