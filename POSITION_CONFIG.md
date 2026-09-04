# Position configuration

MLPRope resolves positional settings to JSON-safe schema v2 before model
construction. The active runtime deliberately supports three core families:

1. fixed standard RoPE, or a NoPE control;
2. additive Fourier features on projected Q/K (AddRoPE); and
3. a sinusoid added immediately before the Q/K projections.

Learned amplitude, phase, or frequency acts only on a sinusoidal carrier. RoPE
is always standard and fixed when enabled; otherwise the backbone is NoPE. The
active design contract is
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

## Carrier-only static frequency bank

The research frequency schema is available only under
`qk_preprojection.frequency`:

```yaml
qk_preprojection:
  frequency:
    mode: fixed                 # fixed | learned_log | learned_horizon
    reference_length: null      # resolves to the training sequence length
    max_grad_norm: 1.0          # clips this bank independently
```

For each Fourier pair, `learned_log` uses

```text
omega_i = omega_i^0 exp(delta_i),       delta_i = 0 initially,
```

while `learned_horizon` uses

```text
omega_i = omega_i^0 + rho_i / L,        rho_i = 0 initially,
theta_i(p) = p omega_i^0 + (p/L) rho_i.
```

Here `L` is the configured reference length. The horizon coordinates express
the parameter directly as an endpoint phase displacement, so the multiplier
in its phase gradient is `p/L <= 1` on the training context instead of `p`.
This is a forward parameterization, not a custom backward rule. Both learned
modes are exact fixed-frequency anchors at initialization.

An enabled learned carrier owns exactly one top-level fp32 vector of
`D_model/2` parameters. It is shared across Q/K, heads, and layers, excluded
from weight decay, and clipped independently from ordinary model gradients.
Diagnostics report frequency multipliers, endpoint phase displacement and
Jacobian, nonpositive frequencies, ordering violations, and actual carrier
movement. This extension preserves causality because the bank is static after
training and its forward value does not read the sequence.

## Pre-Q/K sinusoidal injection

```yaml
qk_preprojection:
  enabled: false
  mode: tied_scalar     # tied_scalar | tied_smooth_amplitude
  basis_dim: null       # resolves to model width; other widths are rejected
  theta: null           # resolves to rope_theta
  gate_init: 1.0
  learnable_gate: true
  smooth_rank: 4        # low-order DCT modes for smooth variants
  frequency:
    mode: fixed               # fixed | learned_log | learned_horizon
    reference_length: null
    max_grad_norm: 1.0
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

The active modes are deliberately small and tied across Q/K:

| Mode | Global gains | Pair amplitudes | Pair phases |
| --- | ---: | ---: | ---: |
| `tied_scalar` | one shared | fixed | fixed |
| `tied_smooth_amplitude` | one shared | tied smooth | fixed |

For Fourier pair `i`, the smooth mode applies

```text
A_i = g exp(delta_i) I,  sum_i delta_i = 0.
```

The `smooth_rank` coordinates use orthogonal, unit-RMS DCT-II modes over Fourier
pair index, which is uniformly spaced in log frequency. Omitting the constant
mode makes every column zero-mean: `g` controls global carrier strength and
`delta_i` only redistributes it across frequencies. Unit-RMS scaling makes a
coordinate's per-band functional effect independent of model width. At rank 4
the smooth mode has five parameters per layer including the global gain, and
its zero-coordinate initialization exactly equals `tied_scalar`.

The gain, log-amplitude coordinates, and DCT basis remain fp32 under module-wide
bf16/fp16 conversion. The completed carrier is cast to the activation dtype
before addition to `x`. `learnable_gate=false` freezes only the global gain;
the smooth amplitude coordinates remain trainable when that mode is selected.

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

Two training fields control sparse read-only optimizer-health sampling:

```yaml
intervention_optimizer_warmup_steps: [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
intervention_optimizer_log_every: 1000  # null disables periodic samples
```

When a learned carrier intervention is active, samples are appended to
`intervention_optimization.jsonl` and sent to the configured tracker. Metrics
are separated into `pre_qk_sinusoid_frequency`,
`pre_qk_sinusoid_adapter`, `additive_qk_sinusoid`, and
`position_content_projection`. They include raw/clipped gradient statistics,
Adam moment diagnostics, realized parameter movement, update/gradient
alignment, carrier-function movement for static amplitude/phase transforms,
and phase/function movement for carrier-frequency coordinates.
Sampling does not alter the optimizer or forward pass.

## Legacy and removed fields

Legacy v1 additive configs still upgrade (`apply=add`). Enabled
`apply=phase_residual` configs now fail with a migration message. Historical
rotary output-scale keys remain parseable inside disabled archival config
shapes but do not affect fixed RoPE.

Historical top-level `rope_frequency` and `rope_frequency_mode` are accepted
only when fixed, then removed from the resolved configuration. Any learned form
fails with a migration message directing frequency experiments to
`qk_preprojection.frequency`.

The historical pre-Q/K modes `split_scalar`, `split_pair_amplitude`,
`split_pair_polar`, `tied_smooth_polar`, and `split_smooth_polar` were removed
after their Phase 33/35 increments were null or below the practical gate. An
enabled block using one of them raises a migration error; a disabled archival
block is accepted and canonicalized to `tied_scalar` because it has no model
effect. The historical configs and result reports remain in the repository.

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
