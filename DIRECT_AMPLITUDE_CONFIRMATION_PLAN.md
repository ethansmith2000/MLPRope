# Phase 37 direct-amplitude confirmation

_Frozen 2026-09-05 before launch._

## Question

Does the favorable 20k rank-4 amplitude result survive a substantially longer
training horizon, and is the direct affine coordinate genuinely preferable to
the earlier exponential coordinate?

This is a parameterization confirmation, not a new architecture sweep. Carrier
frequency is excluded because Phase 36 found no practically material
incremental effect under healthy optimization.

## Arms

All three arms use standard fixed RoPE and the tied pre-Q/K sinusoidal carrier:

| Arm | Spectral amplitude |
| --- | --- |
| scalar parent | learned per-layer scalar gate only |
| exponential rank 4 | `a_i = g * exp((B c)_i)` |
| direct rank 4 | `a_i = g * (1 + (B c)_i)` |

Both rank-4 banks use the same zero-mean, unit-RMS DCT basis and initialize at
`c=0`, exactly matching the scalar parent at `g=1`. Q and K receive the same
carrier but retain separate projection matrices.

## Common protocol

- h768/d8, eight heads, context 1024, OpenWebText cache;
- batch 8, seed and paired-initialization seed 123;
- 200k optimizer steps, 200-step warmup, linear decay from step zero;
- AdamW at `3e-4`, position LR multiplier 1, no positional weight decay;
- method-aware QKNorm, bf16, compiled SDPA;
- 128-example development evaluation every 10k steps;
- disjoint 1,024-example final holdout beginning at validation batch 2,048;
- resumable checkpoint every 10k steps with only the latest retained;
- sparse optimizer/function diagnostics through the full run.

The resolved configs are frozen in
`sweep_configs/phase37_direct_amplitude_200k/`.

## Readout and gate

Primary contrast: direct rank-4 amplitude minus scalar. Secondary contrasts:
exponential minus scalar and direct minus exponential.

Direct amplitude earns seed replication only if, at 200k:

1. its loss improvement over scalar is at least `0.002` nats;
2. the paired-example 95% interval excludes zero;
3. the late development gap is not clearly collapsing;
4. amplitude factors, QKNorm mixtures, and optimization traces remain finite
   and interpretable.

Negative direct factors are permitted by the stated signed parameterization;
they are reported rather than silently clipped. Extra seeds are not launched
until this long-horizon gate is evaluated.
