# Phase-20 learned RoPE frequency screen

## Question

Does learning the base multiplicative RoPE frequency schedule improve language
modeling while preserving norm, translation relativity, and fused SDPA?  This
screen changes frequencies only; it does not reopen arbitrary phase warps,
content-conditioned rotary phase, or scaled rotation.

## Fixed design

- Model: h768/d8, 8 heads, training context 1024.
- Horizon: 5,000 optimizer steps, batch 8, learning rate `3e-4`, AdamW betas
  `0.9/0.98`.
- Seeds: 123, 456, 789.  Each arm shares the data seed and
  `paired_initialization_seed`.
- Development check: 25 batches at step 5,000.
- Final holdout: 256 batches starting at batch 2,048, evaluated independently
  at contexts 1024, 2048, and 4096.
- Multiplicative Q/K frequencies are shared between Q and K.  Learned
  frequencies use `omega = omega_base * exp(log_delta)` with `log_delta=0` at
  initialization.  Angles and trigonometry are evaluated in fp32.
- All arms use fused SDPA, method-aware Q/K RMS normalization, and no other
  position channel.

## Arms

1. `fixed`: standard RoPE, shared across every layer and head.
2. `layer-shared`: each layer learns one frequency schedule shared by heads
   (384 learned scalars total).
3. `layer-head`: each layer and head learns its own schedule (3,072 learned
   scalars total).

The layer-shared arm separates depth adaptation from head-specific adaptation.

## Decision rule

This is a screen, not a promotion result.  The primary hypothesis is improvement
at the 1024-token training context.  Promote `layer-head` to a 30k paired run
only if its mean 1024 loss improves by at least 0.01 with consistent seed signs.
Compare `layer-head` with `layer-shared` to decide whether the gain actually
requires head-specific spectra.  Report throughput separately.

Inspect per-layer multiplier extrema, head dispersion, and minimum log-frequency
spacing.  Frequency collapse or explosive multipliers count against promotion.

## Post-run scope clarification

The 2048/4096 evaluations were collected in the original run, but length
extrapolation is not a current project hypothesis.  They are retained as
incidental diagnostics and excluded from the promotion decision.  The locked
training configurations and raw results are unchanged; this clarification was
made before interpreting the screen outcome.
