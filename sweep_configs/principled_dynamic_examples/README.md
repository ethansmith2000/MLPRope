# Principled dynamic-position examples

These are loadable implementation examples, not a locked sweep and not an
authorization to launch training.

- `qk-preprojection-only-h768d8.json` uses the pre-W_q/W_k sinusoid as the sole
  explicit positional mechanism.
- `rotary-clock-pointwise-h768d8.json` is the minimal content-dependent clock.
- `rotary-clock-causal-conv-h768d8.json` replaces the pointwise controller with
  a four-token causal latent convolution.

For a controlled screen, pair each candidate by data order and
`paired_initialization_seed` with fixed RoPE and use the repository's disjoint
final holdout. Do not compare the Q/K-preprojection cell and the rotary-clock
cell as though they differ on one axis: they use different injection geometry.
