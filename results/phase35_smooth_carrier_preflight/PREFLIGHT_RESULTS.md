# Phase 35 smooth-carrier preflight

All eight h768/d8, context-1024 arms completed 50 optimizer steps through
`gpu-claim` at source commit `c72a28a`. These are operational checks, not loss
comparisons: every arm starts from the same function and 50 steps is too short
to rank nearly coincident variants.

| Backbone and carrier | Target tok/s | Peak MiB | Final functional step RMS | Final descent cosine |
| --- | ---: | ---: | ---: | ---: |
| RoPE, fixed | 186,585 | 5,222 | 0.000014 | -0.012 |
| RoPE, smooth amplitude | 181,693 | 5,242 | 0.000045 | 0.307 |
| RoPE, smooth polar | 180,388 | 5,242 | 0.000062 | 0.422 |
| RoPE, split smooth polar | 177,446 | 5,362 | 0.000070 | 0.307 |
| NoPE, fixed | 185,327 | 5,222 | 0.000042 | 0.797 |
| NoPE, smooth amplitude | 183,940 | 5,242 | 0.000055 | 0.688 |
| NoPE, smooth polar | 183,519 | 5,242 | 0.000075 | 0.703 |
| NoPE, split smooth polar | 180,349 | 5,362 | 0.000070 | 0.499 |

The launch gate passed:

- all losses, gradients, Adam moments, parameter updates, and functional
  carrier movements were finite;
- the gradient clip ratio was exactly 1.0 in every sampled trace, so no arm was
  being rescued or distorted by global clipping;
- every intervention had nonzero coordinate updates and nonzero functional
  movement;
- tied variants had exactly zero Q/K profile difference, while both split
  variants developed nonzero Q/K amplitude and phase differences;
- the full CPU suite passed 124 tests (one CUDA-gated skip), and the compiled
  bf16 production-shaped forward/backward matrix passed before this preflight.

The small negative final descent cosine for the fixed-RoPE arm is one stochastic
trace sample, not an instability signal: its updates and function movement were
finite and active, and the other sampled steps are retained in the JSONL trace.
The 20k analysis therefore checks the whole trace rather than gating on one
cosine value.
