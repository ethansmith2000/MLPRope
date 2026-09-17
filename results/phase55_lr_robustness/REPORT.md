# Phase 55: symmetric learning-rate robustness

| Peak LR | RoPE NLL | Scalar NLL | Scalar - RoPE | IID 95% interval | Block-32 95% interval |
|---:|---:|---:|---:|---:|---:|
| `1.5e-04` | 3.268286 | 3.239122 | -0.029164 | [-0.030680, -0.027668] | [-0.030885, -0.027493] |
| `3.0e-04` | 3.197119 | 3.160763 | -0.036355 | [-0.037844, -0.034809] | [-0.037887, -0.034763] |
| `6.0e-04` | 3.165110 | 3.120329 | -0.044780 | [-0.046389, -0.043157] | [-0.046504, -0.043034] |

## Frozen decisions

- Best RoPE: `6.0e-04`, NLL `3.165110`.
- Best scalar: `6.0e-04`, NLL `3.120329`.
- Best scalar minus best RoPE: `-0.044780`.
- Both outer-LR matched deltas negative: **True**.
- Strong robustness gate: **True**.

Seed 123 audits optimizer sensitivity around an already replicated central LR; it is not an additional training-seed replication or exhaustive tuning.
