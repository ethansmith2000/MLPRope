# Phase 56: learning-rate boundary check

| Peak LR | RoPE development NLL | Scalar development NLL | RoPE selection NLL | Scalar selection NLL |
|---:|---:|---:|---:|---:|
| `6.0e-04` | 3.099196 | 3.055308 | 3.165110 | 3.120329 |
| `1.2e-03` | 3.089271 | 3.035378 | 3.155743 | 3.101270 |

## Frozen decisions

- Boundary scalar minus RoPE: `-0.054474`.
- IID 95% interval: `[-0.056105, -0.052842]`.
- Block-32 95% interval: `[-0.056115, -0.052756]`.
- Both boundary runs finite: **True**.
- Selected common LR: `1.2e-03`.
- LR expansion stops here by protocol.

The selected LR is the best of a resource-bounded candidate set, not a claimed optimizer optimum. Adam betas, warmup, weight decay, clipping, and schedule were not tuned.
