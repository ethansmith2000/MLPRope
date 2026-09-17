# Phase 57: recognized positional baselines

| Method | Final NLL | Delta vs RoPE (block-32 95% CI) | Delta vs scalar (block-32 95% CI) | Position params | ktok/s | Peak alloc GiB |
|---|---:|---:|---:|---:|---:|---:|
| scalar pre-Q/K + RoPE | 3.125383 | `-0.055942 [-0.057417, -0.054423]` | `+0.000000 [+0.000000, +0.000000]` | 8 | 216.3 | 14.34 |
| ALiBi | 3.137824 | `-0.043501 [-0.045356, -0.041637]` | `+0.012441 [+0.010749, +0.014335]` | 0 | 169.9 | 29.28 |
| fixed input sinusoid | 3.138505 | `-0.042821 [-0.044588, -0.041098]` | `+0.013121 [+0.010910, +0.015320]` | 0 | 220.2 | 13.95 |
| standard RoPE | 3.181325 | `+0.000000 [+0.000000, +0.000000]` | `+0.055942 [+0.054417, +0.057406]` | 0 | 218.7 | 13.94 |
| 25% partial RoPE | 3.197193 | `+0.015868 [+0.014180, +0.017639]` | `+0.071809 [+0.069884, +0.073819]` | 0 | 214.9 | 13.94 |
| learned absolute | 3.261448 | `+0.080122 [+0.077880, +0.082306]` | `+0.136064 [+0.133457, +0.138589]` | 786,432 | 220.8 | 13.95 |
| NoPE | 3.276223 | `+0.094898 [+0.092991, +0.096820]` | `+0.150840 [+0.148770, +0.152934]` | 0 | 221.5 | 13.94 |

## Primary confirmation

- Scalar pre-Q/K minus RoPE: `-0.055942` NLL.
- IID 95% interval: `[-0.057594, -0.054276]`.
- Contiguous-block-32 95% interval: `[-0.057417, -0.054423]`.
- All training metrics finite: **True**.

## Artifact closeout

Recovery cleanup reclaimed `12.02` GiB across `7` completed runs. No final weights were saved.

All arms use one training seed. Paired block intervals quantify final-holdout precision, not training-seed variability. The fixed input, partial-RoPE, and ALiBi cells are controlled implementations rather than claims of exact external recipe reproduction; ALiBi also uses a different attention kernel.
