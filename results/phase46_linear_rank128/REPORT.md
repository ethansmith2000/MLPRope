# Phase 46: linear rank-128 projected-space maps

| Arm | Final NLL | Position params | tokens/s |
|---|---:|---:|---:|
| linear-r128-shared | 3.442932 | 1572872 | 209221 |
| linear-r128-separate | 3.441766 | 2359304 | 207227 |

## Paired contrasts

Negative deltas favor the first named arm.

| Contrast | Delta | Paired 95% interval |
|---|---:|---:|
| linear-r128-shared_minus_scalar-parent | -0.012146 | [-0.013443, -0.010857] |
| linear-r128-separate_minus_scalar-parent | -0.013312 | [-0.014527, -0.012093] |
| linear-r128-shared_minus_linear-r128-separate | +0.001166 | [+0.000085, +0.002226] |
| linear-r128-shared_minus_dense-shared | +0.001982 | [+0.001027, +0.002945] |
| linear-r128-separate_minus_dense-separate | +0.000544 | [-0.000482, +0.001581] |
| linear-r128-separate_minus_nonlinear-r128-separate | +0.000300 | [-0.000420, +0.001021] |
| linear-r128-separate_minus_linear-r32-separate | -0.003910 | [-0.004953, -0.002859] |

## Optimizer-scale diagnostic

With the same Adam LR, rank 128's carrier-function step was 3.40x rank 32 through step 64 and 1.80x at the median sampled post-warmup step. The rank result therefore mixes capacity with function-space update scale.

All references use the identical schedule, paired initialization seed, data order, and holdout. Paired-example intervals do not estimate training-seed uncertainty.
