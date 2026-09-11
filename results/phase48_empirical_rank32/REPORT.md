# Phase 48: one-shot empirical rank-32 calibration

Empirical rank 32 used readout LR multiplier `6.367487` and reached final holdout NLL `3.439380`.

## Paired contrasts

Negative deltas favor empirical rank 32.

| Contrast | Delta | Paired 95% interval |
|---|---:|---:|
| empirical-r32_minus_calibrated-r128 | +0.001058 | [+0.000150, +0.001997] |
| empirical-r32_minus_theoretical-r32 | -0.001889 | [-0.002518, -0.001272] |
| empirical-r32_minus_scalar-parent | -0.015698 | [-0.017018, -0.014403] |
| empirical-r32_minus_dense-separate | -0.001842 | [-0.002905, -0.000792] |

## Function-step calibration

Median rank-128/rank-32 carrier-step ratio through step 64: 1.322.
Median ratio from step 1k through 19k: 1.090 (predeclared 0.8--1.25 match: yes).

The matched rank-32 arm closed 73.0% of the original equal-LR rank gap.

## Decision

The function-step gate passed. Calibrated rank 128 retained a small paired endpoint advantage, compatible with a modest capacity effect, but it is below the project's 0.003 scout materiality margin and does not establish a robust rank claim from one training seed.

The rank-128 comparator was frozen before selecting the empirical rank-32 multiplier. Runs share the schedule, initialization seed, data order, and holdout; paired-example intervals do not estimate training-seed uncertainty.
