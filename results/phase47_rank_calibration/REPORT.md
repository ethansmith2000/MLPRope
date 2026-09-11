# Phase 47: dimension-calibrated low-rank Q/K readouts

| Arm | Rank | Readout LR multiplier | Final NLL | Position params | tokens/s | Min clip ratio |
|---|---:|---:|---:|---:|---:|---:|
| calibrated-r32 | 32 | 4.898979 | 3.441269 | 589832 | 209310 | 0.4717 |
| calibrated-r128 | 128 | 2.449490 | 3.438322 | 2359304 | 210098 | 0.4715 |

## Paired contrasts

Negative deltas favor the first named arm.

| Contrast | Delta | Paired 95% interval |
|---|---:|---:|
| calibrated-r32_minus_scalar-parent | -0.013809 | [-0.015103, -0.012510] |
| calibrated-r128_minus_scalar-parent | -0.016755 | [-0.018067, -0.015468] |
| calibrated-r32_minus_calibrated-r128 | +0.002946 | [+0.001948, +0.003957] |
| calibrated-r32_minus_uncalibrated-r32 | -0.004408 | [-0.005554, -0.003274] |
| calibrated-r128_minus_uncalibrated-r128 | -0.003444 | [-0.004513, -0.002377] |
| calibrated-r32_minus_dense-separate | +0.000046 | [-0.000987, +0.001088] |
| calibrated-r128_minus_dense-separate | -0.002900 | [-0.004032, -0.001782] |

## Function-step calibration

Median rank-128/rank-32 carrier-step ratio through step 64: 1.703.
Median ratio from step 1k through 19k: 1.300 (predeclared 0.8--1.25 match: no).

## Decision

The predeclared carrier-function-step match failed; the rank-128 endpoint advantage cannot be attributed cleanly to representational rank.
Both calibrated arms remain valid optimization results, but the rank contrast is not a clean capacity ablation when the match fails.

All references use the identical schedule, paired initialization seed, data order, and holdout. Paired-example intervals do not estimate training-seed uncertainty.
