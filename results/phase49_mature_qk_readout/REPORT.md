# Phase 49: mature Q/K readout confirmation

| Arm | Final NLL | Total params | Position params | tokens/s |
|---|---:|---:|---:|---:|
| rope | 3.207156 | 153495936 | 0 | 216291 |
| scalar-qkpre | 3.169553 | 153495944 | 8 | 213683 |
| scalar-ffnmatch-r32 | 3.168988 | 154086280 | 8 | 214326 |
| qk-readout-r32 | 3.159413 | 154085768 | 589832 | 210457 |
| qk-readout-r128 | 3.158662 | 155855240 | 2359304 | 210442 |

## Paired contrasts on the new holdout

Negative deltas favor the first named arm.

| Contrast | Delta | Paired 95% interval |
|---|---:|---:|
| scalar-qkpre_minus_rope | -0.037603 | [-0.039125, -0.036063] |
| scalar-ffnmatch-r32_minus_scalar-qkpre | -0.000565 | [-0.002017, +0.000875] |
| qk-readout-r32_minus_rope | -0.047744 | [-0.049233, -0.046224] |
| qk-readout-r32_minus_scalar-qkpre | -0.010140 | [-0.011564, -0.008712] |
| qk-readout-r32_minus_scalar-ffnmatch-r32 | -0.009576 | [-0.010979, -0.008172] |
| qk-readout-r128_minus_scalar-qkpre | -0.010891 | [-0.012350, -0.009441] |
| qk-readout-r128_minus_qk-readout-r32 | -0.000750 | [-0.001978, +0.000486] |

## Controls and decisions

The FFN control differs from the rank-32 candidate's incremental parameter count by 512 parameters.
Median rank-128/rank-32 carrier-step ratio from 5k--95k: 1.029 (match: yes).
Rank-32 replication gate: pass.
Material matched-step rank-128 advantage: no.

This mature-horizon cohort uses a newly frozen holdout and exact paired data order and initialization, but only one training seed. Paired-example intervals do not estimate training-seed uncertainty.
