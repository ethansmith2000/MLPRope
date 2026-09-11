# Phase 50: mature training-seed replication

## Final holdout NLL

| Arm | Seed 123 | Seed 456 | Seed 789 | Mean | Sample SD |
|---|---:|---:|---:|---:|---:|
| rope | 3.207156 | 3.199395 | 3.211910 | 3.206154 | 0.006318 |
| scalar-qkpre | 3.169553 | 3.165720 | 3.173226 | 3.169500 | 0.003753 |
| scalar-ffnmatch-r32 | 3.168988 | 3.166062 | 3.166395 | 3.167148 | 0.001602 |
| qk-readout-r32 | 3.159413 | 3.159596 | 3.162180 | 3.160396 | 0.001547 |

## Seed-level contrasts

Negative deltas favor the first named arm.

| Contrast | Seed 123 | Seed 456 | Seed 789 | Mean | Sample SD | Seed t interval |
|---|---:|---:|---:|---:|---:|---:|
| scalar-qkpre_minus_rope | -0.037603 | -0.033675 | -0.038684 | -0.036654 | 0.002636 | [-0.043203, -0.030105] |
| scalar-ffnmatch-r32_minus_scalar-qkpre | -0.000565 | +0.000342 | -0.006831 | -0.002351 | 0.003906 | [-0.012055, +0.007352] |
| qk-readout-r32_minus_rope | -0.047744 | -0.039798 | -0.049730 | -0.045757 | 0.005255 | [-0.058812, -0.032702] |
| qk-readout-r32_minus_scalar-qkpre | -0.010140 | -0.006124 | -0.011046 | -0.009103 | 0.002620 | [-0.015612, -0.002595] |
| qk-readout-r32_minus_scalar-ffnmatch-r32 | -0.009576 | -0.006466 | -0.004215 | -0.006752 | 0.002692 | [-0.013439, -0.000065] |

## Fresh-seed sensitivity

Seed 123 was used to select the rank-32 candidate. The table below
therefore isolates the two subsequently registered seeds.

| Contrast | Mean over seeds 456/789 | Both negative |
|---|---:|---:|
| scalar-qkpre_minus_rope | -0.036179 | yes |
| scalar-ffnmatch-r32_minus_scalar-qkpre | -0.003245 | no |
| qk-readout-r32_minus_rope | -0.044764 | yes |
| qk-readout-r32_minus_scalar-qkpre | -0.008585 | yes |
| qk-readout-r32_minus_scalar-ffnmatch-r32 | -0.005340 | yes |

## Predeclared decision

Rank 32 beats scalar in every seed with mean delta at most -0.003: yes.
Rank 32 beats the FFN control in every seed with mean delta at most -0.003: yes.
All diagnostics finite: yes.
Overall replication gate: pass.

The primary uncertainty unit is the training seed. The paired-example and contiguous-block intervals describe evaluation-sample precision within each seed; the three-seed t interval is descriptive and necessarily low-powered. Seed 123 selected this candidate, so the seeds-456/789 mean is also reported separately as fresh confirmation rather than obscured by the selected seed.
