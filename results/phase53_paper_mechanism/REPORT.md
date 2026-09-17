# Phase 53: paper mechanism measurements

All results use the same retained checkpoints and frozen holdout. Negative NLL deltas favor the first named method.

## Endpoint reproduction

| Arm | Seed 123 | Seed 456 | Seed 789 |
|---|---:|---:|---:|
| rope | 3.207211 | 3.199459 | 3.211929 |
| scalar-qkpre | 3.169648 | 3.165801 | 3.173230 |
| qk-readout-r32 | 3.159455 | 3.159647 | 3.162242 |

## NLL by target-position bin

| Target positions | Scalar - RoPE | Rank 32 - RoPE | Rank 32 - scalar |
|---|---:|---:|---:|
| 1–15 | -0.020661 | -0.029234 | -0.008573 |
| 16–31 | -0.037925 | -0.041807 | -0.003882 |
| 32–63 | -0.032344 | -0.041440 | -0.009096 |
| 64–127 | -0.034891 | -0.046718 | -0.011827 |
| 128–255 | -0.035821 | -0.044056 | -0.008236 |
| 256–511 | -0.036251 | -0.046043 | -0.009792 |
| 512–1023 | -0.037954 | -0.046786 | -0.008831 |

## Global attention geometry

Means first average sampled blocks, layers, and heads within each seed, then average seeds.

| Metric | RoPE | Scalar | Rank 32 |
|---|---:|---:|---:|
| entropy_normalized | 0.615625 | 0.575919 | 0.576567 |
| attended_distance_fraction | 0.217624 | 0.208577 | 0.209458 |
| first_token_mass | 0.015173 | 0.027305 | 0.029795 |
| distance_mass_0 | 0.064833 | 0.070827 | 0.070595 |
| distance_mass_1_3 | 0.198289 | 0.222270 | 0.222543 |
| distance_mass_256_plus | 0.113702 | 0.108775 | 0.109028 |

## Local positional-logit decomposition

| Metric | Scalar | Rank 32 |
|---|---:|---:|
| logit_cc_centered_rms | 1.344055 | 0.913064 |
| logit_cp_centered_rms | 0.344420 | 0.550010 |
| logit_pc_centered_rms | 0.451185 | 1.096074 |
| logit_pp_centered_rms | 0.318666 | 1.147567 |
| logit_position_total_centered_rms | 0.836647 | 1.644923 |
| logit_position_total_cosine | 0.823373 | 0.863760 |
| attention_kl_full_vs_content_only | 0.705940 | 1.913295 |

Maximum four-term logit reconstruction error: `4.005e-05`.

## Interpretation limits

- The training seed is the replication unit.
- Attention block intervals quantify sampling precision within a seed; layers and heads are not treated as independent replicates.
- Logit components use the trained hidden state and the common Q/K RMS denominator. Their sum is algebraically exact and is checked numerically against logits recomputed from full Q/K; the components remain descriptive local decompositions rather than independently runnable models.
- The attention sample is a predeclared evenly spaced subset of the same frozen holdout and is not a new corpus.
