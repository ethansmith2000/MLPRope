# Phase 43: low-rank Q/K positional pathways

All runs use h768/d8, context 1024, sequence batch 32, seed 123, 20k updates, rank 32 where applicable, and the same disjoint 1,024-example holdout.

| Arm | Final NLL | tokens/s | Position params |
|---|---:|---:|---:|
| lowrank-premap | 3.449950 | 215141 | 393224 |
| lowrank-qk-replace | 3.460873 | 210651 | 589824 |
| lowrank-qk-residual | 3.445677 | 210445 | 589832 |
| rope-control | 3.498133 | 217382 | 0 |
| qkpre-control | 3.455078 | 213736 | 8 |

## Paired contrasts

Negative deltas favor the first named arm.

| Contrast | Delta | Paired 95% interval | Pass |
|---|---:|---:|:---:|
| lowrank-premap_minus_qkpre-control | -0.005127 | [-0.006231, -0.004029] | yes |
| lowrank-qk-replace_minus_rope-control | -0.037260 | [-0.038568, -0.035953] | yes |
| lowrank-qk-residual_minus_qkpre-control | -0.009401 | [-0.010502, -0.008308] | yes |
| lowrank-qk-residual_minus_lowrank-premap | -0.004274 | [-0.005138, -0.003404] | no |

Paired-example intervals measure holdout precision within one training seed, not training-seed uncertainty.
