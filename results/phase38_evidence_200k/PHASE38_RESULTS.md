# Phase 38: pre-Q/K evidence strengthening

All contrasts are scalar pre-Q/K + fixed RoPE minus matched fixed RoPE.
Negative deltas favor the candidate.

| Pair | Baseline | Candidate | Delta | Paired bootstrap 95% CI |
| --- | ---: | ---: | ---: | ---: |
| scale-h1024d12-seed123 | 3.213906 | 3.173325 | -0.040581 | [-0.042082, -0.039120] |
| rep-h768d8-seed456 | 3.366275 | 3.321849 | -0.044426 | [-0.045967, -0.042912] |
| rep-h768d8-seed789 | 3.388076 | 3.329330 | -0.058745 | [-0.060313, -0.057204] |
| noqknorm-h768d8-seed123 | 3.394053 | 3.344530 | -0.049523 | [-0.051138, -0.047892] |

## Mature h768 replication

| Seed | Delta |
| ---: | ---: |
| 123 | -0.062831 |
| 456 | -0.044426 |
| 789 | -0.058745 |
| mean | -0.055334 |

## Predeclared gates

- mature_replication_pass: **PASS**
- scale_transfer_pass: **PASS**
- noqknorm_robustness_pass: **PASS**

Paired-example intervals estimate holdout precision within a seed. The three h768 seed deltas, not 3,072 examples pooled as independent, are the unit for training-seed robustness.
