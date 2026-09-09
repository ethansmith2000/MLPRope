# Phase 42: batch-32 paper component matrix

All runs use h768/d8, context 1024, sequence batch 32, seed 123, 100k updates, and the same disjoint 1,024-example holdout.

| Arm | Final NLL | tokens/s | Position params | Gate range |
|---|---:|---:|---:|---:|
| nope | 3.262994 | 213314 | 0 | -- |
| rope | 3.183766 | 210747 | 0 | -- |
| qkpre-nope | 3.169792 | 215739 | 8 | [0.4473, 1.7033] |
| qkpre-rope | 3.146756 | 210370 | 8 | [0.0673, 0.2735] |
| qkpre-fixed-rope | 3.153220 | 209542 | 0 | [1.0000, 1.0000] |
| qkpre-global-rope | 3.147177 | 214108 | 1 | [0.1069, 0.1069] |
| qkpre-first-rope | 3.149452 | 208843 | 1 | [0.2759, 0.2759] |
| input-rope | 3.153390 | 209940 | 1 | -- |
| addrope-fixed-nope | 3.162651 | 214659 | 0 | -- |
| addrope-fixed-postrope | 3.151333 | 210287 | 0 | -- |

## Paired contrasts

Negative deltas favor the first named arm. The factorial interaction is `(C+R - R) - (C - N)`.

| Contrast | Delta | Paired 95% interval |
|---|---:|---:|
| qkpre-rope_minus_rope | -0.037011 | [-0.038598, -0.035435] |
| qkpre-nope_minus_nope | -0.093202 | [-0.095006, -0.091425] |
| qkpre-rope_minus_qkpre-nope | -0.023036 | [-0.024669, -0.021376] |
| rope_minus_nope | -0.079228 | [-0.081078, -0.077354] |
| qkpre-fixed-rope_minus_qkpre-rope | +0.006464 | [+0.005537, +0.007401] |
| qkpre-global-rope_minus_qkpre-rope | +0.000421 | [-0.000478, +0.001308] |
| qkpre-first-rope_minus_qkpre-rope | +0.002696 | [+0.001176, +0.004229] |
| input-rope_minus_qkpre-rope | +0.006635 | [+0.005154, +0.008125] |
| addrope-fixed-nope_minus_rope | -0.021116 | [-0.022640, -0.019594] |
| addrope-fixed-nope_minus_qkpre-rope | +0.015895 | [+0.014293, +0.017506] |
| addrope-fixed-postrope_minus_rope | -0.032433 | [-0.033966, -0.030892] |
| addrope-fixed-postrope_minus_qkpre-rope | +0.004578 | [+0.003156, +0.006047] |
| carrier_by_rope_interaction | +0.056192 | [+0.054009, +0.058424] |

Paired-example intervals quantify holdout precision within one training seed; they do not estimate training-seed uncertainty.
