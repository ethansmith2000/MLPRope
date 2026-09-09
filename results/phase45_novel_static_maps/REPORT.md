# Phase 45: novel static sinusoid maps

| Arm | Final NLL | Parent delta | Paired 95% interval | Position params | tokens/s | Pass |
|---|---:|---:|---:|---:|---:|:---:|
| input-dense-linear | 3.473583 | +0.001721 | [+0.000610, +0.002826] | 589825 | 216212 | no |
| input-mlp-h768 | 3.470171 | -0.001691 | [-0.002543, -0.000843] | 1179649 | 216807 | no |
| qk-dense-premap | 3.449165 | -0.005913 | [-0.007137, -0.004689] | 4718600 | 215383 | yes |
| qk-dense-shared | 3.440949 | -0.014128 | [-0.015390, -0.012857] | 4718600 | 208453 | yes |
| qk-dense-separate | 3.441223 | -0.013855 | [-0.015094, -0.012615] | 9437192 | 209973 | yes |
| qk-mlp-r128 | 3.441467 | -0.013611 | [-0.014836, -0.012372] | 2359304 | 207923 | yes |

Input scalar parent: 3.471862.
Pre-Q/K scalar parent: 3.455078.

## Secondary paired contrasts

Negative deltas favor the first named arm.

| Contrast | Delta | Paired 95% interval |
|---|---:|---:|
| qk-dense-shared_minus_qk-dense-separate | -0.000273 | [-0.001290, +0.000739] |
| qk-dense-shared_minus_qk-mlp-r128 | -0.000517 | [-0.001604, +0.000573] |
| qk-dense-shared_minus_qk-r32-residual | -0.004727 | [-0.005909, -0.003551] |
| qk-dense-separate_minus_qk-r32-residual | -0.004454 | [-0.005615, -0.003275] |
| qk-mlp-r128_minus_qk-r32-residual | -0.004210 | [-0.005254, -0.003152] |
| qk-dense-premap_minus_qk-r32-premap | -0.000785 | [-0.001822, +0.000232] |
| input-mlp-h768_minus_input-pair-amplitude | +0.000940 | [+0.000039, +0.001837] |

Existing Phase-43/44 controls use the identical 20k paper-base schedule, paired initialization seed, data order, and final holdout. Paired-example intervals do not estimate training-seed uncertainty.
