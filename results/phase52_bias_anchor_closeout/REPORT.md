# Phase 52: bias and scalar-anchor close-out

## Endpoint NLL on untouched blocks 5120–6143

| Arm | NLL |
|---|---:|
| scalar-qk-bias | 3.110102 |
| qk-readout-r32-no-anchor | 3.107455 |
| rope | 3.145657 |
| scalar-qkpre | 3.110001 |
| qk-readout-r32 | 3.099160 |

## Paired contrasts

Negative deltas favor the candidate.

| Contrast | Delta | IID 95% interval | Block-32 95% interval |
|---|---:|---:|---:|
| scalar_qk_bias_minus_rope | -0.035555 | [-0.037119, -0.034002] | [-0.037656, -0.033426] |
| scalar_qk_bias_minus_scalar | +0.000101 | [-0.000819, +0.001009] | [-0.000664, +0.000885] |
| scalar_qk_bias_minus_rank32 | +0.010942 | [+0.009515, +0.012343] | [+0.009669, +0.012069] |
| rank32_no_anchor_minus_rope | -0.038202 | [-0.039821, -0.036596] | [-0.040731, -0.035710] |
| rank32_no_anchor_minus_scalar | -0.002546 | [-0.003987, -0.001128] | [-0.004212, -0.000910] |
| rank32_no_anchor_minus_rank32 | +0.008295 | [+0.006805, +0.009833] | [+0.006302, +0.010184] |
| rank32_no_anchor_minus_scalar_qk_bias | -0.002647 | [-0.004146, -0.001193] | [-0.004289, -0.001086] |

## Frozen interpretation rules

- Bias simplification within +0.003 NLL of rank-32: **False**.
- No-anchor rank-32 within +0.003 NLL of rank-32: **False**.
- Q/K biases moved from zero: **True**.

## Inference limit

All arms use the same 1,024 untouched validation blocks, so paired example and contiguous-block intervals quantify holdout-sample precision. This is a seed-123 design close-out, not a new training-seed replication; Phase 50 supplies the three-seed evidence for the parent rank-32 method. The 0.003-NLL margin is a descriptive engineering criterion, not a formal equivalence test.

Neither new run saved periodic checkpoints or final model weights.
