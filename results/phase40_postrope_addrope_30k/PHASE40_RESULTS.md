# Phase 40: post-RoPE AddRoPE ordering

Negative deltas favor the first named arm.

| Run | Final loss | tok/s | position params |
| --- | ---: | ---: | ---: |
| rope-fixed | 3.794820 | 181924 | 0 |
| qkpre-rope | 3.721015 | 181521 | 8 |
| addrope-direct-nope | 3.741599 | 181466 | 12288 |
| addrope-fixed-prerope | 3.775458 | 182366 | 0 |
| addrope-direct-prerope | 3.767408 | 180104 | 12288 |
| addrope-fixed-postrope | 3.739603 | 183485 | 0 |
| addrope-direct-postrope | 3.736785 | 181367 | 12288 |

## Paired ordering contrasts

| Contrast | Delta | 95% paired interval |
| --- | ---: | ---: |
| fixed_postrope_minus_fixed_prerope | -0.035855 | [-0.037309, -0.034362] |
| direct_postrope_minus_direct_prerope | -0.030623 | [-0.032042, -0.029193] |
| direct_postrope_minus_direct_nope | -0.004814 | [-0.006463, -0.003149] |
| direct_postrope_minus_fixed_postrope | -0.002818 | [-0.003516, -0.002118] |
| direct_postrope_minus_rope_fixed | -0.058035 | [-0.059589, -0.056464] |
| direct_postrope_minus_qkpre_rope | +0.015770 | [+0.014208, +0.017381] |

Phase 40 reuses completed Phase-39 controls with identical data, initialization, optimizer, and evaluation protocol. It remains a one-training-seed mechanism screen.
