# Phase 33: static pre-Q/K adapter at 200k

All arms use seed 123 and a common 200k learning-rate horizon. Final loss
uses the disjoint 1,024-example holdout beginning at validation batch 2,048.
Negative deltas favor the candidate.

| Arm | Final loss | Target tok/s |
| --- | ---: | ---: |
| rope-fixed | 3.387442 | 184,385 |
| qkpre-tied-nope | 3.355384 | 185,509 |
| qkpre-tied-rope | 3.324611 | 183,540 |
| qkpre-split-scalar-rope | 3.325601 | 183,307 |
| qkpre-pair-amplitude-rope | 3.325484 | 179,847 |
| qkpre-pair-polar-rope | 3.325650 | 180,617 |

| Contrast | Delta | Paired-example 95% CI |
| --- | ---: | ---: |
| qkpre-tied-nope_vs_rope-fixed | -0.032058 | [-0.033636, -0.030480] |
| qkpre-tied-rope_vs_rope-fixed | -0.062831 | [-0.064426, -0.061237] |
| rope_contribution | -0.030773 | [-0.032424, -0.029123] |
| split-scalar_vs_tied | +0.000990 | [-0.000096, +0.002076] |
| pair-amplitude_vs_split-scalar | -0.000116 | [-0.001212, +0.000979] |
| pair-polar_vs_pair-amplitude | +0.000165 | [-0.000847, +0.001178] |

The tied carrier remains a large improvement and fixed RoPE remains
material on top of it. Separate Q/K gains, pairwise amplitudes, and
pairwise phases are all null at this horizon, so Phase 34 retains only
the tied carrier and tests a globally shared frequency bank.
