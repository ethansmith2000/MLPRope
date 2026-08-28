# Phase-28 qk-preprojection 30k promotion

Losses use the disjoint 1,024-example holdout beginning at validation
batch 2,048. Deltas are qkpre-rope minus fixed RoPE; negative is better.

| Arm | Seed 123 | Seed 456 | Seed 789 | Mean | Median target tok/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| rope-fixed | 3.794252 | 3.760048 | 3.798027 | 3.784109 | 192,714 |
| qkpre-rope | 3.720763 | 3.716620 | 3.719239 | 3.718874 | 192,507 |

| Contrast | Seed 123 | Seed 456 | Seed 789 | Mean | Wins all? | Clears 0.01? |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| qkpre-rope vs rope-fixed | -0.073489 | -0.043428 | -0.078788 | -0.065235 | True | True |

| Step | Mean development-loss delta | Wins all seeds? |
| ---: | ---: | --- |
| 5,000 | -0.069417 | True |
| 10,000 | -0.066357 | True |
| 15,000 | -0.063933 | True |
| 20,000 | -0.064407 | True |
| 25,000 | -0.059546 | True |
| 30,000 | -0.060839 | True |

The JSON companion contains per-seed paired-example intervals, seed
dispersion, protocol fingerprints, throughput, complete development
curves, and qk-preprojection mechanism-health diagnostics.
