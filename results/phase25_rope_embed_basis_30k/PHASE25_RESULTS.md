# Phase-25 additive-carrier 30k promotion

Losses use the disjoint 1,024-example holdout beginning at validation
batch 2,048. Deltas are candidate minus reference; negative is better.

| Arm | Seed 123 | Seed 456 | Seed 789 | Mean | Median target tok/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| rope-fixed | 3.794082 | 3.760904 | 3.795458 | 3.783481 | 193,295 |
| basis16-a03 | 3.731482 | 3.717609 | 3.715437 | 3.721510 | 179,434 |
| basis16-a10 | 3.708271 | 3.704839 | 3.706734 | 3.706615 | 178,479 |

| Contrast | Seed 123 | Seed 456 | Seed 789 | Mean | Wins all? | Clears 0.01? |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| basis16-a03_vs_rope-fixed | -0.062600 | -0.043295 | -0.080021 | -0.061972 | True | True |
| basis16-a10_vs_rope-fixed | -0.085811 | -0.056065 | -0.088725 | -0.076867 | True | True |
| basis16-a10_vs_basis16-a03 | -0.023211 | -0.012770 | -0.008704 | -0.014895 | True | True |
