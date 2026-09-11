# Phase 51: carrier counterfactual summary

Available training seeds: 123, 456, 789.
Positive deltas mean the intervention hurts relative to the trained full path.

| Intervention | Seed deltas | Mean | Seed t interval |
|---|---:|---:|---:|
| direct_mean_only | 123: +0.012807, 456: +0.010967, 789: +0.009506 | +0.011093 | [+0.006984, +0.015203] |
| direct_mean_removed | 123: +3.403207, 456: +3.505109, 789: +3.518266 | +3.475527 | [+3.319086, +3.631968] |
| direct_zero | 123: +3.431421, 456: +3.489651, 789: +3.616430 | +3.512501 | [+3.277509, +3.747493] |
| scalar_zero | 123: +0.000046, 456: +0.000162, 789: +0.000118 | +0.000109 | [-0.000036, +0.000254] |
| all_zero | 123: +3.437592, 456: +3.489160, 789: +3.621973 | +3.516242 | [+3.279933, +3.752550] |

The seed is the replication unit. Per-checkpoint example/block intervals are holdout-sampling sensitivity analyses only.
