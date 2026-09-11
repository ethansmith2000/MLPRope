# Phase 51 carrier counterfactuals — seed 789

All interventions use the same trained checkpoint and holdout blocks.
QK normalization and every downstream hidden state are recomputed.
Negative deltas relative to `full` are better.

| Intervention | NLL | Delta vs full | IID 95% interval | Block-32 95% interval |
|---|---:|---:|---:|---:|
| full | 3.162242 | +0.000000 | [+0.000000, +0.000000] | [+0.000000, +0.000000] |
| direct_mean_only | 3.171748 | +0.009506 | [+0.009101, +0.009919] | [+0.009059, +0.009949] |
| direct_mean_removed | 6.680508 | +3.518266 | [+3.495919, +3.540586] | [+3.480795, +3.552770] |
| direct_zero | 6.778672 | +3.616430 | [+3.593671, +3.639314] | [+3.579847, +3.650316] |
| scalar_zero | 3.162360 | +0.000118 | [+0.000052, +0.000184] | [+0.000048, +0.000188] |
| all_zero | 6.784215 | +3.621973 | [+3.599343, +3.644155] | [+3.585537, +3.656602] |

## Full-path reproduction check

Original saved mean NLL: `3.162180`; fresh full-path mean: `3.162242`; difference: `+0.00006237`.

## Interpretation limits

These are checkpoint interventions, not independently trained models.
A neutral removal can identify what the trained network currently needs,
but it cannot show whether a branch was useful as optimization scaffolding.
The block-32 interval is a sensitivity analysis for adjacent validation
blocks that may share source-document context; it is not a second corpus
or training-seed replicate.
