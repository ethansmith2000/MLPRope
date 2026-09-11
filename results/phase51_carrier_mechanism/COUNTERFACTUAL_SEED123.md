# Phase 51 carrier counterfactuals — seed 123

All interventions use the same trained checkpoint and holdout blocks.
QK normalization and every downstream hidden state are recomputed.
Negative deltas relative to `full` are better.

| Intervention | NLL | Delta vs full | IID 95% interval | Block-32 95% interval |
|---|---:|---:|---:|---:|
| full | 3.159455 | +0.000000 | [+0.000000, +0.000000] | [+0.000000, +0.000000] |
| direct_mean_only | 3.172262 | +0.012807 | [+0.012331, +0.013296] | [+0.012378, +0.013268] |
| direct_mean_removed | 6.562661 | +3.403207 | [+3.379025, +3.427071] | [+3.361210, +3.444140] |
| direct_zero | 6.590876 | +3.431421 | [+3.407786, +3.454842] | [+3.390085, +3.472169] |
| scalar_zero | 3.159501 | +0.000046 | [-0.000015, +0.000107] | [-0.000036, +0.000130] |
| all_zero | 6.597047 | +3.437592 | [+3.413571, +3.461599] | [+3.395680, +3.477715] |

## Full-path reproduction check

Original saved mean NLL: `3.159413`; fresh full-path mean: `3.159455`; difference: `+0.00004191`.

## Interpretation limits

These are checkpoint interventions, not independently trained models.
A neutral removal can identify what the trained network currently needs,
but it cannot show whether a branch was useful as optimization scaffolding.
The block-32 interval is a sensitivity analysis for adjacent validation
blocks that may share source-document context; it is not a second corpus
or training-seed replicate.
