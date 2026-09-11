# Phase 51 carrier counterfactuals — seed 456

All interventions use the same trained checkpoint and holdout blocks.
QK normalization and every downstream hidden state are recomputed.
Negative deltas relative to `full` are better.

| Intervention | NLL | Delta vs full | IID 95% interval | Block-32 95% interval |
|---|---:|---:|---:|---:|
| full | 3.159647 | +0.000000 | [+0.000000, +0.000000] | [+0.000000, +0.000000] |
| direct_mean_only | 3.170615 | +0.010967 | [+0.010538, +0.011414] | [+0.010512, +0.011460] |
| direct_mean_removed | 6.664756 | +3.505109 | [+3.479605, +3.529923] | [+3.464302, +3.544472] |
| direct_zero | 6.649299 | +3.489651 | [+3.465725, +3.513905] | [+3.450516, +3.527406] |
| scalar_zero | 3.159809 | +0.000162 | [+0.000104, +0.000221] | [+0.000114, +0.000209] |
| all_zero | 6.648807 | +3.489160 | [+3.464372, +3.513570] | [+3.449748, +3.528503] |

## Full-path reproduction check

Original saved mean NLL: `3.159596`; fresh full-path mean: `3.159647`; difference: `+0.00005096`.

## Interpretation limits

These are checkpoint interventions, not independently trained models.
A neutral removal can identify what the trained network currently needs,
but it cannot show whether a branch was useful as optimization scaffolding.
The block-32 interval is a sensitivity analysis for adjacent validation
blocks that may share source-document context; it is not a second corpus
or training-seed replicate.
