# Phase 44: input-sinusoid adapters

| Arm | Final holdout NLL |
|---|---:|
| lowrank-linear-r32 | 3.470090 |
| per-pair-amplitude | 3.469231 |
| input-scalar-control | 3.471862 |

| Contrast | Delta | Paired 95% interval |
|---|---:|---:|
| lowrank-linear-r32_minus_input-scalar-control | -0.001772 | [-0.002243, -0.001291] |
| per-pair-amplitude_minus_input-scalar-control | -0.002631 | [-0.003148, -0.002104] |

Paired-example intervals measure holdout precision within one seed, not training-seed uncertainty.
