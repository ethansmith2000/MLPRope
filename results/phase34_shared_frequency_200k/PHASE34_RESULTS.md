# Phase 34: globally shared frequency at 200k

All arms use seed 123 and a common 200k learning-rate horizon. Negative
deltas favor the candidate.

| Arm | Final loss | Target tok/s | Peak MiB |
| --- | ---: | ---: | ---: |
| rope-fixed | 3.385691 | 183,972 | 5,076 |
| rope-global-log | 3.384260 | 176,008 | 5,096 |
| qkpre-fixed | 3.324979 | 181,769 | 5,220 |
| qkpre-frequency-log | 3.325839 | 183,002 | 5,220 |
| qkpre-frequency-horizon | 3.326320 | 183,638 | 5,220 |

| Contrast | Final delta | Paired-example 95% CI | Late delta/10k |
| --- | ---: | ---: | ---: |
| rope-global-log_vs_rope-fixed | -0.001431 | [-0.002680, -0.000183] | +0.000295 |
| qkpre-frequency-log_vs_qkpre-fixed | +0.000861 | [-0.000339, +0.002060] | -0.000037 |
| qkpre-frequency-horizon_vs_qkpre-fixed | +0.001341 | [+0.000300, +0.002382] | -0.000007 |
| qkpre-frequency-horizon_vs_qkpre-frequency-log | +0.000481 | [-0.000745, +0.001706] | +0.000029 |

The JSON companion contains every development-curve point, the complete
learned spectra, spectral-health diagnostics, parameter counts, and the
mechanical endpoint/interval gate. Late-curve promotion remains an explicit
review rather than an arbitrary hidden threshold.
