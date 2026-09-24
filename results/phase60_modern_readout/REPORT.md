# Phase 60: modern dedicated carrier readouts

| Arm | Final NLL | Position params | Total params | ktok/s | Peak alloc GiB | Final gate mean |
|---|---:|---:|---:|---:|---:|---:|
| modern scalar pre-Q/K + RoPE | 3.131003 | 8 | 95,271,176 | 247.8 | 12.79 | 0.040714 |
| modern scalar + rank-32 Q/K readout + RoPE | 3.128782 | 589,832 | 95,861,000 | 239.4 | 12.83 | 0.004235 |
| modern scalar + rank-128 Q/K readout + RoPE | 3.126594 | 2,359,304 | 97,630,472 | 238.8 | 12.85 | 0.003244 |

## Paired final-window contrasts

Negative deltas favor the first named arm.

| Contrast | Mean delta | IID 95% interval | Block-32 95% interval |
|---|---:|---:|---:|
| rank32_minus_scalar | -0.002222 | [-0.003794, -0.000678] | [-0.003978, -0.000465] |
| rank128_minus_scalar | -0.004410 | [-0.005990, -0.002853] | [-0.006475, -0.002440] |
| rank128_minus_rank32 | -0.002188 | [-0.003667, -0.000710] | [-0.003882, -0.000520] |

## Registered decisions

- Median rank-128/rank-32 carrier-function-step ratio: `1.048`; registered match: **True**.
- Rank 32 transfers beyond scalar: **False**.
- Rank 128 is necessary beyond rank 32: **False**.
- Decision: retain the scalar primary method; do not tune the readout on this window.

## Artifact closeout

Recovery cleanup reclaimed `3.23` GiB across `3` completed runs. No final weights were saved.

This conditional modern-backbone extension cohort uses one training seed. Paired block intervals measure endpoint-stream precision, not training-seed uncertainty. The inherited controlled-backbone capacity control is not a new modern-backbone capacity control.
