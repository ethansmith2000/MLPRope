# Phase 61: modern larger-scale transfer

| Arm | Final NLL | Position params | Total params | ktok/s | Peak alloc GiB | Final gate mean |
|---|---:|---:|---:|---:|---:|---:|
| modern L + standard RoPE | 3.018072 | 0 | 203,321,344 | 119.7 | 21.78 | -- |
| modern L + scalar pre-Q/K + RoPE | 3.014530 | 12 | 203,321,356 | 120.5 | 22.58 | 0.046497 |

## Registered decisions

- Scalar minus RoPE: `-0.003542` NLL.
- IID 95% interval: `[-0.005051, -0.001995]`.
- Block-32 95% interval: `[-0.005382, -0.001669]`.
- Positive scale transfer: **True**.
- Material (`<= -0.010`) scale transfer: **False**.

## Development curve

| Step | Scalar minus RoPE NLL | Scalar gate mean |
|---:|---:|---:|
| 5,000 | -0.010359 | 0.304885 |
| 10,000 | -0.003049 | 0.222445 |
| 15,000 | +0.000894 | 0.183011 |
| 20,000 | -0.001535 | 0.158961 |
| 25,000 | +0.000009 | 0.140970 |
| 30,000 | -0.001003 | 0.124903 |
| 35,000 | -0.004435 | 0.114676 |
| 40,000 | +0.002772 | 0.105849 |
| 45,000 | -0.003216 | 0.097926 |
| 50,000 | -0.003290 | 0.090330 |
| 55,000 | -0.004939 | 0.084010 |
| 60,000 | -0.004420 | 0.077966 |
| 65,000 | -0.003639 | 0.072591 |
| 70,000 | -0.002502 | 0.067602 |
| 75,000 | -0.001868 | 0.062444 |
| 80,000 | -0.003840 | 0.058787 |
| 85,000 | -0.003681 | 0.054878 |
| 90,000 | -0.003862 | 0.050908 |
| 95,000 | -0.003775 | 0.048096 |
| 100,000 | -0.003131 | 0.046497 |

## Artifact closeout

Recovery cleanup reclaimed `4.54` GiB across `2` completed runs. No final weights were saved.

One training seed; paired block intervals measure final-stream precision, not seed variability or attribution within the modern bundle.
