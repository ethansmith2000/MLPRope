# Phase 58: scalar-carrier initialization scout

| Arm | Final NLL | Delta vs alpha1 direct (block-32 95% CI) | Final effective gate range | Function-step / parameter-step | Promote? |
|---|---:|---:|---:|---:|:---:|
| learned alpha, init 1.0 | 3.371182 | `+0.000000 [+0.000000, +0.000000]` | `[0.1066, 0.2776]` | 0.7071 | no |
| learned alpha, init 0.1 | 3.422976 | `+0.051794 [+0.049574, +0.054039]` | `[0.0151, 0.2259]` | 0.7071 | no |
| alpha=0.1g, g init 1.0 | 3.431190 | `+0.060008 [+0.057877, +0.062629]` | `[0.0309, 0.1740]` | 0.0707 | no |
| fixed alpha=0.1 | 3.431287 | `+0.060105 [+0.057927, +0.062961]` | `[0.1000, 0.1000]` | n/a | no |

## Registered readout

- Direct alpha=0.1 minus direct alpha=1.0: `+0.051794` NLL.
- IID 95% interval: `[+0.050108, +0.053492]`.
- Contiguous-block-32 95% interval: `[+0.049574, +0.054039]`.
- Direct alpha=0.1 minus scaled alpha=0.1g: `-0.008214 [-0.009897, -0.006586]`.
- Scaled alpha=0.1g minus fixed alpha=0.1: `-0.000098 [-0.001435, +0.001164]`.
- No arm passed the frozen promotion rule.
- All training and optimizer metrics finite: **True**.

No checkpoints or final weights were requested for this short scout.

This is a one-seed, 20k design scout. Paired block intervals quantify endpoint-stream precision, not training-seed variability or mature-horizon performance. At most one passing alternative may advance to 100k.
