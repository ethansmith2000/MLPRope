# Phase 59: modern-backbone transfer

| Arm | Final NLL | Position params | Total params | Non-embedding/head params | ktok/s | Peak alloc GiB |
|---|---:|---:|---:|---:|---:|---:|
| modern + standard RoPE | 3.153352 | 0 | 95,271,168 | 56,636,928 | 249.2 | 12.39 |
| modern + scalar pre-Q/K + RoPE | 3.141150 | 8 | 95,271,176 | 56,636,936 | 248.0 | 12.79 |

## Registered transfer gate

- Scalar minus RoPE: `-0.012201` NLL.
- IID 95% interval: `[-0.013708, -0.010669]`.
- Contiguous-block-32 95% interval: `[-0.013890, -0.010483]`.
- Transfer gate passed: **True**.
- Continuation: eligible for a separately frozen modern rank-32 arm.

## Artifact closeout

Recovery cleanup reclaimed `2.13` GiB across `2` completed runs. No final weights were saved.

This is a one-training-seed bundled architecture transfer. Paired block intervals quantify final-stream precision, not seed variability, and do not attribute the result to an individual backbone component.
