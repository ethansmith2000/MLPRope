# Phase 36 direct-carrier calibration

Eight h768/d8/context-1024 runs used seed 123, 512 optimizer steps, and a
common 200-step warmup. These runs select numerically healthy optimizer speeds;
their four-batch final losses are not promotion evidence. Every run used fixed
RoPE, the live pre-Q/K carrier at gate 1, method-aware Q/K RMSNorm, fp32 carrier
coordinates, and no positional weight decay.

The two new frequency maps are direct and nonsaturating:

```text
global: omega_i = omega_i^0 + omega_i^0 * tau/(L*omega_max) * u
hybrid: omega_i = omega_i^0 + min(omega_i^0, tau/L) * (B c)_i
```

Here `tau=1`, `L=1024`, and the hybrid bank uses four unit-RMS DCT modes. The
new direct amplitude is `a_i = g * (1 + (B c)_i)` with four zero-mean unit-RMS
DCT modes. No sigmoid, softplus, tanh, or exponential appears in a new mode.

## Frequency health

The table reports the largest observed single-update endpoint phase step over
the sparse steps 1,2,4,...,512, followed by the final endpoint displacement.

| Arm | LR multiplier | Max step RMS | Max step abs | Final delta RMS | Final delta abs | Final multiplier range |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| global | 0.25 | 0.000012 | 0.000061 | 0.000157 | 0.000671 | 0.999999--0.999999 |
| global | 1 | 0.000029 | 0.000122 | 0.000620 | 0.002625 | 0.999997--0.999997 |
| global | 4 | 0.000114 | 0.000488 | 0.002538 | 0.010742 | 0.999989--0.999990 |
| hybrid rank 4 | 0.25 | 0.000091 | 0.000182 | 0.025163 | 0.050335 | 0.9643--1.0104 |
| hybrid rank 4 | 1 | 0.000363 | 0.000727 | 0.093839 | 0.187642 | 0.8689--1.0429 |
| hybrid rank 4 | 4 | 0.001424 | 0.002852 | 0.284604 | 0.565844 | 0.6342--1.1793 |

All frequency gradient-clip ratios were exactly 1.0, every frequency remained
positive, and every spectrum preserved its original ordering. The global
coordinate receives a much smaller aggregate learning signal than the
four-coordinate hybrid warp. Phase 36 therefore uses global LR `4x`, hybrid LR
`1x`, and retains a separate hybrid `4x` arm as an explicit optimization-speed
sensitivity test. Loss did not determine this selection.

## Direct amplitude and Q/K normalization health

At 512 steps, direct amplitude factors in layer 0 were `0.966--1.044` for the
amplitude-only arm and `0.965--1.045` when combined with hybrid frequency. They
were finite and positive, although the parameterization intentionally permits
signed values.

Across all eight runs and layers, the measured pre-projection positional-energy
fraction was `0.323--0.356`, close to the `1/3` prediction for gate 1 from
`RMS(x) ~= 1` and `RMS(z) = 1/sqrt(2)`. After Q/K projection and method-aware
RMSNorm, the cosine between carrier-augmented and content-only queries ranged
from about `0.68` to `0.86` in the direct-amplitude arms. The carrier is
therefore live and substantial, while Q/K RMSNorm controls its total scale.

The ordinary model-gradient group clipped during the earliest warmup samples,
which also scales the per-layer carrier-adapter gradients. This is common to
the paired models. Frequency coordinates are excluded from that group and did
not invoke their independent clip.

## Gate

Proceed to the one-seed 20k screen with:

- fixed RoPE;
- scalar pre-Q/K + RoPE;
- direct smooth amplitude + RoPE;
- global direct frequency at `4x` + RoPE;
- hybrid direct frequency at `1x` and `4x` + RoPE;
- direct amplitude plus hybrid frequency at `1x` + RoPE.

The 20k comparison, not this calibration, decides whether any new coordinate
earns longer evaluation.
