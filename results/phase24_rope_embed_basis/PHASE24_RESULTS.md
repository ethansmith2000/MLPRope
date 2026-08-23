# Phase-24 RoPE-embedding basis screen

All values use the disjoint 256-example holdout beginning at validation
batch 2048. The large run directories were intentionally removed after
completion; durable logs retain aggregate losses but not per-example
arrays, so paired-example confidence intervals cannot be reconstructed.

## Primary context 1024

| Arm | Seed 123 | Seed 456 | Seed 789 | Mean | Position params |
| --- | ---: | ---: | ---: | ---: | ---: |
| rope-fixed | 4.598300 | 4.550100 | 4.590300 | 4.579567 | 0 |
| basis16-a03 | 4.470900 | 4.467000 | 4.478800 | 4.472233 | 1,308,672 |
| basis16-a10 | 4.431000 | 4.422500 | 4.430500 | 4.428000 | 1,308,672 |
| ropeembed-a10 | 4.444700 | 4.432800 | 4.448600 | 4.442033 | 1,191,936 |

Deltas are candidate minus reference; negative favors the candidate.

| Contrast | Seed 123 | Seed 456 | Seed 789 | Mean | Wins all? | Clears 0.01? |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| basis16-a03_vs_rope-fixed | -0.127400 | -0.083100 | -0.111500 | -0.107333 | True | True |
| basis16-a10_vs_basis16-a03 | -0.039900 | -0.044500 | -0.048300 | -0.044233 | True | True |
| ropeembed-a10_vs_basis16-a10 | +0.013700 | +0.010300 | +0.018100 | +0.014033 | False | False |
| ropeembed-a10_vs_rope-fixed | -0.153600 | -0.117300 | -0.141700 | -0.137533 | True | True |

## Exploratory longer contexts

These contexts exceed the training length and are not primary endpoints.

| Arm | Mean at 2048 | Mean at 4096 |
| --- | ---: | ---: |
| rope-fixed | 4.620367 | 4.763900 |
| basis16-a03 | 4.542300 | 4.745067 |
| basis16-a10 | 4.515700 | 4.753433 |
| ropeembed-a10 | 4.637067 | 5.119733 |

## Decision

Raising the amplitude anchor from 0.3 to 1.0 is the decisive change in this screen. Replacing the compact basis-plus-scalars with the full native RoPE embedding loses about 0.014 mean loss and loses in all three seeds. The native basis remains much better than fixed RoPE, so this is evidence for attention-local additive injection, not evidence that the full RoPE basis is harmful in absolute terms.

At the exploratory 4096 context, however, the native-basis arm is 0.356 worse than fixed RoPE while the compact arms remain near it. The 1024-context selection must therefore not be advertised as a length-extrapolation result.

As a 5k screen, this can select controls and parameterizations but is not a durable positive claim without a longer confirmation.

