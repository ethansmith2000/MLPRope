# Phase 37: direct-amplitude confirmation at 200k

All arms use seed 123, fixed RoPE, the tied pre-Q/K carrier, and one
common 200k schedule. Negative deltas favor the candidate.

| Arm | Final loss | Target tok/s | Peak MiB | Gate range | Amplitude-factor range | Position-energy mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| scalar | 3.325647 | 179,484 | 5,196 | 0.076--0.257 | 1.000--1.000 | 0.044 |
| exponential-amplitude | 3.325284 | 180,266 | 5,196 | 0.134--0.248 | 0.290--4.368 | 0.095 |
| direct-amplitude | 3.325758 | 178,367 | 5,196 | 0.186--0.408 | 0.004--2.664 | 0.113 |

| Contrast | Final delta | Paired-example 95% CI | Dev late mean | Dev late delta/1k | Gate |
| --- | ---: | ---: | ---: | ---: | --- |
| direct-amplitude_vs_scalar | +0.000111 | [-0.001041, +0.001263] | -0.002427 | -0.000001 | no |
| exponential-amplitude_vs_scalar | -0.000363 | [-0.001559, +0.000833] | +0.001190 | -0.000015 | no |
| direct-amplitude_vs_exponential-amplitude | +0.000474 | [-0.000589, +0.001537] | -0.003617 | +0.000014 | no |

## Decision

The Phase-36 short-horizon direct-amplitude win does not survive the
predeclared 200k primary endpoint. Direct amplitude is effectively tied
with scalar (+0.000111,
CI [-0.001041,
+0.001263]). The interval excludes
the required 0.002-nat improvement. Exponential amplitude is also null
(-0.000363), and
direct does not beat exponential on the holdout
(+0.000474).
No amplitude-shape arm earns seed replication.

The repeatedly measured 128-example development slice favored direct
amplitude by -0.002427 on
average from 150k--200k, while the disjoint 1,024-example primary holdout
did not. Its per-step paired uncertainty is large enough to explain this
difference. The larger frozen holdout governs the decision; the mismatch
is evidence that millinat-scale rankings are slice-sensitive.

## Mechanism and optimization health

All optimization traces finite: True. Direct factors stayed
nonnegative but reached 0.004--2.664;
one band was almost completely suppressed. Exponential factors reached
0.290--4.368. Both models
learned substantial, distinct spectra and reduced their scalar gates,
yet neither improved held-out loss materially. This is evidence against
the amplitude-shape hypothesis at this scale/horizon, not an inactive-path
or saturation diagnosis.

Training resumed once from complete step-70k checkpoints after the
interactive launcher exited. Model, optimizer, scheduler, sampler, and RNG
states were restored. Both launches for every arm recorded the identical
clean source commit, and all arms then completed normally.

## Disposition

Keep the scalar pre-Q/K carrier with fixed RoPE as the active default.
Treat both smooth-amplitude maps and all Phase-36 frequency maps as
completed ablations. Additional seeds, width transfer, and longer-context
tests are not warranted for these refinements without a new hypothesis.
The broader pre-Q/K and AddRoPE mechanisms remain supported by their
separate evidence; this result only closes the smooth carrier-shape branch.

