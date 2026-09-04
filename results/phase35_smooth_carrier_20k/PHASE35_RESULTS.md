# Phase 35: smooth sinusoidal carrier at 20k

All arms use seed 123 and one common 20k schedule. Negative deltas favor
the candidate.

| Arm | Final loss | Target tok/s | Peak MiB | Scalar-gate range | Spectral amp range | Phase max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| rope-fixed | 3.843271 | 182,417 | 5,222 | 0.369--0.930 | 1.000--1.000 | 0.0000 |
| rope-amplitude | 3.840667 | 181,084 | 5,242 | 0.449--0.932 | 0.451--1.906 | 0.0000 |
| rope-polar | 3.840808 | 180,913 | 5,242 | 0.450--0.931 | 0.452--1.912 | 0.1452 |
| rope-split-polar | 3.840545 | 178,936 | 5,362 | 0.471--1.058 | 0.402--1.866 | 0.1817 |
| nope-fixed | 3.888480 | 184,470 | 5,222 | 0.615--2.330 | 1.000--1.000 | 0.0000 |
| nope-amplitude | 3.877836 | 183,821 | 5,242 | 0.556--1.803 | 0.232--3.306 | 0.0000 |
| nope-polar | 3.877977 | 183,999 | 5,242 | 0.556--1.777 | 0.232--3.255 | 0.1960 |
| nope-split-polar | 3.876471 | 180,791 | 5,362 | 0.536--1.984 | 0.178--4.366 | 0.5780 |

| Contrast | Final delta | Paired-example 95% CI | Late delta/1k | Gate |
| --- | ---: | ---: | ---: | --- |
| rope-amplitude_vs_rope-fixed | -0.002604 | [-0.003194, -0.002013] | -0.000183 | no |
| rope-polar_vs_rope-amplitude | +0.000141 | [-0.000121, +0.000403] | +0.000036 | no |
| rope-split-polar_vs_rope-polar | -0.000264 | [-0.000623, +0.000096] | +0.000045 | no |
| nope-amplitude_vs_nope-fixed | -0.010644 | [-0.011700, -0.009589] | -0.000072 | pass |
| nope-polar_vs_nope-amplitude | +0.000142 | [-0.000341, +0.000624] | +0.000000 | no |
| nope-split-polar_vs_nope-polar | -0.001506 | [-0.002149, -0.000863] | -0.000150 | no |
| rope-fixed_vs_nope-fixed | -0.045209 | [-0.046874, -0.043544] | +0.000286 | backbone |
| rope-amplitude_vs_nope-amplitude | -0.037168 | [-0.038797, -0.035539] | +0.000175 | backbone |
| rope-polar_vs_nope-polar | -0.037169 | [-0.038809, -0.035529] | +0.000211 | backbone |
| rope-split-polar_vs_nope-split-polar | -0.035927 | [-0.037545, -0.034308] | +0.000406 | backbone |

## Decision

Only NoPE smooth amplitude clears the direct-parent gate (-0.010644).
Smooth amplitude under RoPE is a small, precise favorable signal (-0.002604),
but it misses the predeclared 0.003-nat practical threshold. Phase is null
under both backbones, and Q/K splitting is below threshold.

Smooth amplitude recovers 17.8% of the fixed-shape RoPE-versus-NoPE gap,
but does not replace RoPE: amplitude+RoPE remains better than
amplitude+NoPE by 0.037168 nats. No automatic seed
or 200k expansion is warranted unless a specifically
RoPE-free model is the research target.

The learned spectra are not disguised scalar-gate changes. With RoPE,
every layer favors the lowest-frequency quartile by 1.54x--3.48x
relative to the highest-frequency quartile. NoPE learns heterogeneous
layer roles spanning 0.26x--3.19x. This is descriptive one-seed
evidence, not yet a general spectral law.

## Optimization audit

Every arm has finite, nonzero functional movement and a gradient-clip
ratio of 1.0 at the last active sample. Across arms, the median
descent-update/gradient cosine is 0.317--0.445, and
78.6%--96.4% of nonzero-update samples have positive alignment.
Some 19k samples turn negative as the linear schedule
approaches zero, including the scalar controls; their function-space
steps are tiny. There is no intervention-specific explosion, clipping,
or Adam suppression that explains the null phase results.

The JSON companion contains every development point, losslessly compact
rank-4 DCT profile coordinates, parameter counts, and optimization-health
summaries. Here `fixed` means a fixed spectral shape with the existing
learned per-layer tied scalar gate; the smooth modes add zero-mean
spectral deformation. Inspect late curves before promoting any endpoint
pass.

