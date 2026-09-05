# Phase 36: direct sinusoid amplitude and frequency at 20k

All arms use seed 123, standard RoPE, and one common 20k schedule unless
the arm is the RoPE-only baseline. Negative deltas favor the candidate.

| Arm | Final loss | Target tok/s | Peak MiB | Shape amp range | Endpoint phase Δ (RMS/max) | Freq-order violations |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| rope-fixed | 3.918938 | 184,460 | 5,076 | n/a | n/a | n/a |
| qkpre-scalar | 3.843334 | 180,413 | 5,218 | 1.000--1.000 | n/a | n/a |
| direct-amplitude | 3.839673 | 181,514 | 5,218 | 0.057--1.698 | n/a | n/a |
| global-frequency-lr4 | 3.843586 | 183,462 | 5,218 | 1.000--1.000 | 0.025/0.105 | 0.00% |
| hybrid-frequency-lr1 | 3.842392 | 180,707 | 5,216 | 1.000--1.000 | 0.384/0.757 | 0.00% |
| hybrid-frequency-lr4 | 3.841076 | 179,512 | 5,216 | 1.000--1.000 | 0.703/1.335 | 6.27% |
| direct-amplitude+hybrid-frequency | 3.839216 | 180,824 | 5,218 | 0.055--1.783 | 0.362/0.706 | 0.00% |

| Contrast | Kind | Final delta | Paired-example 95% CI | Late delta/1k | Gate |
| --- | --- | ---: | ---: | ---: | --- |
| qkpre-scalar_vs_rope-fixed | backbone | -0.075604 | [-0.077218, -0.073990] | -0.000333 | backbone |
| direct-amplitude_vs_qkpre-scalar | primary | -0.003661 | [-0.004247, -0.003074] | -0.000165 | pass |
| global-frequency-lr4_vs_qkpre-scalar | primary | +0.000253 | [-0.000012, +0.000518] | -0.000093 | no |
| hybrid-frequency-lr1_vs_qkpre-scalar | primary | -0.000941 | [-0.001337, -0.000546] | +0.000037 | no |
| hybrid-frequency-lr4_vs_qkpre-scalar | sensitivity | -0.002258 | [-0.002804, -0.001712] | +0.000053 | no |
| direct-amplitude+hybrid-frequency_vs_qkpre-scalar | factorial | -0.004118 | [-0.004723, -0.003512] | -0.000136 | pass |
| direct-amplitude+hybrid-frequency_vs_direct-amplitude | incremental | -0.000457 | [-0.000736, -0.000178] | +0.000029 | no |

## Decision

Direct, signed rank-4 amplitude is the only new component that earns
promotion: -0.003661 nats versus the scalar carrier, with its paired interval wholly beyond
the -0.003 practical threshold.
The amplitude+frequency arm is the lowest-loss arm
(-0.004118 versus scalar),
but almost all of that result is amplitude. Hybrid frequency adds only
-0.000457 nats beyond
direct amplitude, below the practical threshold.

Global direct frequency is null. The conservative rank-4 hybrid is
small and below threshold. The LR4 hybrid moves farther, but still misses
the threshold and reverses 6.27% of
adjacent frequency pairs, so it is not a clean candidate. There is no
frequency variant to promote from this screen.

The scalar pre-Q/K sinusoid remains the dominant architectural effect:
-0.075604 nats versus
RoPE alone at 20k. Direct amplitude is a smaller refinement on top.

## QKNorm and optimization audit

Direct amplitude stayed signed and unconstrained but did not cross zero:
its learned shape factors span 0.057--1.698.
Across layers, the actual pre-projection positional energy fraction spans
0.132--0.418; after
QKNorm, Q's cosine to the content-only direction spans
0.388--0.787.
This confirms that amplitude controls content/position direction rather
than merely acting as attention temperature.
The minimum factor is close enough to zero that a longer run must keep
the signed-factor and per-layer spectrum diagnostics enabled.

All logged intervention values are finite. The frequency groups were
never gradient-clipped (minimum clip ratio 1.000).
Adapter-group clip ratios reach 0.396, reflecting shared
whole-model clipping early in training rather than an intervention-specific
failure. No saturation transform, positivity map, or log-frequency
parameterization was used.

For context, the earlier exponential rank-4 amplitude contrast was
-0.002604; the direct
contrast is better by
-0.001057 nats.
That is an exploratory cross-phase comparison, not a paired claim, but
it supports using the direct coordinate in the confirmation run.

## Evidence limit

The 1,024-example paired intervals establish evaluation precision, not
training-seed robustness. This is one seed and only 20k steps. The clean
next confirmation is direct amplitude (plus its scalar parent) at longer
training and/or additional seeds; frequency should remain shelved unless
a new structural hypothesis is proposed.

The JSON companion records all development contrasts, provenance,
intervention state, QKNorm mixture diagnostics, and optimizer health.

