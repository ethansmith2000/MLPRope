# Phase 35: smooth sinusoidal-carrier screen

_Completed protocol and result, 2026-09-04._

_Historical implementation note: phase and Q/K-split modes were removed from
the active runtime after this screen. The configs and result report remain the
durable record._

## Question

Can a few coherent spectral degrees of freedom improve the tied pre-Q/K
sinusoidal carrier, and does the answer depend on using fixed RoPE versus NoPE?

This is not learned RoPE. RoPE remains the immutable standard rotation when
`use_rope=true`. Every new parameter transforms only the separate sinusoidal
carrier added before `W_q` and `W_k`.

## Geometry

RoPE frequencies are uniformly spaced in log frequency, so low-order DCT-II
modes over pair index provide a smooth orthogonal basis. The columns are scaled
to unit RMS rather than unit L2 norm, giving coordinates a width-independent
functional meaning. For `P` Fourier pairs, let

```text
B_amp[i,r] = sqrt(2) cos(pi (i+1/2) (r+1) / P)
B_phase[i,0] = 1
B_phase[i,r>0] = sqrt(2) cos(pi (i+1/2) r / P).
```

With rank `R=4`, branch `b` uses

```text
delta_b = B_amp u_b
phi_b   = B_phase v_b
A_i^b   = g_b exp(delta_b[i]) R(phi_b[i]).
```

`B_amp` excludes the constant mode, so `mean(delta_b)=0` and it cannot
duplicate the global gain `g_b`. Both bases have mutually orthogonal,
unit-RMS columns. All coordinates initialize to zero and all gains to one,
giving exact equality to the fixed tied carrier.

This differs from Phase 33 in two useful ways. First, tied smooth modes alter
the carrier without simultaneously introducing Q/K disagreement. Second,
rank 4 replaces hundreds of independently moving band parameters with a
condition-number-one, coherent deformation. Unit-RMS scaling avoids a hidden
`1/sqrt(P)` carrier step that Adam would not correct. The final split rung then tests
whether untied Q/K transforms add anything after smoothness is enforced.

## Eight-arm matrix

Every arm uses h768/d8, eight heads, context 1024, batch 8, seed and paired
initialization 123, a 200-step warmup, and one common 20,000-step linear
schedule. Carrier parameters are excluded from weight decay: the global gain
is anchored at one, so decay toward zero would impose an unintended prior to
erase the carrier. Ordinary model weights retain weight decay `0.01`.

| Backbone | Carrier mode | Per-layer learned carrier parameters | Role |
| --- | --- | ---: | --- |
| fixed RoPE | `tied_scalar` | 1 | RoPE anchor |
| fixed RoPE | `tied_smooth_amplitude` | 5 | smooth amplitude increment |
| fixed RoPE | `tied_smooth_polar` | 9 | smooth phase increment |
| fixed RoPE | `split_smooth_polar` | 18 | Q/K untying increment |
| NoPE | `tied_scalar` | 1 | NoPE anchor |
| NoPE | `tied_smooth_amplitude` | 5 | smooth amplitude increment |
| NoPE | `tied_smooth_polar` | 9 | smooth phase increment |
| NoPE | `split_smooth_polar` | 18 | Q/K untying increment |

The primary contrasts are nested within each backbone. The matched RoPE-minus-
NoPE contrast at every rung is secondary and tests whether a carrier transform
substitutes for or complements standard RoPE.

## Measurement and optimization gate

Evaluate every 2,000 steps on the same 128-example development window and at
20,000 on the disjoint 1,024-example final window beginning at validation batch
2,048. Save paired example losses, learned amplitude/phase spectra by layer,
throughput, memory, and the sparse intervention-optimization trace.

For every candidate inspect:

- endpoint paired loss and 95% confidence interval versus its direct parent;
- the full gap curve and its slope over the final 5,000 steps;
- raw/clipped gradient concentration, Adam `sqrt(v)`, momentum alignment, and
  realized parameter updates;
- carrier-function movement relative to parameter movement;
- learned DCT coordinates, effective amplitude, phase, and Q/K disagreement.

A candidate is eligible for longer training—not yet seed replication—only if
its endpoint gain is at least `0.003` nats, the paired interval excludes zero,
the advantage is not clearly collapsing late, and the optimization/function
trace is finite and active. A null result with healthy functional movement is
evidence against the intervention. A null result with severe conditioning or
moment suppression motivates at most one targeted optimizer repair, not a
broad sweep.

## Scope limits

This phase does not vary rank, frequency, mapper family, head granularity,
content conditioning, or AddRoPE location. It does not combine pre-Q/K and
AddRoPE carriers. Those axes remain deferred unless this coherent static test
produces a clear signal.

## Result and disposition

All eight arms completed. Here the `tied_scalar` parent has a learned scalar
gate in each layer; “fixed” refers to its spectral shape, not a frozen global
amplitude. The smooth amplitude rung therefore isolates four zero-mean DCT
coordinates per layer.

- With NoPE, smooth amplitude improved final loss by `-0.010644`, paired-example
  95% CI `[-0.011700,-0.009589]`, and passed the direct-parent screen gate.
- With fixed RoPE, smooth amplitude improved by `-0.002604`, CI
  `[-0.003194,-0.002013]`. This is likely a real seed-123 early-training effect,
  but it missed the predeclared `0.003` practical threshold.
- Adding phase changed loss by `+0.000141/+0.000142` under RoPE/NoPE. It was
  null despite finite, substantial parameter and carrier-function movement.
- Splitting Q/K changed loss by `-0.000264/-0.001506` under RoPE/NoPE. Neither
  cleared the practical threshold.
- Smooth amplitude recovered about 17.8% of the scalar-carrier RoPE-versus-NoPE
  gap, but amplitude+RoPE still beat amplitude+NoPE by `0.037168` nats.
- The RoPE amplitude arm emphasized the lowest-frequency quartile over the
  highest by 1.54x--3.48x in every layer. NoPE learned heterogeneous layer
  profiles spanning 0.26x--3.19x. The spectral coordinates therefore did real
  work beyond the scalar gate, though this one-seed pattern is descriptive.

All optimizer traces were finite and unclipped. Median descent-update/current-
gradient cosine was positive for every arm, and 78.6%--96.4% of sampled
nonzero updates had positive alignment. Late negative samples occurred as the
linear learning rate approached zero, including in scalar controls, with tiny
functional steps. The phase null is therefore scientific evidence rather than
an obvious optimization failure.

No seed or 200k expansion is automatic. Smooth NoPE amplitude is retained as a
conditional candidate only if a specifically RoPE-free architecture becomes
the target; it is not the best absolute model. Phase and Q/K splitting are not
promoted. Full paired curves and compact reconstructible DCT profiles are in
[`results/phase35_smooth_carrier_20k/PHASE35_RESULTS.md`](results/phase35_smooth_carrier_20k/PHASE35_RESULTS.md).
