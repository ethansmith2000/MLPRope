# Phase 35: smooth sinusoidal-carrier screen

_Frozen protocol, 2026-09-04._

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
let

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
unit-RMS columns. All
coordinates initialize to zero and all gains to one, giving exact equality to
the fixed tied carrier.

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
