# Phase-32 AddRoPE EMA coefficient-axis screen

This is a paired seed-123, 15k-step screen. All four arms run
sequentially under one lifetime GPU claim. Final loss uses the disjoint
256-example holdout beginning at validation batch 2,048. Negative
deltas favor the candidate.

| Arm | 5k dev | 10k dev | 15k dev | Final | Tok/s | Peak MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| addrope-content-pointwise | 4.203760 | 3.899014 | 3.786093 | 3.918633 | 152,866 | 5,966 |
| addrope-content-ema-scalar | 4.188764 | 3.891489 | 3.774872 | 3.908008 | 147,961 | 6,957 |
| addrope-content-ema-per-head | 4.189114 | 3.889820 | 3.773432 | 3.907487 | 112,790 | 13,894 |
| addrope-content-ema-per-dim | 4.189881 | 3.889505 | 3.774215 | 3.907780 | 148,116 | 6,957 |

| Candidate | Direct control | Final delta | 95% paired-example CI | Triage |
| --- | --- | ---: | --- | --- |
| addrope-content-ema-scalar | addrope-content-pointwise | -0.010626 | [-0.013805, -0.007446] | survive |
| addrope-content-ema-per-head | addrope-content-pointwise | -0.011146 | [-0.014199, -0.008094] | survive |
| addrope-content-ema-per-dim | addrope-content-pointwise | -0.010853 | [-0.013968, -0.007738] | survive |
| addrope-content-ema-per-head | addrope-content-ema-scalar | -0.000520 | [-0.001402, +0.000361] | unresolved |
| addrope-content-ema-per-dim | addrope-content-ema-scalar | -0.000227 | [-0.001057, +0.000602] | unresolved |
| addrope-content-ema-per-dim | addrope-content-ema-per-head | +0.000293 | [-0.000407, +0.000994] | unresolved |

The JSON companion includes learned decay/window diagnostics,
carrier-delta magnitudes, elapsed time, throughput, and peak memory.
