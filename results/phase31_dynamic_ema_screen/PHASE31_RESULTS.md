# Phase-31 causal-EMA dynamic-position screen

This is a paired seed-123, 15k-step mechanism screen. Final loss uses
the disjoint 256-example holdout beginning at validation batch 2,048.
Negative deltas favor the candidate; seed replication is deferred until
a mechanism shows a clear effect.

| Arm | 5k dev | 10k dev | 15k dev | Final | Tok/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| rope-fixed | 4.318739 | 3.994052 | 3.874938 | 4.022984 | 183,352 |
| clock-pointwise | 4.321578 | 3.994960 | 3.874837 | 4.022623 | 174,914 |
| clock-ema | 4.321041 | 3.994306 | 3.875032 | 4.022622 | 165,339 |
| addrope-position | 4.214719 | 3.913112 | 3.795465 | 3.933450 | 165,999 |
| addrope-content-pointwise | 4.202404 | 3.899219 | 3.786864 | 3.918615 | 156,094 |
| addrope-content-ema | 4.188289 | 3.890894 | 3.774810 | 3.907515 | 102,058 |

| Candidate | Direct control | Final delta | 95% paired-example CI | Triage |
| --- | --- | ---: | --- | --- |
| clock-pointwise | rope-fixed | -0.000360 | [-0.001013, +0.000292] | unresolved |
| clock-ema | rope-fixed | -0.000362 | [-0.001061, +0.000337] | unresolved |
| clock-ema | clock-pointwise | -0.000001 | [-0.000686, +0.000683] | unresolved |
| addrope-position | rope-fixed | -0.089533 | [-0.094497, -0.084569] | survive |
| addrope-content-pointwise | addrope-position | -0.014836 | [-0.017573, -0.012099] | survive |
| addrope-content-ema | addrope-content-pointwise | -0.011100 | [-0.014211, -0.007989] | survive |
| addrope-content-ema | addrope-position | -0.025935 | [-0.029118, -0.022753] | survive |

The JSON companion includes learned EMA windows, clock drift/phase
health, carrier-delta magnitudes, throughput, and peak memory.
