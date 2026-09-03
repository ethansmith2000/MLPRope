# Phase 34 shared-frequency preflight

All five 50-step h768/d8 arms completed compiled bf16 forward, backward,
optimizer update, final evaluation, diagnostics, profile serialization, and
completion marking through `gpu-claim`.

| Arm | Target tok/s | Peak reserved MiB | Frequency parameters |
| --- | ---: | ---: | ---: |
| rope-fixed | 190,825 | 5,076 | 0 |
| rope-global-log | 182,630 | 5,096 | 48 |
| qkpre-fixed | 189,169 | 5,220 | 0 |
| qkpre-frequency-log | 187,066 | 5,220 | 384 |
| qkpre-frequency-horizon | 189,837 | 5,220 | 384 |

Both learned carrier runs initially completed training but exposed a diagnostic
cache-length mismatch: next-token evaluation presents 1,023 inputs from a
nominal 1,024-token row, while the globally shared carrier cache had length
1,024. Commit `966b897` made the preprojection cache obey the same safe-prefix
contract as the RoPE cache and added a regression test. Both arms then passed
from a clean source tree.

At step 50, all three learned spectra were finite, strictly positive, and had
zero ordering violations. The coordinates had moved away from their exact
fixed anchors:

| Arm | Multiplier range | Endpoint-phase RMS | Endpoint-phase max |
| --- | ---: | ---: | ---: |
| rope-global-log | [0.999112, 1.000948] | 0.059679 | 0.357910 |
| qkpre-frequency-log | [0.998732, 1.001066] | 0.051767 | 0.384583 |
| qkpre-frequency-horizon | [0.992289, 1.006785] | 0.000458 | 0.001269 |

The 30-step post-warmup timing window is too short to make a throughput claim.
In particular, the measured 4.3% learned-RoPE deficit should be monitored over
the long run rather than treated as established overhead.
