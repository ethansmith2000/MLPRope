# Phase 39A: carrier location and direct AddRoPE

All runs use h768/d8, seed 123, 30k steps, method-aware Q/K RMSNorm, and the same disjoint 1,024-example holdout.

| Arm | Final loss | vs fixed RoPE | 95% paired interval | tok/s | position params |
| --- | ---: | ---: | ---: | ---: | ---: |
| rope-fixed | 3.794820 | reference | — | 181924 | 0 |
| qkpre-rope | 3.721015 | -0.073805 | [-0.075559, -0.072100] | 181521 | 0 |
| input-rope | 3.781651 | -0.013170 | [-0.014633, -0.011701] | 182331 | 0 |
| addrope-fixed-rope | 3.775458 | -0.019362 | [-0.020843, -0.017867] | 182366 | 0 |
| addrope-direct-nope | 3.741599 | -0.053221 | [-0.055066, -0.051386] | 181466 | 0 |
| addrope-direct-rope | 3.767408 | -0.027412 | [-0.028910, -0.025909] | 180104 | 0 |

## Mechanistic contrasts

Negative values favor the first named arm.

| Contrast | Delta | 95% paired interval |
| --- | ---: | ---: |
| qkpre-rope_minus_input-rope | -0.060635 | [-0.062309, -0.059003] |
| qkpre-rope_minus_addrope-fixed-rope | -0.054443 | [-0.056254, -0.052680] |
| addrope-direct-rope_minus_addrope-fixed-rope | -0.008050 | [-0.008760, -0.007341] |
| addrope-direct-rope_minus_addrope-direct-nope | +0.025809 | [+0.023981, +0.027648] |

Paired-example intervals quantify holdout precision within one training seed. Phase 39A is a breadth screen, not seed-robust publication evidence.
