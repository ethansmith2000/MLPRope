# Phase-29 qkpre x AddRoPE factorial screen

Paired seed 123, 5k steps, with a disjoint 256-example holdout.
AddRoPE is the additive Q/K replacement for multiplicative RoPE; the
combined cell adds qk-preprojection upstream of that additive channel.

| Arm | Final loss | Target tok/s |
| --- | ---: | ---: |
| Fixed RoPE | 4.598101 | 173,930 |
| qkpre + RoPE | 4.516656 | 172,387 |
| AddRoPE a1.0 | 4.430891 | 144,644 |
| qkpre + AddRoPE a1.0 | 4.451152 | 138,911 |

| Contrast | Delta | Paired 95% CI |
| --- | ---: | ---: |
| qkpre-rope_vs_rope-fixed | -0.081444 | [-0.084153, -0.078736] |
| addrope-a10_vs_rope-fixed | -0.167210 | [-0.175862, -0.158558] |
| qkpre-addrope-a10_vs_rope-fixed | -0.146949 | [-0.154648, -0.139250] |
| combo_vs_qkpre-rope | -0.065505 | [-0.072351, -0.058659] |
| combo_vs_addrope-a10 | +0.020261 | [+0.018117, +0.022405] |

Factorial interaction: **+0.101705** (95% paired-example CI [+0.098118, +0.105293]); classification: **sub_additive_or_redundant**.

The best single arm is **AddRoPE a1.0**. Combo minus best single is +0.020261 (CI [+0.018117, +0.022405]).

The JSON companion contains mechanism-health diagnostics and the
fully verified factor blocks.
