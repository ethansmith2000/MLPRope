# Phase-30 qkpre x AddRoPE factorial screen

Paired seed 123, 15k steps, with a disjoint 256-example holdout.
AddRoPE is the additive Q/K replacement for multiplicative RoPE; the
combined cell adds qk-preprojection upstream of that additive channel.

| Arm | Final loss | Target tok/s |
| --- | ---: | ---: |
| Fixed RoPE | 4.022989 | 183,961 |
| qkpre + RoPE | 3.945626 | 183,473 |
| AddRoPE a1.0 | 3.924575 | 170,159 |
| qkpre + AddRoPE a1.0 | 3.929509 | 162,226 |

| Contrast | Delta | Paired 95% CI |
| --- | ---: | ---: |
| qkpre-rope_vs_rope-fixed | -0.077364 | [-0.080528, -0.074199] |
| addrope-a10_vs_rope-fixed | -0.098415 | [-0.103933, -0.092896] |
| qkpre-addrope-a10_vs_rope-fixed | -0.093480 | [-0.098745, -0.088215] |
| combo_vs_qkpre-rope | -0.016117 | [-0.020659, -0.011574] |
| combo_vs_addrope-a10 | +0.004934 | [+0.002487, +0.007382] |

Factorial interaction: **+0.082298** (95% paired-example CI [+0.078148, +0.086448]); classification: **sub_additive_or_redundant**.

The best single arm is **AddRoPE a1.0**. Combo minus best single is +0.004934 (CI [+0.002487, +0.007382]).

The JSON companion contains mechanism-health diagnostics and the
fully verified factor blocks.
