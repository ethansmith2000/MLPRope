# Position Embedding Experiments: Dual-Channel Design

## Background

This project studies two related but distinct ways to extend positional
information in attention:

1. **Q/K position transforms** modify the query and key vectors before their dot
   product. This includes additive MLPRoPE-style embeddings and learned phase
   residuals composed with standard RoPE.
2. **Logit biases** add a scalar function of relative distance after the Q/K dot
   product. This includes the completed position-only sweep and future
   Inkling-style content-conditioned profiles.

These channels should not be forced through one interface. They can be enabled
independently or together, and each channel has its own feature-map and sharing
choices.

Entropy calibration / visible-key scaling remains a separate research thread.

## Baseline and Phase-1 result

The baseline is standard multiplicative RoPE with no learned Q/K residual and no
learned logit bias.

Phase 1 kept RoPE on Q/K and tested position-only logit-bias deltas. At 10k steps:

| Variant | Eval loss | Perplexity |
| --- | ---: | ---: |
| RoPE baseline | 4.129 | 62.1 |
| AddRoPE logit bias | 4.126 | 61.9 |
| Low-rank implementation used in Phase 1 | 4.102 | 60.5 |
| MLP logit bias | 4.095 | 60.0 |
| Linear logit bias | **4.076** | **58.9** |

The completed low-rank run was confounded: the document described
`D → r → D`, but the code implemented `D → r → GELU → scalar`. It is retained as
an empirical result, but it is not the factorized-linear ablation its name
implied.

## Independent channel configuration

Conceptually, the model exposes:

```yaml
qk:
  enabled: false
  feature_map: identity
  sharing: per_head
  apply: phase_residual
  rank: 32
  mlp_hidden: 128

logit_bias:
  enabled: false
  feature_map: identity
  sharing: per_head
  rank: 32
  mlp_hidden: 128
```

With both channels disabled, attention is the RoPE baseline. Enabling both is a
valid experiment: Q/K receives a learned geometric transform and the resulting
dot product receives a learned distance bias.

### Channel-local semantics

| Axis | Q/K channel | Logit-bias channel |
| --- | --- | --- |
| Output | AddRoPE addend (`head_dim`), or `D/2` phase deltas | Scalar curve over relative distance |
| `shared_head` | One vector/phase bank broadcast to all heads | One feature map and readout shared across heads |
| `per_head` | Independent vector/phase bank per head | Independent feature map and readout per head |
| `full_dim` | Joint map in model dimension, reshaped into heads | Joint full-dimension map, then per-head scalar readout |
| Identity initialization | AddRoPE: sinusoid addend (no `R`); phase: zero `δ` ⇒ `R=I` | Zero scalar readout |

`feature_map` and `sharing` therefore have channel-local consequences even
though both channels use the same vocabulary.

## Feature-map taxonomy

Let the sinusoidal basis at a position or distance be `x ∈ R^D`.

1. **Identity:** `f(x) = x`.
2. **AddRoPE affine:** learned per-frequency scale and offset, initialized as
   identity.
3. **Linear:** full `D → D` affine map.
4. **Low-rank:** purely linear residual factorization
   `f(x) = x + up(down(x))`, with `D → r → D`.
5. **Bottleneck MLP:** nonlinear residual
   `f(x) = x + up(GELU(down(x)))`, with `D → r → D`.
6. **MLP:** the same nonlinear residual with an independently selected hidden
   width rather than the low-rank budget.

Biases are enabled in affine layers and initialized to zero. Residual output
layers are zero-initialized so low-rank and MLP feature maps begin as identity.
Logit-channel readouts and Q/K phase readouts are zero-initialized so those
channels have zero effect at step 0. True AddRoPE (`qk.apply=add`) has **no**
zero readout: the feature map *is* the addend, so identity/residual maps start
as sinusoids.

This taxonomy keeps output shape fixed while varying the feature map. In
particular, low-rank and bottleneck MLP are separate ablations.

## Q/K channel

The Q/K channel consumes absolute-position sinusoidal features and emits a
vector-valued transform.

### Additive Q/K (true AddRoPE)

`qk.apply="add"` **replaces** multiplicative RoPE, following
[Additive Rotary Embedding](https://jonathanc.net/blog/additive-rotary-embedding):

```text
q'(p) = q(p) + e(p)
k'(p) = k(p) + e(p)
```

with no `R(θ)q`. Here `e(p)` is the Q/K feature-map output over absolute-position
sinusoids (`cis(θ·p)` in real interleaved form). That is the blog baseline when
`feature_map=identity` or `add_rope` (learned per-frequency scale/offset, identity
at init), and the intended extension when `e = f(cis)` or `e = cis + f(cis)` via
linear / low-rank / bottleneck / MLP maps.

This is **not** `RoPE(q + e)`. An earlier bug composed the addend under still-on
RoPE and zeroed the addend at init; both are fixed.

### Phase residual

The phase path keeps multiplicative RoPE and predicts `δ(p) ∈ R^(D/2)`:

```text
q'(p) = R(theta(p) + delta(p)) q(p)
k'(p) = R(theta(p) + delta(p)) k(p)
```

Equivalently, apply `R(theta) R(delta)`. At initialization `δ = 0`, therefore
`cos(δ)=1`, `sin(δ)=0`, and `R(δ)=I`. Zero phase means zero effect after
conversion to a rotation, not a zero rotation matrix.

Standard RoPE uses the same frequencies for every head. The sharing axis tests
whether learned geometry should preserve that constraint or specialize by head.

This channel introduces content-position cross terms under AddRoPE:

```text
(q + e_q) · (k + e_k)
  = q·k + q·e_k + e_q·k + e_q·e_k
```

Those terms cannot be reproduced by a position-only scalar logit bias.

## Logit-bias channel

The position-only logit path consumes a sinusoidal basis at relative distances:

```text
basis[R,D]
  → feature_map
  → features[H,R,D]
  → scalar readout
  → bias[H,R]
```

The scalar readout is part of the logit channel, not part of the feature map.
It is zero-initialized, so the initial bias is exactly zero and attention starts
as standard RoPE.

FlexAttention reads the curve without materializing `[B,H,Q,K]`:

```python
distance = query_idx - key_value_idx
in_range = (distance >= 0) & (distance < rel_extent)
bias = bias_curves[head_idx, distance.clamp(0, rel_extent - 1)]
score = score + torch.where(in_range, bias, 0.0)
```

Only this channel requires FlexAttention. Q/K-only variants can use SDPA.

## Direction 1b: corrected position-only and Q/K sweep

Completed at 10k steps (same recipe as Phase 1). Anchors not re-run:

| Variant | Channel | Eval loss | Perplexity |
| --- | --- | ---: | ---: |
| RoPE baseline | — | 4.129 | 62.1 |
| Linear logit bias (Phase-1 best) | logit | **4.076** | **58.9** |
| Corrected low-rank logit `r=32` | logit | 4.081 | 59.2 |
| Bottleneck-MLP logit `r=32` | logit | 4.084 | 59.4 |
| Linear phase residual on Q/K | Q/K | 4.109 | 60.9 |
| MLP phase residual on Q/K | Q/K | 4.131 | 62.2 |

Takeaways: factorized linear nearly matches full linear on the logit channel;
bottleneck nonlinearity adds nothing; Q/K phase residuals underperform scalar
logit bias, and MLP phase does not beat RoPE.

## Direction 1c: true AddRoPE (replace multiplicative RoPE)

Phase 1c tests the AddRoPE family as a **replacement** for RoPE, not a residual
under it:

| Variant | Meaning |
| --- | --- |
| `qk.apply=add`, `feature_map=identity` | Blog AddRoPE: `q + cis(θp)` |
| `feature_map=add_rope` | Blog affine: learned scale/offset on `cis` |
| `feature_map=linear` | `q + Linear(cis)` |
| `feature_map=low_rank` / `mlp` | `q + cis + f(cis)` (residual; starts as AddRoPE) |

Anchor: Phase-1 RoPE baseline. Launcher family: `EXPERIMENT_FAMILY=phase1c`.

## Direction 2: content-conditioned relative bias

The Inkling hypothesis remains: query content can select different
relative-distance profiles, a capability no position-only curve provides.

1. **Inkling table:** query hidden states produce mixture weights over a learned
   bank of bounded distance profiles.
2. **Inkling CosNet:** replace the table with parameterized functions of
   distance for lower parameter cost and extrapolation.

Inkling belongs naturally in the logit-bias channel. It does not require Q/K
position alternatives to use the same channel in unrelated ablations.

A separate future idea is a cheap content-conditioned Q/K residual such as:

```text
x + low_rank_MLP(concat(x, rope_features))
```

That is not part of Direction 1b.

## Diagnostics

At every validation point, log:

- Per-layer logit-bias mean, standard deviation, and absolute maximum.
- Distance-profile snapshots `[H,R]` for the first, middle, and last layers.
- Later, for content-conditioned variants: routing-weight distributions and
  representative query-specific profiles.
- Later, attention entropy and length-extrapolation evaluation.

Profiles should answer whether the winning linear map learns smooth,
head-specialized structure while nonlinear maps learn noisy or redundant curves.

## Deferred work

- Inkling table and CosNet implementation.
- Parameter-matched wider-FFN controls.
- The content-conditioned `x + MLP(concat(x, rope))` Q/K residual.
- Replacing RoPE entirely via multiplicative alternatives other than AddRoPE;
  Phase 1c covers the AddRoPE replacement family.
- Full Cartesian sweeps across channel, feature map, and sharing.
