# Position Embedding Experiments: Design Doc

## Background

We've been evaluating variants of RoPE and related position-encoding schemes, motivated by:

1. **MLPRoPE** — a variant where sinusoidal position codes are run through a per-head MLP and *added* to Q/K (rather than multiplied, as in RoPE). Inspired by Jonathan Chang's [Additive Rotary Embedding](https://jonathanc.net/blog/additive-rotary-embedding).
2. **Inkling's content-conditioned relative bias** — the query's hidden state produces a mixture over a learned bank of distance profiles, yielding a per-query, per-head relative-position bias added to attention logits.
3. Prior findings on **entropy calibration** (visible-key scaling) — mostly separate, tracked elsewhere.

The through-line: standard attention has a fixed temperature and a position-independent dot product. All the variants relax one of those constraints, giving the model more degrees of freedom in how content and position interact.

## Key insight: the unifying interface

Every variant we care about produces a **bias tensor added to attention logits** via FlexAttention's `score_mod`. The bias is never materialized as a full `[seq, seq]` matrix — `score_mod` runs inside the fused kernel and reads from precomputed state indexed by `(q_idx, kv_idx)`.

Interface:

```python
bias = position_module(hidden_states, positions)  # semantically [B, H, Q, K]
                                                   # never allocated in full
```

The variants differ only in what `position_module` does internally.

## Design decisions we landed on

### 1. Add-to-logits, not add-to-Q/K

MLPRoPE originally added to Q/K before the dot product. **We're standardizing on add-to-logits (via `score_mod`) across all variants.** Reasons:

- Cheaper (no per-token position code interacting with projections)
- Inkling requires it anyway
- Makes the ablation clean — "does content-conditioning help" is a fair comparison only if position-only and content-conditioned variants hit attention through the same channel

If we later want to separate "add-to-logits itself is the win" from "content-conditioning is the win," we can add an add-to-Q/K version of each position-only variant. Not the priority.

### 2. Keep RoPE on Q/K as the geometric prior

Learned bias is a **delta on top of RoPE**, not a replacement. Init the learned piece near-zero so at step 0 the model behaves like a pure RoPE baseline. This means if the learned piece is unhelpful we degrade gracefully to a known-good baseline, not to noise.

### 3. Per-head granularity

Bias tensor is per-head (`[B, H, Q, K]` semantically). Not per-dim (interacts badly with RoPE geometry, especially post-RoPE), not shared across heads (heads specialize). This matches the granularity that fell out of the entropy-scaling analysis for a different question.

### 4. Anchor to identity at init

Every learned modification starts as a small delta from a known-good baseline:
- Position-only variants: `+ embs` residual from the raw sinusoids, MLP output initialized small
- Content-conditioned: mixing weights softplus-with-identity-backward, initialized so the produced bias ≈ 0
- Positivity where needed: softplus, not exp (avoids the exp-derivative explosion issue)

### 5. Same parameter budget across variants

The honest ablation isn't "new thing vs RoPE" — it's "new thing vs RoPE with the extra parameters spent somewhere else, e.g. wider FFN." At minimum, match parameter counts across position variants.

## Direction 1 (fold-in of what was originally "Directions 1 and 4"): Position-only expressiveness sweep

**Hypothesis:** As we make `f(pos) → bias` more expressive, does loss keep improving, or does it plateau? Finding the plateau tells us whether extra position machinery beyond a certain point is wasted.

**Variants (in order of expressiveness):**

1. **RoPE baseline** — standard multiplicative RoPE, no added bias. Control.
2. **AddRoPE** — learned per-frequency scale + offset on sinusoidal basis, add-to-logits.
3. **Linear** — per-head linear transform over the sinusoidal basis.
4. **Low-rank** — per-head `dim → r → dim` bottleneck, `r` small (say 16-32).
5. **MLPRoPE** — per-head 2-layer MLP over the sinusoidal basis (roughly what the existing MLPRoPE code does, but adapted to add-to-logits).

All produce a bias curve as a function of relative distance, materialized as `[B, H, Q, rel_extent]` and read via `score_mod`.

## Direction 2: Content-conditioned relative bias (Inkling-style)

**Hypothesis:** Letting the query's content decide which relative-distance pattern to apply is a *new capability* — no position-only variant can express "this query wants distance profile A, that query wants distance profile B." If this direction wins over the best Direction 1 variant at matched parameters, content-conditioning is doing real work.

**Variants:**

1. **Inkling-table** — literal Chang/Inkling design. `relative_states = linear(hidden_states)` produces `[B, Q, H, d_rel]`. `proj: [d_rel, rel_extent]` is a learned parameter. `bias_curve = relative_states @ proj` gives `[B, Q, H, rel_extent]`. Read via `score_mod` at `q_idx - kv_idx`, masked to `0 <= distance < rel_extent`, causal mask handled separately.
2. **Inkling-CosNet** — replace the `[d_rel, rel_extent]` lookup table with `d_rel` CosNet-parameterized functions of relative distance (learnable-frequency cosine features + tiny MLP, à la the NOBLE paper). Gets extrapolation beyond `rel_extent` for free and uses fewer parameters. This is the "how do our internal Canva Research findings compose with Inkling" experiment.

**Hyperparameters to think about:**
- `d_rel` — the "vocabulary size" of relative-position patterns. Start small (16-32). Too small → collapses to ALiBi. Too large → overfits and wastes parameters.
- `rel_extent` — for the table variant, the bounded distance range. Set to something like training seqlen or half of it. Beyond `rel_extent`, bias is zero and content-based attention takes over. This hard cutoff is a *feature* for length extrapolation.
- For CosNet variant, no hard cutoff, but pay attention to what the learned frequencies do at long distances.

## What we're NOT doing here

**Entropy calibration / visible-key scaling** is a separate research thread, being handled elsewhere. It lives on Q/K/logit-scale, has different diagnostics (multiplier curves, attention entropy vs count), and mixes it in here would confound the position ablation. Keep it in its own file.

Notes handed to the other thread already: bounded softplus multiplier with identity backward, `log(log1p(n))` input, no intercept, standard SDPA scaling as anchor, 2×2 with denominator/null-token bias.

## Implementation notes

### FlexAttention `score_mod` pattern

```python
def make_score_mod(bias_curves, rel_extent):
    # bias_curves: [B, H, Q, rel_extent], precomputed by the position module
    def score_mod(score, b, h, q_idx, kv_idx):
        distance = q_idx - kv_idx
        in_range = (distance >= 0) & (distance < rel_extent)
        bias = torch.where(in_range, bias_curves[b, h, q_idx, distance.clamp(0, rel_extent - 1)], 0.0)
        return score + bias
    return score_mod
```

For the CosNet variant, replace the table lookup with a call to the CosNet function on `distance` inside `score_mod` (assumes the CosNet is cheap; if not, materialize `[H, rel_extent_large]` once and index).

### Shape audit

For Inkling-table:
- Hidden state to `relative_states`: `nn.Linear(dim, n_heads * d_rel, bias=False)` → reshape to `[B, Q, H, d_rel]`.
- Profile bank: `nn.Parameter(torch.empty(d_rel, rel_extent))`, initialized small (e.g. `nn.init.trunc_normal_(std=0.02)`).
- Bias curves: `relative_states @ proj` → `[B, Q, H, rel_extent]`.
- Transpose to `[B, H, Q, rel_extent]` before feeding to `score_mod`.

For position-only variants (Direction 1):
- Input: sinusoidal basis at distances `0, 1, ..., rel_extent - 1`.
- Output: `[H, rel_extent]` (per-head bias curve, no batch or query dependence).
- Broadcast to `[B, H, Q, rel_extent]` inside score_mod (i.e., all queries share the same curve for a given head).

### Init strategy

- **Position-only variants:** initialize output layers so bias ≈ 0 at step 0. E.g., zero-init the final linear/MLP layer, or scale by a small factor and add a residual from a fixed sinusoid so the initial output is bounded but small.
- **Content-conditioned:** zero-init the `proj` bank OR zero-init the `relative_states` linear. Either way, bias starts at 0 and grows during training.

### Parameter matching

- RoPE baseline: no extra params.
- AddRoPE: ~`n_freqs` per head. Small.
- Linear position-only: `n_heads * head_dim * head_dim` per layer if fully expressive, less if low-rank.
- Low-rank: `2 * n_heads * head_dim * r`.
- MLPRoPE (add-to-logits version): depends on hidden dim.
- Inkling-table: `dim * n_heads * d_rel + d_rel * rel_extent`. With `d_rel=32, rel_extent=512`, this is dominated by the first term.
- Inkling-CosNet: `dim * n_heads * d_rel` for routing + small CosNet parameters shared across profiles.

Log parameter counts for each and match them (or match to within a factor of ~1.5) via bumping `d_rel`, `r`, or hidden dim of MLPs.

### Diagnostics to log

- Bias magnitudes per layer, per head (histograms).
- For content-conditioned: distribution of mixing weights per query — is it near-uniform (routing not working) or peaked (routing engaged)?
- Effective distance profile for a few representative queries across depths. Plot them. If they look like uniform noise, mechanism is broken. If they look like interpretable curves (peak-at-1, peak-at-8, smooth-decay), the mechanism is working.
- Attention entropy per layer for each variant — the position bias will shift this and it's worth watching.
- Length extrapolation: eval at seqlen > training seqlen. Position-only variants with hard cutoffs (or Inkling-table) should show a clean handoff to content attention beyond the cutoff; RoPE tends to degrade in a messier way.

## Experiment plan

**Phase 1: Position-only sweep (Direction 1).**
- Train each of the 5 variants at matched compute and (approximately) matched parameters.
- Report loss curves, final loss, and length-extrapolation curves.
- Goal: find where the position-only expressiveness curve plateaus.

**Phase 2: Content-conditioned comparison (Direction 2).**
- Train the two content-conditioned variants at the same budget as the best Direction 1 result.
- Same reporting.
- Goal: does content-conditioning clear the position-only ceiling?

**Phase 3 (only if Phase 2 shows a real win):** Ablations on the content-conditioned variant.
- Sweep `d_rel`.
- Compare shared vs per-layer distance profile bank.
- Try dropping RoPE entirely under content-conditioning (does it still need the geometric prior?).

Do not mix in entropy-calibration experiments during any of these phases.

## Open questions / things to watch

- The MLPRoPE code we started from has a shape bug (see the review): `nfreqs // n_heads` produces a per-head output smaller than `head_dim`. When porting to add-to-logits, this shape issue disappears (output is now bias-shaped, not Q-shaped), but flag it if it comes back.
- Content-conditioning at inference breaks a small nice property of RoPE: `relative_states` depends on the query's hidden state, so it recomputes per generation step. This is fine but affects wall-clock comparisons.
- The `+ embs` residual pattern from the original MLPRoPE code is a good idea and should be preserved in any position-only variant that has capacity to fail — always fall back to a known-good baseline at init.

## Remaining TODOs

- Implement Phase 2's `inkling_table` and `inkling_cosnet` variants. Their config and model dispatch points are scaffolded, but deliberately raise until the content-conditioned path is implemented.
- Implement `suggest_matched_baselines`: match position variants by parameter count (`pos_rank`, MLP width, or FFN width) and add a short timing probe for wallclock matching.
- Add the planned diagnostics: per-layer/per-head bias histograms, content-routing statistics, representative distance profiles, attention entropy, and length-extrapolation evaluation.
- Add the add-to-Q/K version as a later ablation against the primary add-to-logits implementation.
- If Phase 2 wins, run the Phase 3 ablation that removes RoPE under content-conditioning and compare shared versus per-layer profile banks.
