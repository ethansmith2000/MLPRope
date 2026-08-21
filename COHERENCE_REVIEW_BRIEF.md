# External review brief — coherent aggregate modification of RoPE

_Prepared 2026-08-21 for an independent reviewer (GPT-sol-5) with no access to
this repository. Everything needed is inlined. We want adversarial technical
review and constructive proposals, not validation._

---

## 1. What to review

We have a specific open design problem and we want your independent view on it.

> **The problem.** We want to modify RoPE using a learned function with real
> expressivity (an MLP), but we suspect that letting each token's positional
> embedding move in its own direction destroys something important. RoPE gives
> every *pair* of positions a consistent geometric relationship; an
> unconstrained per-token modification does not. We want modifications that are
> **expressive yet coherent in aggregate across the sequence**.
>
> What is the right formal statement of that constraint, what families of
> modification satisfy it, and which are worth training?

We have a derivation and a hypothesis (§5). Please attack them, and propose
alternatives we have not considered.

---

## 2. Setting (fixed; not under review)

- Decoder-only GPT, h768/d8 and h1024/d12, 8 heads, **fixed context 1024**.
- OpenWebText, GPT-2 tokenizer. Primary metric: **held-out loss at 1024**.
- **Length extrapolation is explicitly not a goal.** Proposals justified only
  by extrapolation are out of scope.
- Hard constraint: **attention must stay on fused SDPA**. Materialized `[H,L,L]`
  logit biases and FlexAttention are closed (measured ~1.9x throughput cost).
  Anything requiring a custom attention kernel is out of scope.
- Standing evidence rules: three paired seeds (123/456/789) sharing data order
  and per-parameter initialization; promotion requires mean improvement
  >= 0.01 nats with favorable signs in all three seeds; 5k-step screens are
  failure screens only (we have three documented 5k->30k ordering reversals).

Notation: rotary pair index `i` with frequency `w_i = theta^(-2i/d)`, positions
`p`, query position `m`, key position `n`, head dimension `d`.

---

## 3. What we have established (evidence, not belief)

All results are held-out loss; negative favors the candidate.

### Confirmed (3 paired seeds, disjoint 1,024-example holdout, h1024/d12, 30k steps)

| Contrast | Mean | All seeds |
| --- | ---: | --- |
| additive positional carrier vs standard RoPE | **-0.051** | yes |
| MLP-hypernetwork carrier vs plain linear-mapped carrier | -0.002 | no (mixed) |
| carrier vs RoPE + parameter-matched FFN widening | -0.050 | yes |
| content-conditioned carrier vs position-only carrier | -0.011 | yes |

The winning mechanism is **additive**, not a modified rotation: a per-head
carrier `c(p) = a(p) * cis(w_i * p + phi(p))` is **added to q and k**, after
which attention runs unmodified on fused SDPA. Multiplicative RoPE is disabled
on that channel. Adding `c` to both q and k introduces logit terms
`q_m . c_n`, `c_m . k_n`, and `c_m . c_n` that pure RoPE lacks.

### Null or negative (each 3 paired seeds unless noted)

| Intervention | Result |
| --- | ---: |
| learned **static** RoPE frequencies (per layer/head/pair, several parameterizations) | -0.002 (null); a -0.012 result at 5k reversed to -0.0016 at 30k |
| **bounded per-token** phase offset `delta_phi = (p/1024) * tanh(MLP(norm_x))` | -0.002 (null) |
| phase-only rotary modification (no amplitude) | worse than plain RoPE |
| **unbounded content-dependent frequency** `w * exp(g(x))` | catastrophic (~+0.25) |

### Structural findings

- The gain lives almost entirely in the **amplitude** branch; phase-only is
  worse than RoPE.
- An **MLP over position buys nothing** over a plain linear map of the same
  positional features (-0.002, mixed signs), despite ~2x the parameters.
- Content conditioning helps (-0.011) but ablations on the trained models show
  its useful signal is **local**: corrupting content by a 4-token lag costs as
  much as a 64-token lag (cost saturates by lag 4), while a *causal running
  mean* of content is cheaper than any single-token misalignment.
- Zeroing the content path at inference costs +1.24 nats although the path is
  worth only -0.011 end-to-end: endpoint reliance is not contribution.

---

## 4. The pattern we think we see

Every intervention *inside the rotation* has been null or worse. Every
intervention *added alongside it* has worked. We want to know whether the
following explains that, or whether we are pattern-matching on noise.

---

## 5. Our derivation and hypothesis — please attack this

**Claim 1 (exact).** For rotary, apply per-position phase offsets `delta_p` to
both q and k. The logit becomes

```
logit(m,n) = q_m^T R( w*(n-m) + delta_n - delta_m ) k_n
```

This depends only on `(n-m)` **iff** `delta_n - delta_m` is a function of
`(n-m)` alone, **iff** `delta_p` is affine in `p`. And an affine phase offset
`delta_p = c + s*p` is exactly a frequency change `w -> w + s` (the constant
cancels). Therefore:

> The only phase modifications preserving exact translation-relativity are
> affine in position, and those are precisely frequency changes.

If true, this says the "safe" family is *exactly* the family we already tested
and found null, which would be an important negative result about the whole
in-rotation direction.

**Claim 2 (the proposed relaxation).** Let phase be `w_i * tau(p)` for a shared
monotone warp `tau`. Then pairwise angle is `w_i * (tau_n - tau_m)`: not
translation-invariant, but every pair still sees a *consistent* notion of
distance measured in warped coordinates. We call this **pairwise coherence**
and propose it as the right constraint, weaker than translation-invariance:

| Class | Condition | Our result |
| --- | --- | --- |
| translation-invariant | `tau` affine (= frequency change) | null (-0.002) |
| **pairwise-coherent** | shared monotone `tau`, any shape | **untested** |
| incoherent | independent per-token `delta_p` | null (-0.002) |
| divergent | offset grows with `p` and depends on content | catastrophic |

**Hypothesis.** An MLP is not too expressive; it is expressive in the wrong
coordinates. Applied to per-token phase it can leave the coherent manifold and
must spend capacity learning not to. Applied to a *shared warp* `tau` it cannot
leave that manifold by construction, so its expressivity should become usable.
This predicts our MLP-vs-linear null and predicts a monotone-warp MLP would do
better.

**Claim 3.** Additive carriers face a weaker constraint than rotary ones,
because the carrier is added rather than multiplied into content: its cross
terms are inherently absolute-position signals (this is the point), so
"incoherence" merely means learning different biases rather than corrupting a
geometric invariant shared by all pairs. This would explain why the additive
site keeps winning.

---

## 6. Questions we want answered

1. **Is Claim 1 correct and is it tight?** Are there phase modifications that
   preserve translation-relativity but are not affine in position (e.g.
   exploiting the block structure across pairs, or per-pair offsets that cancel
   in the logit sum)? If the claim is tight, is the whole in-rotation direction
   closed for in-distribution gains?
2. **Is "pairwise coherence" the right formalization** of "coherent in
   aggregate"? Propose a better one if you have it. Is monotonicity of `tau`
   necessary, or only injectivity/smoothness? What breaks if `tau` is
   non-monotone?
3. **How should a warp be parameterized** so it is expressive, monotone, exactly
   RoPE at initialization, numerically safe at L=1024, and SDPA-compatible?
   Candidates we know of: cumulative sums of positive increments; monotone
   splines; integrals of a positive network. What are the failure modes at our
   scale, and which would you run first?
4. **Should the warp be shared or per-head/per-frequency?** A per-frequency warp
   `tau_i(p)` still gives each pair a consistent distance, but different pairs
   would disagree about relative distance. Does that violate the spirit of the
   constraint, and does it matter?
5. **Given our result that the useful content signal is local (saturating by 4
   tokens) while a causal running mean is cheaper than a misaligned token**, is
   a content-driven cumulative clock still motivated? These two facts seem to
   pull in opposite directions and we would value an outside reading.
6. **Amplitude vs phase.** Amplitude modulation preserves the rotation angle
   entirely and acts like a per-position gain/temperature, and empirically it
   carries our gain. Is there a principled reason amplitude is the "safe"
   expressive channel, and is there an amplitude analogue of the coherence
   constraint we are missing?
7. **What are we not asking?** Name the most important consideration absent
   from this brief.

---

## 7. Ground rules for the review

- Assume fixed context 1024 and in-distribution loss. Do not optimize for
  extrapolation.
- Any proposal must run on **fused SDPA** with no custom kernel, and must have
  an **exact zero-initialized null** that reproduces the current baseline
  (either standard RoPE or the confirmed additive carrier) at step 0.
- Prefer one concrete, well-argued construction with a stated failure mode over
  a survey of options.
- If you believe a claim above is wrong, say so directly and show the
  counterexample. Refuting Claim 1 or Claim 2 is more valuable to us than
  agreeing with them.
- Effect sizes below 0.01 nats are not material at our scale; do not propose
  mechanisms whose expected gain is smaller unless the mechanistic insight is
  the point.
