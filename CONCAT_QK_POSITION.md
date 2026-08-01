# Concatenated Q/K position channels — design note

_Created 2026-07-31. Status: derivation verified, empirical gating probe run, one claim retracted._

## Motivation

Relative logit biases hold the best absolute results in this project
(`4.0345` at h768/d8 10k, `2.9835` at h1024/d12 50k) but require FlexAttention
and run at roughly `0.52x` the throughput of fused SDPA. The question is whether
the same logit structure can be delivered through a mechanism SDPA can execute.

## The core identity

Let `i` be the **query** position and `j` the **key** position. Attention logits are

```
s_ij = (1/sqrt(d)) * q_i . k_j  +  b_h(i - j)
```

Any bias that factors as an inner product of a query-side and a key-side vector can be
folded into the dot product by **concatenating extra dimensions** onto q and k:

```
q' = [q_i, u_i]   k' = [k_j, v_j]   =>   q' . k' = q_i . k_j + u_i . v_j
```

No mask, no `score_mod`, no `[L,L]` materialization. So the question becomes: for which
bias families is `B[i,j] = b_h(i-j)` expressible as `U V^T` with few columns?

### Fourier / DCT form (Toeplitz-preserving, rank 2 per frequency)

```
u_i = [ c_r*cos(w_r*i + phi_r), c_r*sin(w_r*i + phi_r) ]   r = 1..R
v_j = [       cos(w_r*j),             sin(w_r*j)       ]   r = 1..R

u_i . v_j = sum_r c_r * [cos(w_r*i + phi_r)cos(w_r*j) + sin(w_r*i + phi_r)sin(w_r*j)]
          = sum_r c_r * cos(w_r*(i - j) + phi_r)
```

Exact for any bias of that form. `2R` extra Q/K dims; learnables are `c_r, w_r, phi_r`
per head per layer (`3R` values, ~3k for the whole model). Translation invariance is
preserved by construction.

Basis choice matters: use `w_r = pi*r/L` (DCT-style half-period), not `2*pi*r/L`. A DFT
basis is periodic with period `L` and wraps `b(d)` inside the window; the half-period
cosine basis is complete on `d in [0, L)`.

### Free low-rank form (non-Toeplitz, rank 1 per dim)

Let `u_i, v_j in R^r` be free learned per-position vectors. Then `B = U V^T` is an
arbitrary rank-`r` matrix — no translation invariance imposed — and costs only `r`
extra dims (not `2r`, since `u != v`). It strictly subsumes the Fourier form, since
rank-`2R` Toeplitz is a special case.

### The two constraints are orthogonal

This is the key structural point, and it is easy to get backwards:

| Form | Extra Q/K dims | Params/head/layer | Toeplitz | Rank | Kernel |
| --- | ---: | ---: | --- | --- | --- |
| current `[heads, extent]` curve | — | `extent` | yes | full | FlexAttention |
| Fourier-separable | `2R` | `3R` | yes (exact) | `2R` | fused SDPA |
| free low-rank absolute | `r` | `2Lr` | no | `r` | fused SDPA |
| full unstructured | — | `L^2/2` | no | full | FlexAttention |

The existing formulation is **low-parameter but full-rank**. Concatenation requires
**low rank** and says nothing about structure. So the Fourier form is *less* expressive
than what is currently trained (rank-capped), while the free form is *differently*
expressive (non-Toeplitz, which the current form cannot represent at all). There is an
expressiveness trade here, not a strict loss.

## Gating probe (2026-07-31, zero GPU)

Ran against saved `position_profiles/step_*.pt`, which store the learned relative curves
as `[heads, extent]` per layer for layers 0 / mid / last.

### Retracted claim

A first probe took the SVD of `Toeplitz(b)` with hard zeros above the diagonal and found
rank 64–250 for 99% energy. **That bound is invalid.** The causal mask is applied by
attention itself, so a concat bias only has to match `b(i-j)` on the causal support —
entries above the diagonal are free, and forcing them to zero imposes a triangular
cutoff that is itself near-full-rank. The reported ranks measured the mask, not the curve.

### Valid result: DCT truncation error on `d >= 0`

`phase3_promotion` canonical + full linear logit, h768/d8 10k:

| R | mean max abs error | as % of curve range |
| ---: | ---: | ---: |
| 4 | 2.38 – 11.69 | 34 – 40% |
| 8 | 1.32 – 9.32 | 23 – 34% |
| 16 | 0.75 – 6.09 | 18 – 26% |
| 32 | 0.57 – 4.17 | 14 – 18% |
| 64 | 0.39 – 2.95 | 9 – 12% |

Terms for 90% / 99% of DCT energy: `5–35` / `74–175` (h768/d8); `48–112` / `443–819`
(h1024/d12). So the Fourier form is a **lossy** approximation at affordable `R`, not the
exact drop-in originally claimed.

### Why: the learned curves are log-like decays

Sampled `b_h(d)` for `phase3_promotion` layer 0:

```
d:        0     1     2     3     4     8    16    32    64   128   256   512  1023
h0:    2.88  2.82  2.73  2.62  2.52  2.39  2.27  1.92  1.70  1.13  0.85 -0.11 -0.33
h1:    5.21  5.15  5.11  5.05  4.95  4.55  4.15  3.60  2.90  1.70  0.91  0.20 -0.29
h2:    4.43  4.30  4.00  3.67  3.43  3.17  2.77  2.40  2.09  1.07  1.04 -0.19 -0.47
```

Smooth, monotone, decaying by a roughly constant amount per octave — i.e. approximately
`-m_h * log(1 + d) + c_h`. Layer 4 head 2 has the same shape with amplitude `28.6`.

Log-shaped curves are the worst case for a cosine basis: strong curvature near `d = 0`
and a long flat tail, so DCT energy spreads out. This explains the slow convergence
above without implying the curves are noisy or spiky.

**Independent finding worth recording:** the learned relative logit bias is essentially a
**learned per-head attention decay profile** — a generalized, log-shaped ALiBi. This
converges with the `phase10_hyper_geometry` result that the Q/K carrier gain lives almost
entirely in the *amplitude* branch (amplitude-only `4.3024` vs full `4.2840`, phase-only
`4.4348`, worse than RoPE). Both of the project's best directions reduce to per-head
control of how fast attention decays with distance.

## Consequences for what to try

1. **Try the boring fix first.** The bias is content-independent, so it can be
   materialized once as an `[H, L, L]` float tensor and passed to SDPA as `attn_mask`.
   That drops the flash backend but should select the memory-efficient backend, which
   supports a differentiable additive bias. `[8, 1024, 1024]` in bf16 is ~16 MB and is
   not batch-dependent. If mem-efficient-with-bias lands closer to flash than to
   FlexAttention, the entire throughput objection dissolves with no reformulation at
   all. **Benchmark before doing anything clever.** Confirm the selected backend with
   `torch.nn.attention.sdpa_kernel` rather than assuming.
2. **Do not evaluate the Fourier form by distillation error.** The table above measures
   "can `R` terms reproduce a curve FlexAttention already learned". The real question is
   "can a rank-`2R` bias, trained from scratch, reach the same loss" — the model adapts
   to its own constraint. Those are different questions and the second is more
   favourable. The probe should set `R`, not veto the arm. It argues for `R = 16–32`
   (head_dim `96 -> 128` or `160`), not `R = 8`.
3. **`head_dim` 96 + 32 = 128 is the target.** Multiples of 8 are required; 64 and 128
   are the well-tiled flash sizes, and 128 often runs faster per element than 96, so the
   throughput cost may be near zero. Whether a q/k dim differing from the v dim keeps
   the flash path needs a kernel probe, not an assumption.
4. **Normalize the blocks separately.** RMSNorm over 128 dims where 32 carry position
   channels lets position outliers crush the 96 content dims. Use
   `q' = [RMSNorm(q_content) * g_c, pos_block * g_p]` with a learnable per-head `g_p`,
   and pass `scale = 1/sqrt(96)` explicitly to SDPA rather than letting it use
   `1/sqrt(128)`.
5. **Log-ALiBi: fitted 2026-07-31, partial result — see "Closed-form fit" below.**
   Log decisively beats linear ALiBi, confirming the shape, but 2–3 parameters do not
   reproduce the curves well enough to be a drop-in, and the fit degrades with model
   scale.
6. **The free low-rank arm is predicted to lose.** `phase10` orders the arms
   monotonically by how badly they break translation invariance — amplitude (preserved)
   best, phase (bounded violation) slightly worse than RoPE, frequency (violation growing
   linearly in `p`) much worse. A non-Toeplitz absolute bias is maximally
   translation-non-invariant, so the principle predicts it underperforms the Fourier arm.
   That makes it a genuine test of the principle rather than an expressivity grab, but it
   should not be the lead candidate.

## Closed-form fit of the learned curves (2026-07-31, zero GPU)

Per-head least squares on `b_h(d)`, weighted by pair frequency `(L - d)` since a distance
`d` occurs `L - d` times per sequence and an unweighted fit over-penalizes the rarely
exercised tail. `wR2` is the pair-weighted coefficient of determination; error is max abs
over the 95% of pair mass, as a fraction of that head's curve range.

| Model | Layer | linear (ALiBi) | log | log + tau (3p) | log + linear (3p) |
| --- | --- | ---: | ---: | ---: | ---: |
| h768/d8 | 0 | 0.742 | 0.895 | **0.915** | 0.910 |
| h768/d8 | 4 | 0.651 | 0.793 | 0.872 | **0.889** |
| h768/d8 | 7 | 0.749 | 0.874 | **0.933** | 0.918 |
| h1024/d12 | 0 | 0.184 | 0.529 | 0.528 | **0.688** |
| h1024/d12 | 6 | 0.208 | 0.550 | 0.553 | **0.679** |
| h1024/d12 | 11 | 0.124 | 0.418 | 0.416 | **0.584** |

Findings:

- **Linear-in-`d` is the wrong family.** Classic ALiBi reaches `wR2` 0.65–0.75 at h768 and
  0.12–0.21 at h1024; every log form beats it decisively. This corroborates the
  qualitative read that the learned bias is a *log-shaped* decay profile.
- **`tau` grows with depth**: median `7.2 -> 12.7 -> 37.9` across layers 0/4/7 at h768.
  Locality relaxes with depth — early layers are sharply local, later layers flatter and
  longer-range. Interpretable and worth quoting.
- **The fit degrades with scale.** At h1024/d12 the best 3-parameter form reaches only
  `wR2` 0.58–0.69 and `tau` pins at the search floor (`0.5`), i.e. the larger model wants
  sharper near-origin structure than a log allows. The free curve is *using* its freedom,
  and more so as the model grows.
- **Max error stays high**: 16–24% of curve range at h768, 30–38% at h1024.

Caveat, same as for the DCT probe: this is *distillation* fit to a curve FlexAttention
learned freely, not evidence that a scratch-trained 3-parameter model would lose — the
model adapts to its own constraint. Given the parameter reduction (`3` vs `extent` per
head), a single `log + tau` arm is still cheap enough to be worth one run.

### Consequence for the concat route

The log trend is precisely the part that is **not** low-rank-separable, so splitting the
curve into "log trend + residual" does not rescue concatenation: you would still need a
`score_mod` for the trend. Sums of exponentials are rank-1 separable and fit power-law /
log decays well, but `exp(-lambda*(i-j)) = exp(-lambda*i)*exp(+lambda*j)` requires storing
factors spanning `e^(±lambda*L)`, which is only representable for `lambda <~ 0.04`
(decay lengths above ~25 tokens) and loses precision in the dot-product accumulation.
Short-range terms — the ones the curves most need — are exactly the infeasible ones.

**Net assessment of the concat direction: weakened.** The curves are neither spectrally
compact (DCT) nor low-rank in a form concatenation can exploit, and the cheap closed form
that does capture their shape is not separable. The Fourier arm at `R = 16–32` trained
from scratch remains worth one run, but it is no longer the headline. This was established
for zero GPU time rather than after an experiment family, which is the point of the probe.

## Reproducing the probe

Scripts used (scratchpad, not checked in): DCT truncation error and per-head singular
spectra over `position_profiles/step_*.pt`. Re-derive with
`torch.load(..., weights_only=False)["profiles"]["layer_NN"]`, shape `[heads, extent]`.
Note the retraction above: measure error on `d >= 0` only; do **not** impose zeros above
the diagonal.
