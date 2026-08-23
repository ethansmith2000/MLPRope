# Positional-encoding literature review

_Compiled 2026-08-16 from four parallel web sweeps (static spectra; contextual
position; hybrid/absolute mechanisms; novelty + frontier). A prior review, if
one existed, was never committed and is lost with the old box — this file is
git-tracked. Claims below were verified against arXiv/venue pages by the sweep
agents except where flagged **[unverified]**; treat flagged numbers as
directional until read directly. Preprint numbers are self-reported._

Framing used throughout: our setting is **in-distribution eval loss at fixed
context 1024, fused SDPA only** (no materialized logit biases, no
FlexAttention), small GPT (h768/d8, h1024/d12), RoPE as the anchor/prior.
Compatibility tags: **SDPA-OK** / **custom kernel** / **materialized bias**.

**Implementation/novelty update (2026-08-22).** The new
`position/clock.py` path is not a claim that accumulated content-dependent
rotation is a new family. [CARoPE](https://arxiv.org/abs/2507.23083) is the
closest softmax-attention neighbor: it accumulates a token/head-conditioned
scalar whose powers generate the frequency planes. [Selective
RoPE](https://arxiv.org/abs/2511.17388) learns input-dependent arbitrary angles,
uses sigmoid angle gates, learnable frequencies and a short convolution, and
is evaluated primarily with gated linear attention. [PaTH
Attention](https://arxiv.org/abs/2505.16381) is the noncommutative accumulated-
Householder generalization with a custom kernel. Our implementation is a
narrower control: it fixes the standard RoPE spectrum, learns only one bounded
positive local speed per head (or globally), uses an exact standard-RoPE null,
and applies the resulting phase before ordinary fused SDPA. Any contribution
would come from this controlled restriction and the direct comparison with the
confirmed additive carrier, not from claiming the cumulative idea itself.

The June-2026 analysis [Why Do Accumulated Transformations
Extrapolate?](https://arxiv.org/abs/2606.24975) adds an important limit: learned
accumulated rotations can create a useful finite mixing window, but rotation
alone eventually loses control of far attention mass at extreme lengths. This
does not invalidate a fixed-1024 mechanistic screen; it does rule out describing
the clock as an unbounded-context solution without a separate decay/far-mass
mechanism.

---

## 1. Where our questions sit in the field

The field's 2024–2026 energy is overwhelmingly on **length extrapolation**,
which we have excluded. Once extrapolation papers are set aside, the remaining
in-distribution evidence is thin and small-effect — which makes our measured
effect sizes (`0.04` for the carrier, `0.002` frequency nulls) quantitatively
consistent with the literature rather than anomalous. Three cross-cutting
findings matter most for us:

1. **RoPE's geometric spectrum is inefficient but only mildly suboptimal
   in-distribution.** Sub-context-wavelength frequencies are barely used as
   positional signal and get repurposed as sink/bias machinery; fixing this
   buys ~0.4% ppl at best (§2).
2. **Content-conditioned position pays only when the signal is *cumulative*.**
   Every verified in-distribution LM gain in that family integrates content
   over the causal interval (CoPE, FoX, PaTH, stick-breaking, Selective RoPE);
   no paper reports gains from local bounded per-token modulation — our
   phase-23 null matches the literature's silence (§3).
3. **The decay/amplitude component beats the rotation/phase component
   everywhere the two are separated** (FoX vs RoPE, Selective RoPE's
   factorization, Mamba-3's complex-update equivalence) — independently
   mirroring our own amplitude-branch finding (§3, §4).

---

## 2. Direction A — a better static spectrum than geometric theta^(-2i/d)?

### What the literature says

- **Round and Round We Go** (Barbero et al., ICLR 2025, arXiv:2410.06205).
  Gemma-7B heads use the *highest* RoPE frequencies for sharp positional
  attention and the *lowest* as a quasi-position-free semantic channel; the
  long-range-decay folk story is rejected. **p-RoPE** (replace the lowest
  fraction of frequencies with no rotation) trained from scratch at 2B/8k:
  0.75-RoPE val ppl 4.441 vs RoPE(10k) 4.463 vs RoPE(500k) 4.449 vs NoPE
  4.859. The only from-scratch in-distribution win over standard RoPE in this
  family. **SDPA-OK.**
- **Partial RoPE in practice**: GPT-J/NeoX rotate 25% of head dims
  (parity-plus-speed claim, informal evidence); phi family uses 0.4
  **[unverified as a quality choice]**; DeepSeek MLA confines RoPE to a small
  decoupled key. **MHA2MLA** (arXiv:2502.14837) fine-tune ablations at 135M/1.7B:
  keeping only ~12.5% of rotary subspaces is nearly lossless if the kept set is
  uniform/2-norm-selected or high-frequency; keeping only *low* frequencies is
  catastrophic (−5.25 pts at 135M).
- **Base/theta choice is the weakest lever at 1024.** "Base of RoPE Bounds
  Context Length" (Men et al., NeurIPS 2024, arXiv:2405.14591): the bound at
  context 1k is theta ≥ 4.3e3 — 10000 already clears it — and their sweeps show
  ppl is nearly base-insensitive (only retrieval moves). Below-bound bases
  degrade retrieval *silently* (ppl stays fine). "Frequency Bands in RoPE"
  (ICLR 2026, OpenReview PR1PPxvG9Q) **[unverified — abstract only]**: larger
  theta favors interpolation over extrapolation; a high-norm frequency band
  set by theta and train length does the positional work.
- **Learned frequencies work when globally tied — the boundary of our null.**
  **LeRoPE** (arXiv:2607.10134, 2026): one scalar per frequency band **shared
  across all layers and heads** (32 params at d_head 64), log-space, zero-init,
  no weight decay, separate grad clipping. 52M–2.5B Chinchilla-optimal C4 at
  2048: consistent val gains; RoPE needs +3.4% compute to match at 2.5B.
  Learned spectra are consistent across scales: highest frequencies suppressed,
  extra density near wavelength ≈ 2.2×L_train, low end unchanged. Contrast:
  our phases 20–22 gave per-layer(-head) freedom and found ~−0.002.
  **AdaRoPE** (arXiv:2607.19363, 2026): per-head learned frequencies + per-head
  length-dependent temperature, −0.016 loss at 430M–2.7B/8k/100B tokens — but
  confounded with the temperature component **[no isolating ablation found]**.
- **What the "dead" low-frequency dims actually do:** "Rotary Offset Features"
  (arXiv:2503.01832): sub-one-cycle pairs host offset/outlier features that
  implement sink-like biases — removing them via partial RoPE doesn't remove
  the need for sinks, it relocates the mechanism to cleaner unrotated dims.
  **FoPE** (ICML 2025, arXiv:2412.17739) zero-clips sub-cycle frequencies for
  extrapolation reasons; in-distribution it is parity with RoPE.
- **LieRE / ComRoPE** (learned rotation generators): vision-only evidence;
  ComRoPE's theory says any valid offset-robust generalization must commute —
  i.e., is a spectrum choice in some basis — so our per-pair parameterization
  already spans the legal space.

### Synthesis for us

The standard spectrum's inefficiency is real but worth ~0.002–0.005 loss at our
scale — the size of our phase-20/22 null, not of our carrier result. The two
tests actually supported: **partial RoPE** (sweep rotated fraction
p ∈ {0.6, 0.75, 0.9}; at L=1024/theta=10k roughly the last 35–40% of pairs
never complete a cycle) and a **fully-global 32-parameter learned spectrum**
(LeRoPE replication; tests whether our null was an artifact of per-layer/head
overparameterization). Both are SDPA-OK and one-line-ish. Base sweeps are not
worth GPU time. Both tests should be held to the standing `-0.01` materiality
gate with the explicit expectation that the literature predicts they land
*below* it — the value is closing Direction A with a literature-aligned
negative or catching a cheap surprise.

---

## 3. Direction B — content-conditioning from norm_x

### What the literature says (all condition on cheap reads of the token/residual)

- **CoPE** (Golovneva et al., 2024, arXiv:2405.18719). Positions = cumulative
  sigmoid gate counts per query–key; WT103 124M ppl 23.46 vs 23.81 (relative);
  large wins on counting/copy tasks. Gates use q·k; needs per-pair interpolated
  embedding logits. **Materialized bias — incompatible.**
- **FoX / Forgetting Transformer** (Lin et al., ICLR 2025, arXiv:2503.02130).
  Per-head scalar forget gate **from the residual stream** (our preferred
  input!), cumulative log-decay bias c_i − c_j. Beats RoPE at long context;
  in-distribution roughly parity-to-small-gain without the Pro block; makes
  RoPE largely redundant. Data-dependent gates beat data-independent in their
  ablation. **Custom kernel as published** — but see §7.3 for a fixed-context
  SDPA path.
- **PaTH** (Yang et al., NeurIPS 2025, arXiv:2505.16381). Cumulative
  data-dependent Householder maps; WikiText ppl 18.03 vs RoPE 19.01 at
  760M/50B — the largest verified in-distribution gain in the family.
  **Custom kernel.** Upper-bound reference, not a template.
- **Stick-breaking attention** (Tan et al., ICLR 2025, arXiv:2410.17980).
  Replaces softmax; cumulative recency allocation; 1B: ppl 13.4 vs 13.8.
  **Custom kernel** (~29% overhead).
- **Selective RoPE** (Movahedi et al., ICLR 2026, arXiv:2511.17388). **This is
  nearly our backlog cumulative clock**: per-token input-dependent angles
  (linear + short conv + learned temperature), **cumsum**, then standard RoPE
  machinery as a prelude to an *unmodified* attention kernel. **SDPA-OK.**
  Frames the SSM unification ("real parts forget, imaginary parts encode
  position"); shows rotation-without-decay suffers spectral leakage. LM gains
  on softmax attention are **modest**; big wins are on state-tracking/copy
  tasks. One extracted FoX-composition number looked anomalous
  **[possible extraction error — read the actual table]**.
- **Mamba-3** (ICLR 2026, OpenReview HwCvaJOiCj). Proves the discretized
  selective-SSM update is a data-dependent rotary embedding — the theoretical
  license for "phase = omega · cumsum(f(x))" clocks; note rotation always
  arrives packaged with input-dependent *decay*.
- **TAPE** (Zhu et al., ICML 2025, arXiv:2501.00712): positional state updated
  every layer by content under equivariance constraints; gains mostly
  long-context/reasoning; in-distribution ppl deltas not extractable
  **[unverified]**. Heavier machinery than our design point.
- **DAPE/DAPE-V2** (NeurIPS 2024, arXiv:2405.14722): logit-conditioned bias
  MLP; gains almost entirely at extrapolated lengths. **Materialized bias.**
- **Gated Attention** (Qiu et al., NeurIPS 2025, arXiv:2505.06708). Sigmoid
  gate on the SDPA *output* conditioned on the pre-attention hidden state;
  verified production adoption (Qwen3-Next); consistent gains at 1.7B–15B.
  Not a PE, but captures much of "content-dependent salience" — the
  highest-evidence, lowest-risk content-conditioned intervention that exists.
  **SDPA-OK** (pure post-processing).
- **CARoPE** (arXiv:2507.23083): token-conditioned per-head frequencies from
  embeddings; claims sub-train-length ppl gains at GPT-2 scale — in tension
  with our content-conditioning findings; small-scale preprint.

### Synthesis for us

Two hard lessons. First, **cumulative beats local**: no published gain exists
for bounded per-token positional modulation (our phase-23 shape); every winner
integrates content along the causal interval so that it reparameterizes
*distance*, not individual angles. Our backlog clock (§9.3 of the consolidated
plan) is the right structural shape and is now literature-validated — but
Selective RoPE has substantially built it (input-projected increments, cumsum,
RoPE prelude, SDPA-compatible), so it is no longer a fresh contribution; if
run, it should be framed as a controlled replication at small scale, citing
them. Second, **the decay half outperforms the phase half** across every
framework that separates them. If we open one content-conditioned arm, the
literature says it should be a *decay clock*, not a *phase clock* — see §7.3
for an SDPA-compatible construction specific to fixed context. norm_x as the
conditioning input is affirmed: FoX's gate is exactly a linear read of the
residual stream, and their ablations show query-independence suffices.

---

## 4. Why an additive absolute carrier can beat RoPE (mechanism context)

The carrier's algebra — c(p) added to both q and k yields logit terms q·c_j,
c_i·k, and c_i·c_j on top of q·k — has a classical ancestry and three modern
mechanistic explanations.

**Ancestry.** Transformer-XL's four-term decomposition (content–content,
content–position, u·k global key bias, v·R global distance profile) is the
direct ancestor; it needed materialized relative logits, which is exactly our
FlexAttention 1.9x problem — the carrier gets the analogous terms inside one
fused SDPA call. **TUPE** (ICLR 2021, arXiv:2006.15595) found the
position–position term valuable but argued the *cross*-terms were noisy — under
shared content/position projections, a complaint the carrier's learned per-head
in-geometry profiles sidestep. **DeBERTa** (ICLR 2021, arXiv:2006.03654) found
the opposite: both cross-terms carry signal on every benchmark (their p–p term
is dropped instead). The union of their ablations weakly favors keeping all
three terms, which is what the carrier does. No 2024–2026 paper was found that
simply adds a learned absolute PE into Q/K of a RoPE decoder and reports an
in-distribution gain — the "wpe+RoPE helps nanoGPT" claim is folklore
**[unverified]**.

**Explanation 1 — disentanglement from content norms** (strongest). Round and
Round shows RoPE models must fight to build content-independent positional
attention because rotation multiplies the content vector. **Goat** ("You Need
Better Attention Priors", Litman & Guo, Jan-2026 preprint, arXiv:2601.15380
**[preprint; numbers self-reported]**) makes this precise: attention as
entropic OT with a learnable additive log-prior, implemented *inside stock
SDPA* by appending Fourier cos/sin channels to Q/K plus a key-only sink bias
via a constant query channel; their Theorem 2 says SDPA-expressible bounded
translation-equivariant priors are exactly finite trigonometric polynomials —
i.e., the carrier's function class is canonical, not incidental. 125M/C4:
−1.55 ppl vs ALiBi. Goat is the closest competitor-and-validator: it keeps
position in *dedicated* dims (pure prior, no cross-terms); we share dims with
content (cross-terms included). That difference is testable (§7.4).

**Explanation 2 — sink/default-attention formation.** StreamingLLM
(ICLR 2024), "Why do LLMs attend to the first token?" (COLM 2025,
arXiv:2504.02732 — sinks as over-mixing control), "When Attention Sink
Emerges" (ICLR 2025 — the sink is a key-bias the model is forced to synthesize
from tokens), and gpt-oss's shipped per-head learned sink logit all say models
need static attention biases and normally contort token representations to get
them. A carrier with amplitude peaked at early positions supplies this
natively. Prediction to check on trained models: early-position amplitude
concentration.

**Explanation 3 — entropy/gain control.** SSMax (arXiv:2501.19399) reports
*pretraining* loss gains from s·log n logit scaling; SSA (NeurIPS 2024,
arXiv:2411.12892) learns query/position-dependent temperature with ppl gains
at GPT-2 scale; YaRN's temperature is precedent that a scalar gain folded into
the positional embedding moves loss. Since ||q + c(p)|| varies with p, part of
the carrier's effect may be a soft per-position temperature schedule.

These three decompose into cheap controls (§7.4–7.6) that can attribute our
`0.04`.

---

## 5. Novelty assessment of the carrier contribution

**Verdict from a targeted sweep:** no paper implements the specific package —
per-head/layer additive Q/K carrier `a(p)·cis(omega·p + phi(p))` with amplitude
and phase as smooth learned functions of absolute position from a
zero-initialized hypernetwork over a frozen Fourier basis, exact RoPE null at
init, fused-SDPA-preserving, evaluated on in-distribution loss. No "AddRoPE"
paper or repo exists. But every *ingredient* has precedent, and "learnable
generalizations of RoPE" is now crowded:

- learned per-head frequency/phase/gain carriers in Q/K space: **sineSPE**
  (ICML 2021 — must-cite precedent);
- amplitude-envelope × cis: **xPos** (2022, fixed envelope), **MoPE**
  (2026 preprint, learned but input-additive, tiny scale);
- MLP over a Fourier basis of position: **Learnable Fourier Features**
  (NeurIPS 2021), **FIRE** (ICLR 2024, but logit-bias output);
- zero-init RoPE-anchored learned deltas: **LeRoPE** (2026);
- additive position terms in attention: **Transformer-XL, TUPE, DeBERTa**;
- appended-Fourier-channel SDPA priors: **Goat** (2026).

**Frame the contribution narrowly:** (i) the additive **application site** that
preserves fused SDPA where competitors modify the rotation, condition on
content, or need custom kernels; (ii) the **position-only** result — content
conditioning unnecessary — against CARoPE/TAPA/Selective RoPE which all bet on
content; (iii) the **amplitude-not-phase** attribution with depth-varying decay
profiles.

**Two vulnerabilities:** LeRoPE's positive learned-frequency result must be
reconciled with our null (tying granularity, optimizer hygiene, scale — §7.2
resolves it empirically); **Bifocal Attention** (arXiv:2601.22402
**[abstract only]**) uses "learnable frequency, amplitude, and phase" language
and must be read closely and differentiated explicitly.

---

## 6. Must-cite list (reviewer-proofing)

Core: LeRoPE (2607.10134) · PaTH (2505.16381) · Round and Round (2410.06205) ·
FoPE (2412.17739) · TAPA (2509.12635) · CARoPE (2507.23083) · sineSPE
(ICML 2021) · xPos (2212.10554) · Learnable Fourier Features (2106.02795) ·
Transformer-XL (1901.02860) · TUPE (2006.15595) · DeBERTa (2006.03654) · FIRE
(2310.04418) · Rope-to-Nope (2501.18795) · Goat (2601.15380) · Bifocal
(2601.22402) · CoPE (2405.18719) · FoX (2503.02130) · Selective RoPE
(2511.17388) · NoPE papers (Kazemnejad 2305.19466; Haviv 2203.16634) ·
Deconstructing Positional Information (2505.13027).

Methodology section: Signal and Noise (Ai2, 2508.13144 — benchmark
signal-to-noise; supports eval-loss-only design) · "Small-Scale Experiments:
Are We There Yet?" (2608.11859) · Wortsman small-scale proxies (2309.14322) ·
Show Your Work (Dodge 2019) · Bouthillier variance (MLSys 2021). Our
5k-unreliability finding (three documented reversals, same-seed floor 0.0015,
cross-seed 0.002–0.027) has no direct precedent in the PE literature — worth
elevating as a contribution.

Surveys: Zhao et al. length-extrapolation survey (2312.17044) is still the
canonical PE survey; **no dedicated 2025–2026 LLM-PE survey exists** — a gap.

---

## 7. Experiment candidates this review motivates

Ordered by information-per-GPU-hour under our standing gates (three paired
seeds, `-0.01` materiality, no factorial crossing). 7.4–7.6 are attribution
controls for the carrier and only matter if phase-19 confirms.

1. **Partial RoPE screen** (Direction A). Rotate top-p fraction of pairs,
   p ∈ {0.6, 0.75, 0.9}, identity elsewhere. One-line change, SDPA-OK.
   Literature-predicted outcome: below gate; closes Direction A cleanly either
   way. Also the natural "RoPE prior + freed content dims" companion to the
   carrier story (Barbero's decomposition made static).
2. **Globally-tied learned spectrum** (Direction A / null-boundary). One
   32-param spectrum shared across all layers and heads, log-space, zero-init,
   no WD, separate clipping — LeRoPE's exact recipe on our harness. Directly
   reconciles their positive with our phase-20/22 null for the writeup.
3. **SDPA-compatible decay clock** (Direction B — the literature's pick over
   phase clocks). FoX-style per-head gate f_t = sigmoid(Linear(norm_x_t)),
   c_t = cumsum(log f); the bias c_i − c_j is row-shift-invariant in softmax,
   so it reduces to a per-key term −c_j implementable by **appending one head
   dimension** (q̃=[q, s], k̃=[k, −c_j/s]). Needs a bf16/fp32 drift check of the
   cumsum at L=1024 and care with QK-norm interaction; zero-init ⇒ exact
   baseline. To our knowledge this fixed-context SDPA folding is not published
   (FoX ships a custom kernel for unbounded contexts) — both a real experiment
   and a small novel engineering claim. The phase-clock variant (≈ Selective
   RoPE) is demoted: published, and modest on softmax LM loss.
4. **Goat-style dedicated-dims control** (carrier attribution). Same
   positional function class, but carried in r appended Q/K channels invisible
   to content (pure prior, includes a constant-query sink channel) vs our
   shared-channel carrier (cross-terms included). Plus an eval-time four-term
   logit decomposition (zero q·c, c·k, c·c separately on analysis batches).
   Together these answer *which term buys the 0.04* — the single highest-value
   mechanistic add to the writeup.
5. **Per-head sink logit** (carrier attribution, ~8 params/layer). gpt-oss
   style virtual-key scalar. Isolates the sink share of the carrier's gain.
6. **Per-position query gain** (carrier attribution, ~2 params/head).
   g(p)=1+gamma·log(1+p) scaling q, per head (SSMax/SSA-shaped). Isolates the
   temperature share.
7. **Direct per-position table** (carrier capacity ceiling; from the internal
   review, reinforced by TUPE's untied-P result). Learned [L, d_head] table per
   head/layer added to Q/K, zero-init. If it matches the hypernetwork, the
   smoothness prior is decorative; if it loses, the Fourier-basis prior is
   load-bearing. Mirrors our own phase-20 "direct table first" doctrine.

Explicitly *not* motivated: base-theta sweeps (bounded upside, silent retrieval
risk), CoPE/DAPE/stick-breaking/PaTH ports (kernel-incompatible), local bounded
per-token modulation (uniformly unrewarded in the literature), any factorial
sweep.

---

## 8. Open verification tasks

Read directly before citing in prose: Bifocal (2601.22402), Frequency Bands in
RoPE (OpenReview PR1PPxvG9Q), Selective RoPE's FoX-composition table, AdaRoPE's
frequency-vs-temperature ablation, TAPE's in-distribution C4 numbers, sineSPE's
exact learnable-parameter set, Goat end-to-end (single most important close
read — theorem + implementation overlap with our method). Several early-2026
arXiv IDs were seen at snippet level only and are flagged inline above.
