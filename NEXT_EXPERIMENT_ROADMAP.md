# MLPRope next-experiment roadmap

_Proposed 2026-08-23. This roadmap assumes intermittent access to 2–4 RTX
5090s through `gpu-claim`. It does not authorize bypassing an existing claim._

## Objective and ordering

The next work should answer three questions in this order:

1. Does phase 24's large amplitude-1.0 improvement survive 30k training?
2. How much of the additive carrier gain is query-temperature control, key
   leverage, or genuinely additive carrier geometry?
3. Do either of the two new attention-local mechanisms—the tied Q/K
   preprojection sinusoid or the causal rotary clock—earn promotion?

The ordering is evidence-weighted. Phase 24 already produced a large,
three-seed effect and therefore comes before new architectural exploration.
Gain/salience is the highest-value attribution experiment. The rotary clock has
a weak prior and should remain a small mechanistic screen.

## Global protocol

- Model for screens and the first promotion: h768/d8, 8 heads, training context
  1024, the phase-24 optimizer/data recipe, fused SDPA.
- Seeds: `123/456/789`, with `paired_initialization_seed=seed` and identical
  data order across arms.
- Primary endpoint: context-1024 loss. A 5k screen uses the disjoint 256-example
  holdout starting at validation batch 2048. A 30k promotion uses 1,024
  examples from the same start.
- Negative candidate-minus-reference loss favors the candidate. Promotion
  requires mean improvement of at least `0.01`, favorable signs in all three
  seeds, finite/sane mechanism diagnostics, and acceptable measured cost.
- A 5k result selects a mechanism; it is not a durable positive claim.
- Keep per-example losses, final weights, resolved configs, logs, diagnostics,
  and an analysis JSON. Intermediate checkpoints may be pruned only after the
  final artifacts are verified and copied to durable storage.
- Use checkpoints at 10k and 20k for 30k runs, then remove completed
  intermediates. The current workspace has ample capacity, but the phase-19
  disk failure should not be repeated.
- Interleave arms across seeds/GPUs rather than running one entire arm at one
  time. Treat the phase-24 51k-token/s outlier as contention, not mechanism
  cost; report medians and a dedicated steady-state throughput measurement.

## Stage 0 — engineering and reproducibility gate

Before a full launch:

1. Run `scripts/position_dynamics_cuda_smoke.py` through `gpu-claim`. It checks
   bf16, `torch.compile`, exact clock anchoring, and backward gradients.
2. Freeze the code/config state used by the sweep. Record the git commit and
   whether the tree is dirty in a run manifest.
3. Dry-run every generated config and assert parameter counts and run-name
   uniqueness.
4. Preflight one 20–50-step job per new mechanism, including final diagnostic
   collection.

No full training sweep starts if the exact-anchor, prefix-causality,
full-versus-incremental, or compiled-backward gates fail.

## Stage 1 / Phase 25 — promote the amplitude anchor

### Question

Was phase 24's anchor-1.0 gain a transient early-training effect, or should it
replace anchor 0.3 as the mapped-carrier default?

### Arms

| Arm | Purpose |
| --- | --- |
| fixed RoPE | fresh paired reference and per-example baseline |
| compact basis, anchor 0.3 | phase-19 continuity control |
| compact basis, anchor 1.0 | phase-24 winner |

Run all three seeds to 30k: 9 jobs. Evaluate only the primary 1024 context in
this tranche; phase 24 already showed that the 4096 behavior is a different
question. At phase-24 throughput, this is roughly 3.5–4 aggregate GPU-hours,
plus final evaluation.

### Decision

- If anchor 1.0 beats anchor 0.3 by at least `0.01` in every seed, it becomes
  the mapped-carrier reference.
- If it retains a large advantage over RoPE but not over 0.3, the carrier result
  remains but the initialization-scale claim is unresolved.
- If the 1.0 advantage collapses or changes sign, record phase 24 as an
  early-training optimization effect and keep 0.3 as the confirmed default.

An h1024/d12 confirmation is considered only after this gate. Do not jump
directly from the 5k h768 result to a large-model claim.

## Stage 2 — implement the position-only gain decomposition

The existing `conditioning.kind=adaptive_gain` is content-conditioned. Reusing
it would answer the wrong question. Add a dedicated position-only multiplicative
gain with the same compact position input used by the carrier:

```text
u_h(t)   = mapper_h([Fourier16(t), normalized_t, log_t])
g_q,h(t) = exp(readout_q,h(u_h(t)))
g_k,h(t) = exp(readout_k,h(u_h(t)))
```

The final scalar readouts are zero-initialized, giving `g_q=g_k=1` and exact
standard RoPE at initialization. Gains are applied after per-head Q/K RMSNorm;
because they are scalars, applying them before or after rotation is equivalent.

Required targets are `q`, `k`, and `both`. The `both` arm uses a shared
position trunk with separate zero-initialized Q/K readouts. The inactive branch
is exactly one, with no unused parameters.

Required tests:

- exact fixed-RoPE logits at initialization, including bf16;
- immediate nonzero gradient in each active readout;
- Q-only gain scales every unmasked logit in a query row by the same scalar;
- K-only gain changes only logits involving that key position;
- target isolation, parameter counts, serialization, compile, and CUDA smoke.

## Stage 3 / Phase 26 — gain, salience, and carrier geometry

### Arms

| Arm | Mechanistic role |
| --- | --- |
| fixed RoPE | zero-gain reference |
| compact mapped carrier | phase-25 winner: anchor 1.0 if promoted, otherwise 0.3 |
| position gain on Q only | row sharpness / temperature-like control |
| position gain on K only | position-dependent key leverage |
| position gain on Q and K | interaction and combined allocation control |

Run 5 arms × 3 seeds × 5k = 15 jobs. Rerun both controls so per-example paired
comparisons are available; phase-24 outputs no longer contain those arrays.

### Interpretation

For Q-only gain,

```text
attention_t = softmax(g_q(t) * logits_t)
```

so the intervention is genuinely temperature-like at each query row. K-only
gain is not literally an additive salience bias: it scales a signed dot product,
so it can strengthen either compatibility or incompatibility. Call it key
leverage unless incoming attention mass confirms a salience interpretation.

Report each gain arm versus RoPE and versus the carrier. Also report the
fraction of the carrier improvement captured, but do not use that noisy ratio
as a decision statistic.

### Attention diagnostics

Recompute attention only in a no-grad diagnostic pass on fixed held-out
sequences; do not replace fused SDPA during training. For selected layers and
heads record:

- query gain and key gain versus absolute position;
- per-query logit standard deviation;
- raw entropy `H_t = -sum_j a_tj log(a_tj)`;
- support-normalized entropy `H_t/log(t+1)`—raw entropy has a mechanical
  position correlation because later rows have more visible keys;
- mean incoming attention mass per key position, normalized by the number of
  queries that can see that key;
- per-position next-token loss;
- Pearson and Spearman correlations among gain, normalized entropy, incoming
  mass, and position, plus cross-seed profile correlations.

### Decision

- Q-only near the carrier (carrier-minus-Q gap under `0.01`) supports an
  entropy-allocation explanation.
- K-only near the carrier supports key-leverage/salience as the main effect.
- Both materially beating either single branch indicates an interaction.
- All gain arms materially behind the carrier favors the additive
  content–carrier cross terms and carrier Gram term rather than scalar gain.

This phase is valuable as attribution even if no gain arm deserves 30k
promotion.

## Stage 4 / Phase 27 — isolated new-mechanism screen

Do not combine these mechanisms with the additive carrier in this stage.

### Arms

| Arm | Comparison it supports |
| --- | --- |
| fixed RoPE | reference for hybrid preprojection and clocks |
| no explicit PE | reference for preprojection-only |
| Q/K preprojection sinusoid, no RoPE | tied additive carrier as sole PE |
| Q/K preprojection sinusoid + fixed RoPE | complementarity of additive read and rotation |
| pointwise per-head rotary clock | minimal cumulative content clock |
| four-token causal-convolution rotary clock | value of short causal state |

Run 6 arms × 3 seeds × 5k = 18 jobs. The compact carrier result from phase 26
is an external reference, not a one-axis contrast with these arms.

For clock arms, retain per-layer speed min/max/mean, final clock-drift RMS,
phase-drift RMS/max by frequency band, and the fraction of speeds near their
bounds. Monotonicity alone is not a health certificate: persistent small speed
errors accumulate.

### Decision

- Preprojection-only must beat NoPE materially and is compared descriptively
  with RoPE and the compact carrier. The hybrid arm must beat fixed RoPE by the
  standing gate to earn promotion.
- A clock must beat fixed RoPE by `0.01` in all seeds. A small favorable result
  below that line is a mechanistic null, consistent with earlier phase results.
- Implement a learned EMA/associative-scan controller only if the causal
  convolution beats the pointwise clock by at least `0.01` with healthy drift.
  Otherwise temporal-state expansion stops here.
- Do not add frequency-wise clocks, Q/K-separate clocks, ranks, bounds, or
  additive-carrier combinations before a simple clock passes.

## Stage 5 — promotion and cost

At most one winner from each stage is promoted. A promoted arm receives a fresh
three-seed 30k comparison against its direct control. Only a result surviving
that gate is considered for h1024/d12.

For every survivor, run dedicated steady-state throughput measurements after
compile warmup. Report tokens/s, peak memory, parameter count, and loss versus
both step count and wall-clock. The old iso-wallclock estimates are not reused.

## Immediate next action

The first launchable work item is Stage 0's claimed-GPU smoke, followed by
generation and dry-run validation of the nine phase-25 configs. Phase 25 can
train while the position-gain implementation and diagnostic evaluator are
built. No clock or preprojection sweep needs to occupy GPUs before the stronger
amplitude result is promoted.

The locked generator, analyzer, and shared-claim launcher are:

- `prepare_rope_embed_basis_30k.py`
- `analyze_rope_embed_basis_30k.py`
- `launch_rope_embed_basis_30k.sh`
