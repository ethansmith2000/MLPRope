# MLPRope consolidated research plan

> Historical plan. Phase 19 and phase 24 are now complete and the implementation
> has changed. Use [`CURRENT_STATUS.md`](CURRENT_STATUS.md) for current evidence
> and priorities. Locked protocol files still govern completed experiments.

_Consolidated 2026-08-05 from the original methods discussion, phases 19-23,
and the independent implementation/evidence/methods review. This is the current
decision roadmap; narrower protocol files remain authoritative for experiments
that were already locked._

## 1. Research objective

The primary question is now:

> Does the learned position-only additive Q/K carrier provide a reproducible,
> practically material improvement over standard RoPE at context 1024 and 30k
> steps, and does that improvement survive a credible compute adjustment?

The secondary question is mechanistic:

> Is the useful degree of freedom principally positional gain/amplitude, or is
> there any material value in learned rotary spectra or content-dependent phase?

Length extrapolation is not a current target. Context 2048/4096 cells should not
be added by default. The project should prefer resolving existing claims over
opening more architectural axes.

## 2. Current evidence and status

| Workstream | Best current evidence | Status |
| --- | --- | --- |
| Position-only additive carrier | Phase-18 seed-123 development gap `-0.0421` vs RoPE; same-window phase-19 rerun agrees within `0.00325` | strong screen, not confirmed |
| Content conditioning increment | `~0.004-0.005` in unpaired single-seed phase-17/18 contrasts | unresolved at that precision |
| Compute-adjusted position-only result | reported `+0.0289` iso-wallclock margin, based on a single throughput observation | plausible but provisional |
| Static learned RoPE frequencies | phase-20 `~-0.0019`; phase-22 30k additive `-0.0016` with mixed signs and severe winding | closed for the tested direct/static family |
| Local token-conditioned phase | phase-23 best `-0.0021`, all signs favorable but below gate | bounded formulation closed as material result |
| Free horizon-normalized phase | not trained; tanh was mostly near-linear in the best phase-23 arms | coherent control, low priority |
| Cumulative content clock | not implemented or trained | distinct conditional hypothesis, not an earned next run |

No correctness bug has been found that invalidates phases 20-23. The largest
current weakness is evidentiary: phase 19 was interrupted before any paired
primary contrast completed.

## 3. Evidence rules

These rules apply unless a protocol explicitly locks something stricter.

- Primary endpoint: held-out loss at configured context 1024. Each row produces
  1023 next-token targets; external writing should footnote this rather than
  silently relabeling old experiments.
- Pair data order and shared model parameters through seeds `123/456/789` and
  `paired_initialization_seed`.
- Use a disjoint final holdout. For phase 19 this is 1,024 batch-size-one
  examples beginning at validation batch 2,048.
- Treat paired-example confidence intervals as evaluation-sampling uncertainty,
  not training-seed uncertainty.
- A 5k result is a failure/materiality screen. It is never a durable positive
  claim; phase 21 demonstrated that a `-0.012` 5k result can collapse at 30k.
- Promotion requires mean improvement of at least `0.01`, favorable signs in
  all three seeds, sane diagnostics, and acceptable measured cost.
- Do not infer mechanism from a small ordering when capacity, optimization, or
  implicit regularization remains equally plausible.
- Use `gpu-claim` for every GPU process. Long suites run under supervisor. The
  worker cap is agreed for each batch; there is no permanent two-GPU project
  limit.
- Do not change the optimizer or parameter grouping inside an already partially
  completed locked protocol.

## 4. Decision flow

```text
phase-19 position-only vs RoPE gate
├── fails (<0.01 mean or mixed signs)
│   └── downgrade headline, stop method expansion, benchmark throughput, write up
└── passes
    ├── finish mapped-AddRoPE, matched-FFN, and content controls
    ├── run controlled throughput benchmark
    └── state only the contrasts those controls resolve

phase-23 checkpoint ablations (after or alongside credibility work)
├── no material native-vs-zero or token-alignment effect
│   └── close local dynamic phase branch
└── material token-alignment effect
    └── consider one structurally distinct cumulative-clock screen
```

Free-vs-bounded local output is not on the critical path. It remains available
as a one-axis control if a later result specifically makes saturation causal.

## 5. Work package A — finish the h1024 evidence

### A1. Minimum headline tranche

Reuse the completed phase-19 `position-only/seed123` run and run these five
locked configs from scratch:

1. `standard-rope/seed123`
2. `standard-rope/seed456`
3. `standard-rope/seed789`
4. `position-only/seed456`
5. `position-only/seed789`

Before launch, move incomplete output directories to a dated archival location
with a manifest. Do not append fresh metrics to interrupted directories. The
interrupted runs are not resumable because `checkpointing_steps` was `null`.
Keep the locked phase-19 recipe unchanged so the completed seed-123 candidate
remains reusable.

Primary decision:

- confirm the headline if position-only beats RoPE in all three paired seeds
  and mean candidate-minus-RoPE loss is at most `-0.01` on the final holdout;
- otherwise downgrade it to a screening result and stop expanding the method.

Estimated new training: five h1024 30k runs, roughly 4-5 GPU-hours based on the
surviving logs, plus queue time.

### A2. Conditional controls

If A1 passes, finish the remaining locked phase-19 arms across all three seeds:

- mapped AddRoPE amplitude-0.3;
- content+position carrier;
- RoPE with matched FFN capacity.

This is nine additional fresh runs. A single matched-FFN seed123 can be run in
the A1 batch as an early capacity check, but it does not replace the three-seed
locked contrast.

Interpretation gates:

- Position-only vs mapped AddRoPE asks whether the hypernetwork/profile is worth
  more than the simpler additive carrier.
- Position-only vs matched FFN asks whether the gain is specific to positional
  structure rather than comparable useful capacity.
- Content+position vs position-only estimates the content increment. A result
  inside `0.01` is unresolved scientifically; the cheaper position-only method
  may still be preferred as an engineering default.

Whatever the outcome, phase 19 ends the headline-confirmation question. Do not
respond to an unfavorable result by tuning the protocol.

## 6. Work package B — controlled throughput

Benchmark these h1024/d12 phase-18 shapes:

1. standard RoPE;
2. position-only;
3. content+position/free control;
4. wide-trunk 256.

Lock the benchmark before measuring:

- same physical GPU and software state;
- exclusive `gpu-claim` occupancy;
- same batch, context, dtype, attention implementation, compile setting, and
  gradient accumulation as training;
- no evaluation, W&B, per-step profiling, or checkpoint I/O in the timed region;
- discard compile and allocator warmup;
- at least three interleaved repetitions per arm;
- report median and dispersion for target tokens/s and peak allocated memory.

Recompute iso-wallclock conclusions with throughput uncertainty or a sensitivity
range. Fixed-token loss remains the primary scientific endpoint; iso-wallclock
is a hardware/software-specific secondary view.

Decision:

- retain the position-only compute claim only if its margin remains positive
  across the measured throughput range;
- withdraw exact negative rankings among close heavy arms if throughput noise
  can reverse them.

## 7. Work package C — phase-23 checkpoint diagnostics

Use the saved rank-32 SiLU checkpoints for seeds `123/456/789`. No retraining is
needed. Evaluate the same disjoint holdout under deterministic forward modes:

1. **native:** trained controller unchanged; reuse saved losses where possible;
2. **zero:** set every dynamic phase output to zero;
3. **sequence mean:** replace each controller's tokenwise raw output by its mean
   over tokens within that example, preserving an example-level schedule while
   removing same-token alignment;
4. **token shuffle:** apply a deterministic within-sequence permutation to raw
   controller outputs while leaving the language-model activations in place;
5. **global mean, optional:** inject a precomputed layer/head/pair mean to
   approximate a genuinely static learned schedule.

Report paired per-example loss deltas against native. Run all layers ablated
together first; add layerwise ablations only if the aggregate effect is material.

Interpret carefully:

- Native vs zero measures endpoint reliance on the controller after coadaptation;
  it does not recover the counterfactual training trajectory.
- Native vs sequence-mean/shuffle tests token alignment, not whether the dynamic
  model beats a separately trained static model.
- A material ablation cost can coexist with the trained model's merely `-0.002`
  advantage over fixed RoPE: the model may rely on a mechanism that did not
  improve the final system materially.

Use `0.005` as the mechanism-diagnostic threshold. If neither zeroing nor
breaking token alignment changes loss by that much, close the local dynamic
branch. If token alignment matters by more than `0.005`, record genuine use but
still require a new trained mechanism to clear the normal `0.01` promotion gate.

Expected cost: brief GPU inference through `gpu-claim`, not zero compute but far
less than a training run.

## 8. Work package D — optimizer/width deconfound

The earlier suggestion that one h768 run can isolate the phase17/18 optimizer
confound is insufficient: a gap requires both endpoints under the same recipe.

Run paired seed-123 h768/d8 cells for:

- standard RoPE;
- position-only;

Use the phase-18 optimizer recipe (`4e-4`, betas `0.95/0.999`) and both the old
development window and a disjoint final holdout. Compare the resulting gap with
the phase-19 h1024 paired gap. This is still a one-seed scale diagnostic, so it
can qualify the width claim but cannot establish a scaling law.

Do this after A1 and B unless scale is central to the intended write-up.

## 9. Conditional methods backlog

### 9.1 Static spectrum functions — closed

The original candidates were affine functions, splines/fixed bases, and small
coordinate MLPs over layer/head/log-frequency. Direct per-pair learning was the
capacity ceiling and failed at 30k. A smaller structured function is not a
priority without a new inductive-bias prediction that explains why it should
beat the direct table.

### 9.2 Free local horizon phase — deferred control

The clean map is

```text
delta_phase[b,h,t,i] = (t / 1024) * raw(norm_x[b,t])
```

It has an exact zero anchor and no arbitrary one-radian boundary. It permits
phase winding and grows beyond the reference horizon. Existing low-rank tanh
outputs were mostly in the near-linear region, so this is unlikely to turn the
phase-23 screen into a material result.

If run, change only the output map of the rank-32 SiLU arm, keep all three seeds
and the existing holdout, and retain the `-0.01` gate. Rational squash and
clamp-STE are optimizer controls, not answers to whether a boundary is justified.

### 9.3 Causal cumulative content clock — conditional new hypothesis

A structurally cleaner content-dependent rotary mechanism is a positive causal
clock. The minimal per-head version is:

```text
raw[b,t,h] = Linear(D, H)(norm_x[b,t])
step[b,t,h] = exp(raw[b,t,h])
tau[b,0,h] = 0
tau[b,t,h] = sum_{s=1..t} step[b,s,h]
phase[b,h,t,i] = omega_base[i] * tau[b,t,h]
```

Zero-initialize the final projection, giving `step=1` and exact standard RoPE.
Positive increments make `tau` monotone. Q and K share the same clock. A
per-head scalar output is the correct first dimensionality: `Linear(D,H)` is
cheaper and more interpretable than a `D -> r -> D/2` pairwise controller and
preserves the geometric spectrum within each head.

This makes phase differences depend on content increments along the causal
interval rather than explicitly multiplying a local prediction by absolute
position. It is still a new hypothesis and inherits content-dependent state and
identifiability questions.

Only open it if:

1. phase-19 credibility work is settled;
2. phase-23 ablations show a material same-token/content-aligned effect; and
3. the project owner prefers another mechanism test over writing up.

First screen: fixed RoPE vs the per-head clock, three paired seeds at 5k. A 30k
confirmation is earned only by the usual `-0.01`, all-signs gate. Per-pair
clocks, head sharing, low-rank mappers, alternative inputs, and Q/K separation
are later axes, never a Cartesian sweep.

### 9.4 Original architecture axes — sequential only after a winner

The original discussion remains useful as a conditional ordering:

1. Q/K-shared before Q/K-separate;
2. normalized residual before pre-norm magnitude, Q/K RMS, or raw projected Q/K;
3. simplest full linear map appropriate to output size before low-rank linear
   and low-rank SiLU capacity/regularization comparisons;
4. per-head before head-shared when head specialization is the hypothesis;
5. rank and per-pair dimensionality only after a mechanism wins;
6. phase parameterization and mapper form are separate axes and must not be
   crossed factorially.

## 10. Engineering cleanup and process

- Preserve the current optimizer behavior through phase 19. A reviewer found
  that LayerNorm weights inside `nn.Sequential` are not recognized by the
  name-based no-decay matcher. It affects all compared arms equally, but fixing
  it mid-protocol would change the locked recipe. Replace name matching with a
  module/type-aware grouping only after phase 19.
- Future long runs should save resumable checkpoints. Do not retrofit checkpoint
  overhead into phase 19 while reusing its completed seed-123 run.
- Add an explicit worker limit to suite orchestration and run the parent under
  supervisor. Set the limit from the current shared-GPU agreement at launch.
- Archive interrupted runs rather than deleting or mixing them with reruns.
- Record every launched, interrupted, skipped, and completed protocol in the
  journal. Empty queue logs are "never started," not "failed training."
- Keep result summaries subordinate to raw configs, per-example evaluations,
  checkpoints, and machine-readable analysis.

## 11. Write-up gate

After A1 and B, there should be enough evidence to write regardless of outcome.

The write-up should separate:

- fixed-token loss from compute-adjusted performance;
- screening results from paired confirmation;
- evaluation-sampling uncertainty from seed/training uncertainty;
- additive gain/amplitude mechanisms from rotary angle/frequency mechanisms;
- verified implementation facts from mechanistic interpretation;
- current context-1024 conclusions from untested length extrapolation.

If A1 confirms, complete A2 when the stronger mechanistic claims are important.
If A1 fails, do not search for a replacement positive result; document the
screening-to-confirmation reversal and the robust negative/structural findings.
