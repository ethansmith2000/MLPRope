# MLPRope next-experiment roadmap

_Revised 2026-09-05 after repository consolidation._

## Objective

Stop modifying the local carrier shape. Strengthen or falsify the broader
claim that a tied scalar pre-Q/K sinusoid improves a standard-RoPE decoder.

The completed evidence is unusually consistent but incomplete: three paired
seeds agree at 30k, while only seed 123 has run to 200k. There is no mature
normalization or scale transfer test. These are the largest remaining holes.

## Phase 38 — evidence-strengthening matrix

Every cell compares the unchanged candidate against its matched fixed-RoPE
baseline. Within a pair, data seed and paired parameter initialization are
identical.

| Pair | Model | Q/K normalization | Seed | Steps | Purpose |
| --- | --- | --- | ---: | ---: | --- |
| A | h1024/d12, 8 heads | method-aware RMS | 123 | 200k | scale-up transfer |
| B | h768/d8, 8 heads | method-aware RMS | 456 | 200k | mature replication |
| C | h768/d8, 8 heads | method-aware RMS | 789 | 200k | mature replication |
| D | h768/d8, 8 heads | disabled | 123 | 200k | QKNorm robustness |

There are eight runs total. Pair A should be queued first because it is the
longest. The matrix intentionally does not add amplitude, phase, frequency,
EMA, mapper, or Q/K-coupling variants.

## Frozen common protocol

- OpenWebText GPT-2 cache at context 1024;
- microbatch 8, no gradient accumulation;
- AdamW, LR `3e-4`, betas `(0.9, 0.98)`, weight decay `0.01`;
- linear schedule with 200 warmup steps and 200k total steps;
- standard fixed RoPE in both arms;
- scalar carrier gate initialized to 1.0 and learned per layer;
- fused SDPA, bf16, compile default;
- development evaluation: 128 examples every 10k steps;
- final evaluation: 1,024 examples beginning at validation batch 2,048,
  disjoint from development evaluation;
- per-example losses, position diagnostics, optimizer traces, provenance, and
  final weights saved;
- one rolling checkpoint per active run, removed only after completion audit.

## Predeclared readout

For each pair, report candidate-minus-baseline final loss with a paired
per-example bootstrap confidence interval and the complete 10k--200k curve.
Also report throughput, memory, parameter count, late-window slope, learned
gates, carrier/content energy ratios, and optimizer/function-step health.

Interpretation gates:

- **mature replication succeeds** if both new h768 seeds favor the carrier and
  the mean across seeds 123/456/789 is at most `-0.03` nats;
- **scale transfer succeeds** if the h1024 candidate is at least `-0.03` and
  its paired 95% interval is below zero;
- **QKNorm robustness succeeds** if the no-QKNorm candidate is at least
  `-0.02` and its paired interval is below zero.

These thresholds are deliberately far larger than the `~0.001` noise scale
that closed carrier refinements. A miss does not invite tuning; it narrows the
claim to the architecture where the method works.

## Phase 39 — mechanism and generalization

Run only after Phase 38 artifacts are checked. Split this phase into one small
architectural comparison and analyses of the trained checkpoints.

### Phase 39A — carrier location and AddRoPE comparison

The phrase "post-Q/K sinusoid" is ambiguous. If the post-projection carrier is
`W_q z/W_k z`, it is algebraically identical to pre-Q/K injection. A distinct
post-projection arm must use a native head-space carrier. The immediate screen
should answer two questions without reopening amplitude/frequency search:

1. is repeated attention-local access responsible for the gain, rather than
   merely supplying the network with another sinusoidal input; and
2. how does the promoted method compare with a native AddRoPE carrier, both as
   a replacement for RoPE and in composition with it?

Use h768/d8, seed 123, 30k steps, paired initialization and data order, and the
same disjoint 1,024-example holdout. The proposed cells are:

| Cell | Operation (normalization suppressed) | Scientific role |
| --- | --- | --- |
| fixed RoPE | `R_p W_q x_p` | common control |
| input sinusoid + RoPE | add `beta z(p)` once at the residual-stream entrance | classic-input/location control |
| pre-Q/K sinusoid + RoPE | `R_p W_q(x_p + alpha z(p))` | promoted method |
| fixed AddRoPE + RoPE | `R_p(W_q x_p + e(p))` | parameter-free native head-space placement control |
| direct AddRoPE | `W_q x_p + e_q(p)` | direct static amplitude/phase, replacing RoPE |
| direct AddRoPE + RoPE | `R_p(W_q x_p + e_q(p))` | tests whether the two geometries compose |

There is one separate, high-value factorial hole. Phase 33 already trained
fixed RoPE, pre-Q/K+NoPE, and pre-Q/K+RoPE for 200k under one protocol, but not
the plain NoPE cell. A matched plain-NoPE run from the exact Phase-33 source
would complete the mature `RoPE x pre-Q/K carrier` 2x2 and permit a formal
interaction estimate. Prefer that single missing-cell run to retraining the
other three, provided source, initialization, and dataset provenance match
exactly. It is not required for ranking methods, but it is required for a
mature claim about additivity or synergy.

The input arm should restore only a minimal one-shot residual carrier after
`in_proj`, with a learned scalar initialized to 1.0; do not restore the former
generic residual/per-layer machinery. The fixed post-projection control should
use the unit native head-space carrier without learned parameters. The two
direct AddRoPE cells should use the same signed, nonsaturating, position-only
per-head/per-frequency Q/K amplitude and phase parameters, with the standard
frequency bank fixed. They intentionally omit both the historical position
mapper and all content conditioning.

Raw amplitudes are not comparable across sites. Before launch, report the
initial carrier/content RMS ratio and positional energy fraction at the actual
mixture point. Use gate 1.0 where it reproduces the promoted method's roughly
one-third initial positional energy; otherwise choose a predeclared
RMS-matched scale. Keep the amplitude anchor and optimizer settings identical
across the paired direct-AddRoPE cells.

The paired-initialization audit supports the unit anchor. On the first fixed
validation example, the one-shot input carrier has carrier/content RMS ratio
`0.7116` and positional energy fraction `0.3362`; layer-0 pre-Q/K ratios are
`0.7071` before projection and `0.6994/0.7060` after the Q/K projections. The
unit native AddRoPE carrier gives `0.7065/0.7024` against raw projected Q/K.
Thus all three sites begin near the same `0.70` RMS ratio without a calibration
fit, while their placement and parameterization remain genuinely different.

This is a breadth screen, not publication evidence. Promote only cells that
beat their direct control materially and remain competitive in throughput. In
particular, do not spend mature runs on both direct AddRoPE orderings if the
30k screen clearly resolves them.

An addend applied *after* standard RoPE,
`R_p W_q x_p + e_q(p)`, is a useful secondary ordering ablation only if
AddRoPE+RoPE survives. It requires new runtime placement machinery and should
not block the primary screen.

### Phase 39B — mechanism analyses

This stage should primarily evaluate trained models rather than retrain
variants:

1. **position-length profile:** loss by token position and contiguous context
   length, separating startup tokens from late-context behavior;
2. **attention entropy:** entropy normalized by the number of visible keys,
   reported per head/layer and by query-position bucket;
3. **distance profile:** expected attended distance and mass in logarithmic
   distance bins;
4. **position correlation:** correlation or mutual-information proxy between
   attention patterns and absolute/relative position, conditioned on layer;
5. **inference ablation:** set learned carrier gates to zero in a trained
   candidate, clearly labeled as a distribution-shift diagnostic rather than
   a matched training control.

Add one carrier-logit attribution analysis. Ignoring Q/K normalization, the
pre-Q/K attention logit contains content-content, content-position,
position-content, and position-position terms. With method-aware RMSNorm this
decomposition is not exactly linear, so report both pre-normalization term
magnitudes and counterfactual full-attention changes. This can test whether the
gain is associated with a pure positional prior, cross terms, attention sinks,
or position-dependent entropy.

The mechanism analysis should use identical examples and bootstrap at the
example/document level. Token-level samples within a document are not
independent and must not be treated as such.

## Paper-strength evidence, conditional on Phases 38--39

The paper claim should determine the breadth. For the narrow claim—an
attention-local sinusoidal carrier improves a RoPE decoder—the minimum
evidence package is:

1. **mature replication:** three paired seeds at the main scale and horizon;
2. **scale transfer:** at least one larger model under the frozen method;
3. **architecture robustness:** QKNorm on/off, with normalization order stated;
4. **core factorial:** NoPE, RoPE, pre-Q/K+NoPE, and pre-Q/K+RoPE at one mature
   matched horizon if making a complementarity or interaction claim;
5. **method comparison:** RoPE, residual-input sinusoid+RoPE, standalone
   AddRoPE, AddRoPE+RoPE if it survives, and pre-Q/K+RoPE under one protocol;
6. **mechanism:** position-stratified loss, attention entropy/distance/sink
   profiles, and carrier-logit attribution;
7. **efficiency:** tokens/s, memory, parameter count, and equal-wall-clock as
   well as equal-step comparisons where throughput differs.

A broader positional-encoding paper should additionally include NoPE, learned
absolute position, classic sinusoidal input, and at least one recognized
relative-bias baseline such as ALiBi. Partial RoPE/p-RoPE and a dedicated
Fourier-prior construction are more informative modern comparisons than a
large collection of theta/rescaling variants. Every added baseline must be
implemented under the same decoder, tokenizer, data order, token budget, and
evaluation protocol.

After the principal comparison is frozen, the highest-value transfer tests
are:

- a second text corpus, before claiming modality generality;
- training at context 512 or 2048 plus evaluation across context lengths;
- one learning-rate robustness pair, rather than a broad hyperparameter sweep;
- longer training closer to a compute-appropriate token budget.

Cross-modality experiments are optional and should be attempted only if the
claim is deliberately expanded beyond autoregressive language modeling.

### Deferred 2D transfer design

If the claim is expanded to spatial models, test the same architectural idea
rather than inventing a new dynamic controller. For a ViT, construct a fixed
separable 2D carrier from row and column sinusoids, inject it locally before
Q/K, and compare NoPE, standard 2D absolute position, 2D RoPE, and pre-Q/K
carrier + 2D RoPE under matched ImageNet classification training. The central
question is again whether additive absolute Fourier features complement a
relative rotational geometry.

A DiT is a higher-cost second transfer test. Keep the diffusion-timestep and
class-conditioning paths unchanged, apply the new carrier only to spatial
token coordinates, and compare 2D RoPE with and without the pre-Q/K carrier.
Report optimization curves and sample quality at matched training compute;
do not infer generative benefit from the ViT result alone. In either setting,
start with tied scalar carrier gates and fixed spatial frequencies. Separate
row/column or Q/K amplitudes are follow-up ablations only after transfer is
established.

## Explicitly deferred

- further learned phase, amplitude-envelope, or frequency searches;
- new Q/K coupling or mapper sweeps;
- EMA, scan, recurrent, or content-dependent RoPE controllers;
- a full after-RoPE carrier implementation unless the pre-RoPE hybrid survives;
- three-seed replication of every Phase 39A cell.

These do not presently answer a higher-value question than the fixed-method
robustness, location, AddRoPE, and mechanism comparisons above.
