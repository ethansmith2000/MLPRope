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

Run only after Phase 38 artifacts are checked. This stage should primarily
evaluate trained models rather than retrain variants:

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

The mechanism analysis should use identical examples and bootstrap at the
example/document level. Token-level samples within a document are not
independent and must not be treated as such.

## Later evidence, not automatic

- longer training closer to compute-optimal tokens per parameter;
- context-length transfer above 1024 after allocating matching caches;
- a second dataset or modality;
- a direct mature AddRoPE-versus-pre-Q/K comparison under the same protocol.

These are paper-strength extensions, but Phase 38 should determine whether the
core result merits their cost.
