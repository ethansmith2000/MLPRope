# Phase 54 position-origin sensitivity protocol

Status: frozen before checkpoint evaluation on 2026-09-13.

## Question

Does a trained attention-local sinusoidal carrier depend narrowly on the
absolute origin `0` used for every 1,024-token training block?

This is an inference-time sensitivity analysis, not evidence that a model
trained with shifted or randomized origins would behave the same way. It
isolates the absolute carrier phase from the relative RoPE geometry.

## Models and data

- Retained scalar pre-Q/K and calibrated rank-32 checkpoints from Phases 49
  and 50.
- Training seeds `123`, `456`, and `789`.
- The same 1,024 validation blocks `[4096, 5119]` used by Phase 53.
- Per-block next-token NLL is retained for paired comparisons.
- No model is trained or modified persistently; no weights or checkpoints are
  written.

## Intervention

For carrier offset `c`, replace every layer's fixed carrier

```text
s(p) -> s(p + c)
```

for `c in {0, 1, 4, 16, 64, 256, 1024, 4096}`. The learned gates, Q/K
projections, low-rank readouts, Q/K normalization, tokens, and causal mask are
unchanged. Standard RoPE remains indexed by `p`: shifting both RoPE indices by
the same `c` cancels exactly in Q/K inner products, so leaving it at `p`
isolates the only functionally relevant common-origin change.

Offset zero must reproduce the Phase-53 endpoint within `5e-4` NLL. Report the
mean carrier-basis cosine between `s(p)` and `s(p+c)`, the NLL penalty relative
to offset zero, and the candidate-minus-RoPE contrast at each offset.

## Statistics and interpretation

- Training seed is the replication unit.
- Preserve every seed and report descriptive seed mean, sample standard
  deviation, and three-seed t interval.
- Use paired per-block intervals only as within-seed sampling precision.
- Small-offset degradation indicates phase-origin fragility of the trained
  solution. Large-offset degradation alone is weaker evidence because those
  phases are outside the training-origin distribution.
- Robustness here does not establish corpus, architecture, length, or
  randomized-origin training generalization.

## Follow-up rule

This audit does not select a new carrier. If offsets `1` or `4` cause a
material mean penalty (predeclared as at least `0.01` NLL), add one matched
random-origin training pair before making an origin-robustness claim. Otherwise
continue to the already frozen symmetric learning-rate experiment.
