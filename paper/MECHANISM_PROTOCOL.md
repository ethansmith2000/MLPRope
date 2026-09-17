# Phase 53 paper-mechanism protocol

Status: frozen before checkpoint evaluation and completed on 2026-09-12.

## Purpose

Phase 53 characterizes the already-trained RoPE, scalar pre-Q/K, and calibrated
rank-32 checkpoints. It does not select or train a method. All three training
seeds (123, 456, 789) are included, and the seed—not a token, head, or
layer—is the replication unit.

## Inputs and sampling

- Source checkpoints: the Phase-49 seed-123 and Phase-50 seed-456/789 final
  weights.
- Loss window: all 1,024 validation blocks `[4096, 5119]`, identical across
  methods within each seed.
- Attention window: 64 evenly spaced blocks from that window, using offsets
  `8 + 16j` for `j=0,...,63`.
- Sequence length: 1,024 stored tokens, giving 1,023 next-token targets and
  1,023 attention queries.
- No source checkpoint is modified. New artifacts are JSON summaries and
  compressed NumPy arrays only.

The attention subset is fixed before inspecting any mechanism result. Spacing
the blocks reduces immediate adjacency but does not make them independent
documents; within-seed block intervals remain sampling-sensitivity analyses.

## Position-stratified loss

Compute unreduced next-token cross-entropy and retain the mean for target
position bins `[1,16)`, `[16,32)`, `[32,64)`, `[64,128)`, `[128,256)`,
`[256,512)`, and `[512,1024)`. Preserve per-block bin means so comparisons can
remain paired. The full mean must reproduce the saved final evaluation within
`5e-4` NLL or the analysis fails.

## Attention geometry

For every sampled block, layer, and head, reconstruct causal attention in
fp32 from the trained hidden state and report:

- entropy divided by `log(number of visible keys)`, excluding the first query;
- expected attended distance in tokens and divided by the query position;
- mean mass on the first key;
- mass in exclusive relative-distance bins `0`, `1--3`, `4--15`, `16--63`,
  `64--255`, and `256+`;
- across-query correlations of query position with entropy, normalized
  distance, and first-key mass, using queries 16 onward.

These are descriptive attention summaries. They condition on each model's
trained hidden states and do not treat layers or heads as independent samples.

## Exact local logit decomposition

For each attention layer write the raw projected vectors as

```text
q_raw = q_c + q_p
k_raw = k_c + k_p
```

where `q_c=W_q h`, while `q_p` contains `W_q(alpha s)` and, for rank 32, its
dedicated `U_q D s` readout. The K branch is analogous. Use the RMS denominator
computed from the complete trained vector, apply the learned Q/K RMS gain to
each component, and rotate every component with the same standard RoPE. The
four scaled score terms are then

```text
L_cc = <R q_c, R k_c>
L_cp = <R q_c, R k_p>
L_pc = <R q_p, R k_c>
L_pp = <R q_p, R k_p>.
```

They must sum numerically to the reconstructed trained logits. Report
per-query-centered RMS for every term, the combined positional term, its
cosine with the total centered logits, and the local attention KL between the
full distribution and `softmax(L_cc)`.

Centering over each query's visible keys removes row-constant score shifts,
which cannot affect softmax. Because every component shares the denominator
from the full trained vector, this is an exact additive decomposition at the
observed state. It is not a counterfactual network: independently normalizing
or removing a component would change the denominator and downstream hidden
states. Phase 51 supplies the separate model-level causal interventions.

## Reporting

- Preserve each training seed separately and report means and sample standard
  deviations across seeds.
- Use paired block bootstrap intervals only as within-seed sampling precision.
- Do not pool heads, layers, tokens, or blocks and call them training
  replications.
- Do not use Phase 53 to tune the method or determine which examples enter a
  later headline evaluation.

## Outcome

All nine checkpoint evaluations passed endpoint reproduction. The aggregate
report and machine-readable evidence are in
`results/phase53_paper_mechanism/`. No weights or checkpoints were written.
