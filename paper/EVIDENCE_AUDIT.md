# Paper evidence audit

_Updated 2026-09-18 after Phase 57 completion. This document maps proposed claims to
completed evidence and remaining experiments. It is not a new search plan._

## Candidate hierarchy

### Primary method: scalar pre-Q/K carrier + RoPE

At each attention layer, add a fixed model-width sinusoid to the normalized
residual input of `W_q` and `W_k`, controlled by one learned scalar initialized
to one and tied between Q and K. Standard Q/K RMS normalization and standard
RoPE follow. The method adds one parameter per layer and does not change the
attention kernel, V, or the persistent residual stream.

### Enhanced method: scalar carrier + calibrated rank-32 Q/K readout

Add a shared rank-32 projection of the same sinusoid with separate Q/K
readouts after `W_q` and `W_k`, while retaining the scalar carrier as the exact
training anchor. At h768/d8 this adds 589,824 parameters. Only the
zero-initialized output factors use the frozen LR multiplier `6.367487` with
inverse AdamW-decay compensation.

The scalar method is the clean central contribution. Rank 32 is the strongest
learned extension, not a replacement for the minimal method. AddRoPE,
residual-input sinusoids, amplitude/frequency variants, ordinary Q/K biases,
and no-anchor rank 32 are controls or closed branches.

## Claim-to-evidence matrix

| Proposed claim | Evidence | Status | Remaining requirement |
|---|---|---|---|
| Scalar pre-Q/K improves RoPE | Phase 38: three batch-8 seeds, mean `-0.055334`; Phase 50 batch-32 seeds, mean `-0.036654` | strong | none for the canonical OpenWebText claim |
| The gain survives scale | h1024/d12, delta `-0.040581` | positive, one seed/older recipe | fresh matched L-scale pair only if scale is a headline claim |
| The gain is not caused by QKNorm | no-QKNorm pair, delta `-0.049523` | positive, one seed | none unless adopting a new backbone changes the interaction |
| Carrier and RoPE are complementary | Phase-42 mature `N/R/C/C+R` factorial | complete at one seed | report the positive but sub-additive interaction accurately |
| Learned magnitude matters | fixed gate is `+0.006464` worse than learned | complete at one seed | do not claim no fixed scalar can work without a separately tuned fixed-alpha study |
| Per-layer gates matter | global gate differs by only `+0.000421` | not supported | present global sharing as an equivalent simplification, not a necessity |
| Repeated attention-local access matters | first-block-only `+0.002696`; input-only `+0.006635` | supported at one seed | phrase first-block result as modest; location result is stronger |
| Pre-Q/K beats native head-space placement | fixed post-RoPE carrier is `+0.004578` worse | supported at one seed | retain ordering/QKNorm caveat |
| Rank-32 improves scalar | Phase 50: all three seeds, mean `-0.009103` | strong at canonical architecture | at least one corpus or backbone transfer if emphasized beyond an extension |
| Rank-32 is positional rather than generic capacity | beats matched FFN control in all seeds, mean `-0.006752` | supported | mechanism profiles strengthen interpretation but do not replace the control |
| Rank 128 is necessary | only `-0.000750` vs rank 32, interval crosses zero | rejected | none; omit as a promoted method |
| Ordinary Q/K bias explains rank 32 | bias arm vs scalar `+0.000101` despite active biases | rejected | none |
| Scalar is training scaffolding for rank 32 | no-anchor rank 32 is `+0.008295` worse; trained scalar-zero intervention is null | supported at seed 123 | extra no-anchor seeds only if this becomes a headline causal claim |
| The scalar changes attention concentration consistently | Phase 53: entropy lower and first-token mass higher in all three seeds; mass shifts from distance 64--255 to 1--3 | supported | do not replace this with the unsupported claim that mean attended distance always decreases |
| Carrier gains occur throughout the trained context | Phase 53: scalar-minus-RoPE and seed-mean rank32-minus-scalar are negative in every target-position bin | supported in-distribution | no length-extrapolation claim |
| Explicit carrier logits include meaningful content--position terms | Phase 53 exact conditioned-denominator decomposition | supported descriptively | retain distinction from a network-level causal ablation |
| The carrier is not fragile to a neighboring phase-origin shift | Phase 54: offsets 1/4 cost at most `0.000439` mean NLL across both methods | supported at trained checkpoints | large offsets are damaging; do not claim randomized-origin training robustness |
| Scalar benefit is not a single-LR artifact | Phases 55--56: deltas `-0.029164/-0.036355/-0.044780/-0.054474` at `1.5e-4/3e-4/6e-4/1.2e-3`, all block-32 intervals below zero | strong at one seed; `1.2e-3` selected prospectively | no further optimizer expansion planned |
| Corpus generalization | none | missing | pinned FineWeb-Edu matched comparison |
| Modern decoder transfer | none | missing | matched modern-backbone comparison |
| Broad comparison with recognized PE methods | Phase 57: scalar `3.125383`, ALiBi `3.137824`, fixed input sinusoid `3.138505`, RoPE `3.181325`; scalar-minus-RoPE `-0.055942` | complete at one seed | controlled implementations are not exact external recipe reproductions; ALiBi used FlexAttention |
| Two-dimensional transfer | none | optional/missing | ViT-S/16 screen then full recipe if positive |

## Completed canonical paper components

The Phase-42 batch-32, 100k, seed-123 matrix already supplies the core
component and location table: NoPE, RoPE, carrier without RoPE, scalar carrier
with RoPE, fixed gate, global gate, first-layer-only carrier, input sinusoid +
RoPE, fixed AddRoPE without RoPE, and fixed post-RoPE carrier + RoPE. Do not
rerun this table merely to rename methods.

Phase 49 supplies the mature rank-32 and matched-FFN comparison. Phase 50
supplies fresh seeds 456 and 789. Phases 51 and 52 supply checkpoint
counterfactuals, the Q/K-bias control, and the scalar-scaffold training test.
Phase 53 supplies the registered three-seed position-loss, attention-geometry,
and exact local-logit analysis. Its endpoint re-evaluation reproduced every
saved checkpoint within `1.0e-4` NLL.

Phase 57 supplies the fresh seven-arm recognized positional-baseline table at
the prospectively selected `1.2e-3` recipe. Scalar pre-Q/K + RoPE ranked first;
its delta versus standard RoPE was `-0.055942`, with block-32 interval
`[-0.057417,-0.054423]`. ALiBi and the fixed input sinusoid also beat RoPE but
trailed scalar by `+0.012441` and `+0.013121`. This is a one-training-seed
comparison; its paired block intervals are endpoint precision, not seed
variability.

## Completed mechanism evidence

On the common 1,024-block Phase-53 holdout, scalar-minus-RoPE is `-0.036640`
NLL and rank32-minus-scalar is `-0.009112`, consistent with the original
endpoint report. The scalar benefit is negative in every predeclared target
position bin and every seed. Rank 32's seed-average increment is negative in
all seven bins; one seed is slightly positive in the 16--31 bin, so the
stronger per-seed statement is not warranted.

Across 64 predeclared attention blocks per checkpoint, the scalar method
reduces normalized entropy by `0.039706`, increases first-token mass by
`0.012133`, increases mass at relative distances 1--3 by `0.023981`, and
reduces mass at distances 64--255 by `0.021888`. All four directions agree
across the three seeds. Mean attended-distance fraction does not: it is
slightly positive in seed 123 and negative in seeds 456 and 789. The rank-32
extension has nearly the same coarse attention geometry as scalar.

The exact local decomposition uses the common RMS denominator of the trained
sum. Its four components are algebraically exact and match logits recomputed
from full Q/K to maximum absolute error `4.005e-5`. Scalar and rank 32
have combined position-involving centered-logit RMS `0.836647` and `1.644923`,
respectively; content-only attention differs from full attention by mean KL
`0.705940` and `1.913295`. Both have substantial content--position cross
terms. These statistics are conditioned on each model's own hidden states and
must not be described as independently normalized components or causal
network ablations.

## Remaining language-model experiments

### Required for a credible generalization claim

1. **Modern M-scale decoder:** RoPE, scalar, and rank 32 with pre-RMSNorm,
   SwiGLU, tied embeddings, bias-free linears, and no learned input projection.
   Treat this as bundle transfer, not an attribution ablation.
2. **FineWeb-Edu M scale:** RoPE, scalar, and rank 32 at one paired seed. Pin
   the dataset revision and define document-hash train/development/final
   partitions before tokenization. Freeze a common token budget before launch.

### Strong reviewer-proofing, but schedulable after the required set

1. **Fresh L-scale pair:** RoPE and scalar under a newly frozen matched token
   budget. Include rank 32 only if it transfers on FineWeb-Edu or the modern
   backbone.
2. **Additional seeds on a new axis:** only after a one-seed paired result is
   positive and material. Do not replicate failed transfers automatically.

Context length is fixed at 1,024 by design. No context-size sweep, learned
frequency search, mapper expansion, or blanket repetition of negative variants
is part of the remaining paper work.

## Optional broader paper

Run ViT-S/16 ImageNet-1k as the first 2D test: learned absolute position, 2D
RoPE, separable 2D scalar pre-Q/K + RoPE, and a fixed post-RoPE location
control. Use a 100-epoch failure screen before the 300-epoch evidence recipe.
Only a positive full result triggers seeds or downstream vision tasks.

A roughly 1.3B modern decoder on four H200s is conditional on positive
M-scale modern-backbone transfer. A DiT experiment is conditional on positive
ViT transfer.

## Reporting constraints

- Keep batch-8/200k and batch-32/100k cohorts separate; they differ in tokens
  and optimizer updates.
- Training seeds are replicates. Block bootstraps quantify holdout precision
  within a seed.
- Do not claim context extrapolation, cross-corpus robustness, modern-backbone
  robustness, or modality generality before the corresponding experiment.
- Report the factorial as complementary but sub-additive: adding the carrier
  helps with and without RoPE, but their joint improvement is smaller than the
  sum of their separate improvements.
- Treat Phase-51 removals as endpoint interventions and Phase-52 no-anchor as
  the training-path test.
- Preserve the rank-32 FFN-capacity control and the fact that global scalar
  sharing was statistically tied with per-layer gates.
