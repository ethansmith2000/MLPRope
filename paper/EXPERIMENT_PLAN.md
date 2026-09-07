# Frozen experiment plan: attention-local sinusoidal carriers

Status: working protocol, frozen before the paper-evidence runs. Changes that
affect a primary endpoint must be dated and justified here before launching.

Revision 2026-09-06: refocused the paper ablations on necessity of the final
method's components, removed exploratory learned carrier variants from the
main matrix, and added a measured 8x5090/conditional 4xH200 execution plan.

## 1. Claim and method freeze

The narrow primary claim is:

> A fixed sinusoidal carrier injected locally before Q/K projections improves
> a RoPE causal decoder at negligible parameter and throughput cost.

The primary method is fixed:

- standard RoPE remains unchanged;
- at every attention layer, add the same model-space sinusoid to the normalized
  residual separately on the Q and K branches;
- use one directly optimized, unconstrained fp32 scalar per layer, tied between
  Q and K and initialized to `1.0`;
- allow the existing `W_q` and `W_k` matrices to read the shared carrier
  separately;
- do not add the carrier to V or write it into the persistent residual stream;
- use fixed model-space frequencies with the same base `theta=10000` as RoPE
  (the sampled banks differ because model width and head width differ);
- use ordinary fused causal SDPA.

Separate Q/K gates, per-frequency amplitudes, learned phase or frequency,
content-dependent mappers, EMA/scan controllers, and dynamic RoPE are closed
exploratory branches. They are not part of the paper's main ablation matrix.
At most, consolidate them into a compact supplementary table or a separate
negative-results archive if they help delimit the method; do not spend paper
compute reproducing them.

## 2. Questions the paper must answer

1. **Quality:** Does the method improve held-out language-model NLL over RoPE
   after mature training and across training seeds?
2. **Component necessity:** Are both the carrier and RoPE needed, must the
   carrier magnitude be learned, must it vary by layer, and must the carrier
   be presented again at every attention block?
3. **Location:** Is repeated attention-local injection better than conventional
   residual-input injection or native head-space addition?
4. **Robustness:** Does the effect survive changes in batch regime, scale,
   corpus, QK normalization, and a modernized decoder backbone?
5. **Mechanism:** Does it alter positional loss, entropy, attended distance,
   sink behavior, or content-position logit terms in a consistent way?
6. **Transfer:** Does the same construction help separable 2D spatial
   attention in a ViT?

## 3. Candidate and comparison set

### 3.1 Core mature comparison at the canonical model

All cells below should use width 768, depth 8, eight heads, context 1024, and
seed 123. The completed batch-8/200k-step cohort is retained as preliminary
evidence. The proposed primary paper cohort uses batch 32 and 100k steps
(3.277B nominal tokens); every matched cell must be rerun under that recipe.

| ID | Method | Purpose | Existing batch-8 evidence |
|---|---|---|---|
| `N` | no positional encoding | completes the factorial | missing |
| `R` | standard RoPE | primary control | complete |
| `C+R` | pre-Q/K carrier + RoPE | primary method | complete |
| `C` | pre-Q/K carrier, no RoPE | carrier-only factorial cell | complete |
| `F+R` | pre-Q/K carrier + RoPE, all gates fixed at 1 | tests whether natural fixed unit scale suffices | required |
| `G+R` | pre-Q/K carrier + RoPE, one learned global gate | tests whether layerwise gates are necessary | required |
| `C_1+R` | pre-Q/K carrier only in the first block + RoPE | tests whether repeated access is necessary | required |
| `I+R` | one-shot residual-input sinusoid + RoPE | conventional injection location | required location control |
| `A` | fixed native head-space AddRoPE, no RoPE | closest additive-attention comparator | required comparator |
| `A_post+R` | fixed native head-space carrier after RoPE | head-space location control | required location control |

The complete `N`, `R`, `C`, and `C+R` factorial estimates the RoPE-by-carrier
interaction. `F+R`, `G+R`, and `C_1+R` then remove, one at a time, learning,
layerwise freedom, and repeated presentation from the surviving method.
`I+R`, `C+R`, and `A_post+R` compare three meaningful application sites,
although the head-space carrier is not algebraically identical to the
model-space carrier because it bypasses `W_q` and `W_k`.

The fixed native AddRoPE method without RoPE is the closest additive-attention
comparator and should receive one mature run. It is a comparison method, not a
component ablation of `C+R`. The learned direct post-RoPE carrier is not a core
candidate: at 30k it improves over the fixed carrier by only 0.0028 nats while
adding 12,288 parameters.

The incorrect `A_pre+R` ordering is retained as a 30k mechanistic ablation. It
need not receive a mature run unless ordering becomes a headline claim rather
than an explanatory result.

### 3.2 Recognized positional baselines

Run these under the same canonical decoder for a single-seed comparison table:

| Baseline | Why include it | Priority |
|---|---|---|
| classic input sinusoid without RoPE | original Transformer baseline | required |
| learned absolute position embedding | standard unconstrained absolute baseline | required |
| ALiBi | recognized relative-bias baseline | required if the paper claims a broader PE comparison |
| partial RoPE (`p`-RoPE) | contemporary, inexpensive RoPE-frequency comparator | recommended |
| globally learned RoPE frequency | connects the result to learned-spectrum work | reuse Phase 34 as an internal ablation; do not call it an exact reproduction without auditing the external recipe |
| appended-channel Fourier prior | closest functional comparator to a dedicated additive attention prior | conditional on a direct implementation/code audit |

ALiBi requires a bias-capable attention path and should report its actual
throughput rather than silently comparing a slower kernel. CoPE, accumulated
content-conditioned rotations, and RoPE/NoPE layer mixtures belong in related
work; they are not mandatory head-to-head baselines for the narrow fixed-
carrier claim because they change the mechanism and often the attention path.

### 3.3 What not to multiply across every axis

Only `R` and `C+R` are carried through the scale/corpus/architecture axes.
`A_post+R` is promoted beyond the canonical model only if its 200k result
remains materially competitive. The large baseline table is run at one model
size, not at every scale. This prevents the paper matrix from becoming a
Cartesian product.

## 4. Ablations

### 4.1 Primary component ablations

| Question | Contrast | Interpretation |
|---|---|---|
| Is the carrier useful, and does it complement RoPE? | `N`, `R`, `C`, `C+R` | complete 2-by-2 carrier/RoPE factorial |
| Does natural fixed unit scale suffice? | `F+R` vs `C+R` | fixed `alpha_l=1` versus direct learned gates initialized at 1 |
| Must the learned magnitude vary by layer? | `G+R` vs `C+R` | one globally shared gate versus one gate per layer |
| Must position be re-presented at every block? | `C_1+R` vs `C+R` | first-block-only versus all-block pre-Q/K injection |
| Does attention-local placement matter? | `I+R`, `C+R`, `A_post+R` | persistent input, model-space pre-projection, and native head-space paths |

These are the paper ablations because each removes a stated ingredient of the
method. Run them to the canonical mature endpoint at one paired seed. The
central three-seed result remains `C+R` versus `R`; the ablations do not need
three seeds unless one becomes a headline claim or lands near a decision
boundary.

The fixed-gate test deliberately fixes every gate to the declared
initialization value, 1.0. Choosing a post-hoc fixed value from the learned run
would test distillation of a discovered schedule, not whether the original
learnable component is necessary. Such a follow-up can be labeled separately
if useful. Conversely, `F+R` alone cannot establish that no fixed scalar can
match learned gates. If the stronger claim that online gate learning is
necessary matters, predeclare a small global fixed-alpha sweep using only the
development split, choose once, and confirm that value in a fresh mature run.

### 4.2 Mechanism and robustness controls

| Control | Status/use |
|---|---|
| QK normalization on/off for `R` and `C+R` | mature and positive; robustness evidence |
| Q-only and K-only injection | optional 30k decomposition of the two cross terms |
| early-half and late-half injection | run only if learned gates or attribution show a depth pattern |
| native carrier before versus after RoPE | completed 30k ordering explanation; appendix unless promoted |

Q/K split gates, per-frequency amplitudes, phases, learned frequencies,
content-dependent mappers, and EMA controllers were development searches, not
component ablations of the final method. Preserve their resolved configs and
results, but do not place them alongside the causal decomposition above or
rerun them merely to make a larger ablation table.

## 5. Hyperparameter policy

### 5.1 Do not tune each method independently

Use one architecture-level recipe for all paired methods. Per-method learning
rates would make it unclear whether an improvement comes from position or from
extra optimization. The carrier gate uses the model learning rate, direct
parameterization, initialization 1.0, and the existing weight-decay policy.

The current gate is active at initialization. A zero initialization would make
the early model position-unaware on the carrier branch and would answer a
different optimization question. Existing runs show finite gradients,
substantial gate movement, and no intervention-specific late clipping; there
is no evidence that a special gate learning-rate multiplier is needed.

### 5.2 One method-by-learning-rate robustness check

At the canonical architecture, compare `R` and `C+R` at peak learning rates:

```text
1.5e-4, 3.0e-4 (existing primary), 6.0e-4
```

Keep AdamW betas, 200 warmup steps, weight decay, batch tokens, and linear decay
fixed. The two outer points may use 100k steps as a robustness test; they are
not used to select a new headline endpoint. If either arm is unstable at
`6e-4`, replace it with `4.5e-4` and record the change before inspecting final
holdout results.

No theta, phase, frequency, amplitude, or gate-LR sweep is planned.

## 6. Architecture

### 6.1 Controlled primary backbone

Freeze the architecture already supporting the mature evidence:

- decoder-only causal Transformer;
- learned token embedding, followed by LayerNorm and a learned input linear;
- pre-LayerNorm residual blocks;
- multi-head attention with bias-free Q/K/V projections and biased output;
- head-vector Q and K RMS normalization with learned gains shared across
  heads, applied before RoPE;
- full-head RoPE, `theta=10000`;
- GeGLU MLP with hidden width `4d`;
- no dropout;
- untied output language-model head;
- PyTorch fused SDPA, bf16, `torch.compile`.

This is a controlled research architecture, not a Llama replica. State its
unusual choices explicitly, especially the input projection, untied embedding
and head, and the relatively wide GeGLU.

### 6.2 Modern-backbone transfer

After the core mature table, test only `R` and `C+R` in one deliberately
modernized decoder:

- pre-RMSNorm blocks;
- SwiGLU with approximately `8d/3` hidden width, rounded for kernels;
- bias-free attention and MLP linears;
- tied token embedding and LM head;
- standard MHA for the first transfer (do not add GQA simultaneously);
- full-head RoPE and the same method-aware QK RMSNorm;
- no separate learned input projection.

This changes a bundle of architectural conventions intentionally: it is a
transfer test, not an attribution experiment. If the result fails, decompose
the bundle in a separate development study. GQA/MQA is a later robustness
test, not part of the first modern transfer.

## 7. Model scales, batch size, and hardware

### 7.1 Controlled scales

The controlled scale family is:

| Label | Width | Layers | Heads | Head dim | Total params | Non-embedding/head params | Status |
|---|---:|---:|---:|---:|---:|---:|---|
| S | 512 | 6 | 8 | 64 | 77.04M | 25.47M | optional new scale |
| M | 768 | 8 | 8 | 96 | 153.50M | 76.18M | primary, mature |
| L | 1024 | 12 | 8 | 128 | 305.63M | 202.56M | transfer, mature |

Because embeddings are untied, report both total and non-embedding/head
counts. Do not call this a scaling-law study: head dimension and token-to-
parameter ratio vary. The paper's scale claim is only that the paired method
effect transfers from M to L. Add S if a three-point trend is desired.

### 7.2 Batch-size policy and measured 5090 envelope

The current machine has eight RTX 5090 GPUs with 32 GB each. Measurements from
the existing compiled bf16 training path are:

| Model | Context | Sequences/update | Peak allocated | Throughput | Endpoint wall time |
|---|---:|---:|---:|---:|---:|
| M, 153M | 1024 | 8 | 5.0 GiB | 188k tokens/s | 2.47 h at 200k steps |
| M, 153M | 1024 | 16 | 7.9 GiB | 208k tokens/s | benchmark only |
| M, 153M | 1024 | 32 | 13.9 GiB | 215k tokens/s | about 4.2 h at 100k steps |
| M, 153M | 1024 | 64 | 25.9 GiB | 222k tokens/s | benchmark only |
| L, 306M | 1024 | 8 | 9.1 GiB | 88k tokens/s | 5.15 h at 200k steps |

These figures are implementation-specific measurements, not hardware claims.
They show that VRAM is not limiting the current M/L experiments. The completed
cohort used 8192 tokens per optimizer update and must remain labeled as that
optimization regime. A measured benchmark of sequence batches 8, 16, 32, and
64 selected batch 32 for the paper cohort: it provides 32,768 tokens per update
and 100k updates, for 3.277B training tokens. Batch 64 improves throughput by
only 3.2% over batch 32 while increasing peak allocated memory by 86%.
Freeze the batch, learning-rate schedule, and token budget before comparing
methods; do not mix the new cohort with old controls as if their training
protocols were identical.

Run the canonical jobs as independent single-GPU processes. Eight independent
paired experiments provide much more evidence per wall-clock hour than using
all eight GPUs for data parallelism on these small models. In particular,
eight-way data parallelism with 8192 local tokens would change the global batch
to 65,536 and reduce the number of optimizer updates at a matched token budget.
Use the old batch-8 results as preliminary evidence and as an explicit
small-batch robustness cohort. If batch 32 is adopted, rerun every primary
paper control and ablation under the new frozen recipe.

For any new architecture with no reused controls, choose the effective batch
once from a short throughput/memory pilot, then freeze it for both `R` and
`C+R`. Report microbatch, accumulation, number of devices, global tokens per
update, and optimizer steps separately.

### 7.3 Token budgets

| Cohort | Sequence batch | Steps | Nominal tokens | Role |
|---|---:|---:|---:|---|
| completed small-batch | 8 | 200k | 1.638B | preliminary evidence and batch robustness |
| proposed M paper cohort | 32 | 100k | 3.277B | primary matched table |
| implementation screen | 32 | at most 10k | at most 328M | failures only, never headline evidence |

The proposed cohort has half as many optimizer updates but twice as many
training tokens as the completed cohort. This is a deliberate new optimization
regime, not a continuation of an old checkpoint. If a longer endpoint is later
needed, its scheduler must be defined for that endpoint from the start.

At measured throughput, a 3.277B-token M run should take roughly 4.2 hours
before periodic evaluation and compilation. Refresh the estimate after the
first completed run rather than extrapolating it to other model scales.

### 7.4 Conditional H200 scale tier

Four H200s are unnecessary for the canonical M/L evidence. Reserve them for a
qualitatively new regime: a modern roughly billion-parameter decoder or a
diffusion experiment whose activation memory makes the 5090s inefficient. A
concrete language-model candidate is width 2048, 24
layers, 16 heads, tied embeddings, and the modern backbone in Section 6.2
(approximately 1.3B parameters, subject to an exact implementation count).

For that new tier, start with a shared target of 65,536 global tokens per
optimizer update and a token-based training schedule; this is a starting point
for a paired pilot, not an inherited claim from the small-model setup.
Benchmark single-GPU memory and four-GPU distributed throughput before
freezing the microbatch and accumulation. Only run the full scaled pair if the
M-scale modern-backbone transfer is positive. This makes H200 time
evidence-bearing rather than an expensive way to repeat jobs that already fit
comfortably.

The proposed evidence configuration is context 1024 and 20B training tokens,
or about 305k optimizer updates at the proposed global batch. Use a schedule
defined for the full 20B tokens from the start and retain intermediate 5B and
10B checkpoints; do not train a decayed 10B-token run and then extend it. The
four devices may run the paired methods concurrently with two GPUs each or
sequentially with four GPUs each, whichever gives better measured end-to-end
throughput. Use a corpus slice with enough unique tokens rather than cycling a
10B-token sample without reporting it.

## 8. Data protocol and corpus generalization

### 8.1 Canonical corpus

OpenWebText protocol:

- `train[5%:]` training and `train[:5%]` validation;
- GPT-2 fast tokenizer;
- no EOS insertion;
- concatenate documents and chunk into 1024 tokens;
- deterministic cached block order;
- 128-block development slice and disjoint 1024-block final holdout starting
  at block 2048.

### 8.2 Second corpus

Use a pinned revision of the 10B-token FineWeb-Edu sample. Preserve the GPT-2
tokenizer and chunking policy so the main change is data distribution. Create
train/development/final partitions by deterministic document hash before
concatenation, record the dataset revision and file hashes, and use the same
1.638B-token training budget for `R` and `C+R` at scale M.

Do not compare raw NLL numerically across corpora as if token distributions
were identical; compare the paired method delta within each corpus.

All language-model training and primary evaluation use context 1024. Context
length is deliberately held fixed rather than treated as another experimental
axis. This keeps the paper focused and avoids conflating the carrier effect
with length extrapolation or RoPE-scaling choices.

## 9. Mechanism measurements

Prefer checkpoint analysis over training more variants:

1. token NLL by absolute-position and available-context buckets;
2. attention entropy divided by `log(number of visible keys)` for queries with
   at least two visible keys (report the first position separately);
3. expected attended distance and attention mass in logarithmic distance bins;
4. mass assigned to the first token and other attention sinks;
5. correlation of attention with absolute query/key position and relative
   distance, stratified by layer and head and controlled for the triangular
   causal support;
6. pre-normalization RMS of content-content, content-position,
   position-content, and position-position logit terms;
7. full-attention counterfactuals with carrier terms removed before QK
   normalization;
8. inference-time gate-zeroing, labeled as distribution shift;
9. learned gate trajectories and final values by depth.

Use identical examples for every model. Bootstrap documents/blocks, not tokens
or heads as if they were independent training replications. Correct or clearly
label exploratory multiple comparisons.

## 10. Statistical and reporting protocol

- Primary metric: final holdout mean token NLL; perplexity is a monotone
  secondary presentation.
- Primary contrast: `C+R - R`; negative is favorable.
- Pair data order and name-stable initialization within every contrast.
- Report every training seed individually plus mean and standard deviation of
  seed-level deltas.
- Paired bootstrap intervals over final-holdout blocks describe evaluation
  precision within a seed, not seed uncertainty.
- Use the same frozen holdout once per finalized run; development slices guide
  monitoring but not paper-method selection.
- Report parameters, peak allocated/reserved memory, tokens/s, and loss at
  matched wall-clock in addition to matched steps.
- Preserve resolved configs, source commit, dataset manifest, per-example
  losses, checkpoints needed for mechanism analysis, and completion markers.

The primary method already has three M-scale seeds. Defer blanket seed
replication. Add seeds to new axes only after a one-seed paired result shows
that the effect is present and scientifically relevant. A final paper should
have at least three seeds for the central result and for any new modality that
becomes a headline claim.

## 11. Two-dimensional transfer

The first cross-modality test is ImageNet-1k classification with ViT-S/16:

- 224x224 inputs, 16x16 patches;
- width 384, depth 12, six heads;
- a standard 300-epoch DeiT-style recipe shared by every positional method;
- standard axial or mixed 2D RoPE taken from an audited RoPE-ViT reference;
- a separable 2D carrier with half the frequency pairs assigned to row and
  half to column coordinates;
- one learned scalar per attention block, initialized with an RMS-matched
  contribution and tied between Q and K;
- carrier injection only before Q/K; V, class token handling, augmentation,
  and optimization remain unchanged.

Core ViT comparison:

| Method | Purpose |
|---|---|
| learned 2D absolute embedding | standard ViT control |
| 2D RoPE | rotary control |
| pre-Q/K 2D carrier + 2D RoPE | transferred primary method |
| post-RoPE fixed 2D carrier + 2D RoPE | transferred location control |

Run a one-seed 100-epoch screen only to catch failures and grossly negative
results. The evidence run is the complete 300-epoch recipe. Evaluate top-1 and
top-5 accuracy, throughput, memory, and resolution transfer from 224 to 384.
Only after a positive full-recipe result should additional seeds or downstream
detection/segmentation be scheduled.

A DiT is conditional on successful ViT transfer. The initial diffusion test
should use a small standard latent DiT on ImageNet 256, alter only spatial
position handling, compare 2D RoPE with and without the pre-Q/K carrier, and
report FID-50k at matched training compute. Do not use diffusion results as a
cheap substitute for the cleaner ViT attribution.

## 12. Execution order and stop rules

### Stage A: close the mature canonical table

1. Preserve the completed batch-8 `R`, `C`, and `C+R` evidence as its own
   cohort.
2. Under the frozen paper recipe, train `N`, `R`, `C`, `C+R`, `F+R`, `G+R`,
   and `C_1+R` at seed 123.
3. Train the matched comparator/location cells `I+R`, `A`, and `A_post+R`.
4. Preserve historical search results outside the main ablation matrix.
5. Run checkpoint-only mechanism analyses.

This stage is mandatory. If `C+R` does not remain the best attention-local
method at mature horizon, revise the paper hierarchy but do not hide the
result.

### Stage B: robustness breadth

1. FineWeb-Edu M-scale pair (`R`, `C+R`) at the frozen paper budget.
2. Learning-rate outer-point pairs at the frozen batch and a reduced budget.
3. Fresh L-scale `R`/`C+R` pair with an appropriately increased token budget.
4. Optional S-scale pair.
5. Modern-backbone M-scale pair.

Stop expanding an axis if the paired effect reverses materially. Diagnose the
interaction before averaging incompatible settings.

### Stage C: broader paper

1. Implement and audit the recognized baseline table at canonical M scale.
2. ViT-S/16 2D transfer.
3. Replicate only the successful transferred result.
4. If the modern M-scale result is positive, benchmark and run the roughly
   1.3B `R`/`C+R` pair on four H200s.
5. Consider DiT only after the ViT endpoint.

### Discovery during paper experiments

If a scale or modality suggests an improvement, treat it as a new development
hypothesis. Test it on a separate development slice/configuration, document
the revision, then rerun every affected primary comparison. Do not silently
tune against the frozen paper holdout.

The first such isolated branch is Phase 43: three rank-32, position-only
low-rank pathway variants at batch 32 for 20k steps. They test a residual
pre-map, a dedicated Q/K replacement, and a dedicated Q/K residual, alongside
matched 20k RoPE and scalar-carrier controls. Phase 43 does not alter the
Stage-A table. Promotion requires a meaningful improvement over the matched
parent together with active, finite adapter optimization; any promoted method
then requires a newly frozen matched evidence cohort.

For this development screen, replacement is paired with the 20k RoPE control;
pre-map and Q/K residual are paired with the 20k scalar-carrier control. The
predeclared promotion gate is `-0.003` mean NLL on the disjoint 1,024-example
holdout with a below-zero paired interval, a non-collapsing late curve, and
finite active adapter updates. A passing arm receives a longer confirmation
and a parameter-matched FFN control before affecting the primary method.

## 13. Immediate run recommendation

After the batch benchmark freezes the paper recipe, the first evidence batch
should be the primary seed-123 cells, not more learned carrier-shape variants:

1. NoPE (`N`);
2. RoPE (`R`);
3. pre-Q/K carrier without RoPE (`C`);
4. pre-Q/K carrier + RoPE (`C+R`);
5. fixed-gate carrier + RoPE (`F+R`);
6. globally shared learned gate + RoPE (`G+R`);
7. first-block-only pre-Q/K carrier + RoPE (`C_1+R`);
8. residual-input sinusoid + RoPE (`I+R`);
9. fixed native AddRoPE without RoPE (`A`);
10. fixed post-RoPE head-space carrier + RoPE (`A_post+R`).

Run up to seven concurrently on the currently available 5090s and let the
shared queue start the remaining three as devices free up. The old batch-8
controls are reused as preliminary and batch-robustness evidence, not as
matched controls for this new cohort.

## 14. Literature anchors for the protocol

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762): classic input
  sinusoid.
- [RoFormer](https://arxiv.org/abs/2104.09864): RoPE definition.
- [Round and Round We Go](https://arxiv.org/abs/2410.06205): mechanistic RoPE
  frequency analysis and partial RoPE.
- [RoPE to NoPE and Back Again](https://arxiv.org/abs/2501.18795): related
  hybrid RoPE/NoPE architecture, distinct from additive carrier composition.
- [Contextual Position Encoding](https://arxiv.org/abs/2405.18719): relevant
  content-conditioned positional work, distinct from the fixed carrier.
- [Rotary Position Embedding for Vision Transformer](https://arxiv.org/abs/2403.13298):
  audited starting point for 2D RoPE and resolution transfer.
- [FineWeb](https://arxiv.org/abs/2406.17557) and a candidate pinned
  [FineWeb-Edu 10B sample revision](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/tree/05c1931294b0d1379055d1f802d369f2c3bb2f4b/sample/10BT):
  proposed second-corpus source; pin a commit rather than using a moving branch.
- [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556):
  motivates increasing the token budget with model scale; the proposed XL
  setting is a transfer test, not a new scaling-law estimate.
