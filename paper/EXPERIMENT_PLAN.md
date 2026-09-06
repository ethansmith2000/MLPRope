# Frozen experiment plan: attention-local sinusoidal carriers

Status: working protocol, frozen before the paper-evidence runs. Changes that
affect a primary endpoint must be dated and justified here before launching.

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

Separate Q/K gates, smooth per-frequency amplitudes, learned phase, learned
frequency, content-dependent mappers, EMA/scan controllers, and dynamic RoPE
are closed design branches. Existing mature results show no durable gain from
the static refinements, and the dynamic branches do not justify their
complexity. They may be reported as negative ablations but are not candidates.

## 2. Questions the paper must answer

1. **Quality:** Does the method improve held-out language-model NLL over RoPE
   after mature training and across training seeds?
2. **Complementarity:** Does the additive carrier contribute with and without
   RoPE, and is the joint effect more than either positional mechanism alone?
3. **Location:** Is repeated attention-local injection better than conventional
   residual-input injection or native head-space addition?
4. **Robustness:** Does the effect survive changes in scale, corpus, context
   length, QK normalization, and a modernized decoder backbone?
5. **Mechanism:** Does it alter positional loss, entropy, attended distance,
   sink behavior, or content-position logit terms in a consistent way?
6. **Transfer:** Does the same construction help separable 2D spatial
   attention in a ViT?

## 3. Candidate and comparison set

### 3.1 Core mature comparison at the canonical model

All cells below should reach the same 200k-step/1.638B-token endpoint at width
768, depth 8, eight heads, context 1024, and seed 123. Existing completed
models are reused only when their protocol fingerprint matches.

| ID | Method | Purpose | 200k status |
|---|---|---|---|
| `N` | no positional encoding | completes the factorial | required |
| `R` | standard RoPE | primary control | complete |
| `C+R` | pre-Q/K carrier + RoPE | primary method | complete |
| `C` | pre-Q/K carrier, no RoPE | carrier-only factorial cell | complete |
| `I+R` | one-shot residual-input sinusoid + RoPE | conventional injection location | required |
| `A` | fixed native head-space AddRoPE, no RoPE | additive-attention comparator | required |
| `A_post+R` | fixed native carrier after RoPE | clean head-space hybrid | required |

The complete `N`, `R`, `C`, and `C+R` factorial is required to estimate the
RoPE-by-carrier interaction. `I+R`, `C+R`, and `A_post+R` compare the same
general signal at three meaningful application sites. The learned direct
post-RoPE carrier is not a core candidate: at 30k it improves over the fixed
carrier by only 0.0028 nats while adding 12,288 parameters.

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

Only `R` and `C+R` are carried through the full scale/corpus/context matrix.
`A_post+R` is promoted beyond the canonical model only if its 200k result
remains materially competitive. The large baseline table is run at one model
size, not at every scale. This prevents the paper matrix from becoming a
Cartesian product.

## 4. Ablations

### 4.1 Required, mostly already complete

| Ablation | Contrast | Evidence/status |
|---|---|---|
| RoPE complementarity | NoPE/RoPE x carrier off/on | missing mature NoPE only |
| injection location | input vs pre-Q/K vs post-RoPE head space | 30k complete; mature controls specified above |
| AddRoPE ordering | native carrier before vs after RoPE | 30k complete; large effect |
| QK normalization | QKNorm on vs off for `R` and `C+R` | 200k complete |
| Q/K carrier coupling | tied scalar vs split Q/K scalar | 200k complete; null |
| carrier amplitude | scalar vs per-frequency amplitude | 200k complete; null |
| carrier phase | amplitude vs amplitude+phase | 200k complete; null |
| carrier frequency | fixed vs learned global/smooth frequency | 200k complete; null |

These historical ablations should be consolidated into one appendix table
rather than rerun.

### 4.2 Small targeted ablations still worth considering

These answer questions about the surviving method rather than searching for a
new carrier:

1. **Learned gate versus fixed gate:** `alpha_l` learned from 1.0 versus fixed
   at 1.0. This determines whether adaptation is needed at all.
2. **Per-layer versus globally shared gate:** tests whether eight independent
   layer magnitudes are important. It changes only seven parameters.
3. **Q-only, K-only, and Q+K injection:** isolates the two content-position
   cross terms. This is mechanism evidence, not a promotion sweep.
4. **Layer support:** all layers versus early half versus late half, only if
   gate trajectories or attribution show a strong depth pattern.

Run items 1--3 initially for 30k at one paired seed. Promote only an ablation
with an effect larger than 0.01 nats or one essential to interpreting the
method. Do not use these cells to revise the primary method after seeing paper
test sets without declaring a new development phase.

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

No theta, phase, frequency, amplitude, or gate-LR sweep is planned. For a
longer-context experiment, change RoPE scaling only if both `R` and `C+R` use
the identical standard scaling rule.

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

## 7. Model scales and token budgets

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

Primary language runs use a fixed token budget across methods and, initially,
across scales:

| Steps | Nominal tokens at 8192/step | Use |
|---:|---:|---|
| 20k--30k | 164M--246M | implementation/large-effect screen only |
| 100k | 819M | robustness or promoted ablation |
| 200k | 1.638B | minimum mature paper endpoint |
| 400k | 3.277B | optional late-training test for L; requires a fresh schedule |

Do not extend a run whose linear scheduler already reached zero at 200k and
describe it as a continuous 400k run. A 400k endpoint must be trained with a
400k schedule from the beginning.

At the measured rates, one 200k M run takes about 2.5 hours on the current GPU;
one 200k L run takes about 5.2 hours. Wall-clock estimates should be refreshed
after any architecture or context change.

## 8. Data and context generalization

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

### 8.3 Context-length training

Test `R` and `C+R` at training lengths 512, 1024, and 2048 while keeping
nominal tokens per optimizer step fixed at 8192:

| Context | Sequences/step | Steps | Nominal tokens |
|---:|---:|---:|---:|
| 512 | 16 | 200k | 1.638B |
| 1024 | 8 | 200k | 1.638B |
| 2048 | 4 | 200k | 1.638B |

Use gradient accumulation if a microbatch does not fit, without changing the
effective batch tokens. Primary evaluation is in-distribution at the training
length. Extrapolation evaluation at 2x and 4x length is secondary and must
report position-wise loss, short-context retention, and the exact RoPE scaling
rule. A model's ability to execute the sinusoid at a longer length is not by
itself evidence of useful length generalization.

## 9. Mechanism measurements

Prefer checkpoint analysis over training more variants:

1. token NLL by absolute-position and available-context buckets;
2. attention entropy divided by `log(number of visible keys)`;
3. expected attended distance and attention mass in logarithmic distance bins;
4. mass assigned to the first token and other attention sinks;
5. correlation of attention with absolute query/key position and relative
   distance, stratified by layer and head;
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

1. Verify/reuse `R`, `C`, and `C+R` at 200k.
2. Train `N`, `I+R`, fixed `A`, and fixed `A_post+R` at 200k, seed 123.
3. Consolidate historical static-shape/frequency/QK-coupling ablations.
4. Run checkpoint-only mechanism analyses.

This stage is mandatory. If `C+R` does not remain the best attention-local
method at mature horizon, revise the paper hierarchy but do not hide the
result.

### Stage B: robustness breadth

1. FineWeb-Edu M-scale pair (`R`, `C+R`), 200k.
2. Context-512 and context-2048 M-scale pairs, 200k.
3. Learning-rate outer-point pairs, 100k.
4. Optional S-scale pair, 200k.
5. Modern-backbone M-scale pair, 200k.

Stop expanding an axis if the paired effect reverses materially. Diagnose the
interaction before averaging incompatible settings.

### Stage C: broader paper

1. Implement and audit the recognized baseline table at canonical M scale.
2. ViT-S/16 2D transfer.
3. Replicate only the successful transferred result.
4. Consider DiT only after the ViT endpoint.

### Discovery during paper experiments

If a scale or modality suggests an improvement, treat it as a new development
hypothesis. Test it on a separate development slice/configuration, document
the revision, then rerun every affected primary comparison. Do not silently
tune against the frozen paper holdout.

## 13. Immediate run recommendation

The next GPU batch should be Stage A, not more seed replication:

1. mature NoPE;
2. mature residual-input sinusoid + RoPE;
3. mature fixed AddRoPE without RoPE;
4. mature fixed post-RoPE AddRoPE + RoPE.

All four are M-scale, seed 123, 200k-step runs and can execute in parallel.
They close the factorial and location comparisons using already completed
`R`, `C`, and `C+R` controls. Expected wall time is about 2.5 hours per GPU on
the current box, plus validation and compilation.

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
