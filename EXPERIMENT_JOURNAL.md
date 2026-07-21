# MLPRope Experiment Journal

This is the living record of experiments, implementation changes, conclusions,
and deferred research directions. Add dated entries rather than rewriting old
results when the interpretation or code changes.

## Current status

As of 2026-07-21:

- Phase-1 and Phase-1b are both complete at 10k steps.
- Best overall remains Phase-1 **linear logit bias** (4.076 / 58.9).
- Corrected low-rank and bottleneck-MLP logit biases nearly match linear.
- Q/K phase residuals help much less than logit bias; MLP phase is ~tied with RoPE.

## 2026-07-20 — Phase 1: position-only logit biases

### Setup

- Model: hidden size 768, depth 8, 8 heads.
- Batch size 8, sequence length 1024.
- 10,000 optimizer steps, 200 warmup steps.
- Dataset: OpenWebText with the GPT-2 tokenizer.
- Baseline: standard multiplicative RoPE, no learned logit bias.
- Learned variants: standard RoPE on Q/K plus a position-only logit-bias curve.

### Final results

| Variant | Eval loss | Perplexity |
| --- | ---: | ---: |
| RoPE baseline | 4.129 | 62.1 |
| AddRoPE logit bias | 4.126 | 61.9 |
| Phase-1 low-rank implementation | 4.102 | 60.5 |
| MLP logit bias | 4.095 | 60.0 |
| Linear logit bias | **4.076** | **58.9** |

### Interpretation

- Linear is not the baseline; it is the best learned Phase-1 variant.
- A learned relative-position logit correction can improve over standard RoPE.
- More nominal expressiveness did not monotonically improve loss: linear beat
  both nonlinear variants.
- AddRoPE was effectively tied with the RoPE baseline.
- Parameter matching is not yet controlled, but the added parameter counts are
  not large enough to assume parameter count is the sole explanation.

### Important confound

The Phase-1 `low_rank` name did not match its intended design:

```text
documented: D -> r -> D, purely linear
implemented: D -> r -> GELU -> scalar
```

That run remains a valid empirical result for its actual architecture, but it
cannot answer whether a factorized full linear map matches the full-rank linear
variant. The corrected implementation is now called low-rank and keeps the
feature output at `D`; the nonlinear counterpart is separately named
`bottleneck_mlp`.

The completed MLP run also predates the stricter exact-identity initialization
now in the code. Its scalar readout was zero, so the initial logit bias was still
zero, but its residual MLP output matrix used a small random initialization
rather than exact zeros.

## 2026-07-20 — Runtime and compiler findings

The first FlexAttention runs failed because the generated Triton kernels needed
122,880 bytes of shared memory while the GPU limit reported by the compiler was
101,376 bytes.

Working configuration:

- Pin FlexAttention block sizes to 32 and use two stages / four warps.
- Keep FlexAttention outside the outer compiled graph.
- Allow graph breaks (`fullgraph=false` for the outer model).
- Compile FlexAttention with `mode="default"`.

`max-autotune` enabled CUDAGraphs and caused a severe progressive slowdown:
roughly 0.8–1.0 seconds per step initially, growing to 4–6 seconds. A controlled
benchmark reproduced the growth, while `default` stayed flat. Current runs and
future launch configs therefore use default compile mode.

## 2026-07-20 — Dual-channel architecture

Position mechanisms are now factored into two independently enabled channels:

```yaml
qk:
  enabled: false
  feature_map: identity
  sharing: per_head
  apply: phase_residual

logit_bias:
  enabled: false
  feature_map: identity
  sharing: per_head
```

Each channel independently selects:

- Feature map: identity, AddRoPE affine, linear, low-rank,
  bottleneck MLP, or MLP.
- Sharing: shared across heads, independent per head, or a joint full-dimension
  map.

The Q/K channel additionally selects:

- `add`: add a zero-initialized position vector before standard RoPE.
- `phase_residual`: predict a phase delta and compose it with standard RoPE.
  A zero phase delta converts to the identity rotation.

The logit channel always ends in a zero-initialized scalar readout. Both channels
disabled is the standard RoPE baseline; both channels may be enabled together.

Legacy `pos_variant` configs remain accepted as logit-channel presets so the
completed Phase-1 configs remain interpretable.

### Verification

- CPU shape and zero-initialization tests cover all feature maps and sharing
  modes.
- Tests verify that Q/K add and phase residuals exactly match the RoPE baseline
  at initialization.
- Compiled CUDA forward/backward smoke tests passed for Q/K phase, Q/K add,
  logit-only, and both channels together.

## 2026-07-21 — Phase 1b launched

Queued four jobs (`GPU_SELECTOR=any`, `PARALLEL=true`, WandB on):

1. Corrected linear low-rank logit bias: `D -> r -> D` (`r=32`, per-head).
2. Bottleneck-MLP logit bias: `D -> r -> GELU -> D` (`r=32`, per-head).
3. Per-head linear phase residual on Q/K.
4. Per-head MLP phase residual on Q/K (`mlp_hidden=128`).

Three claimed free GPUs immediately; `qk-phase-linear` waited via `gpu-claim --wait`.
Configs: `sweep_configs/`. Logs: `logs/<job>.log`. Project: `mlprope-position-bias`.

## 2026-07-21 — Phase 1b results

Same recipe as Phase 1 (h768 d8, bs8, 10k steps). Anchors not re-run.

| Variant | Channel | Eval loss | Perplexity |
| --- | --- | ---: | ---: |
| RoPE baseline (Phase 1) | — | 4.129 | 62.1 |
| Linear logit (Phase 1 best) | logit | **4.076** | **58.9** |
| Corrected low-rank logit `r=32` | logit | 4.081 | 59.2 |
| Bottleneck-MLP logit `r=32` | logit | 4.084 | 59.4 |
| Phase-1 confounded "low_rank" | logit | 4.102 | 60.5 |
| Phase-1 MLP logit | logit | 4.095 | 60.0 |
| Linear phase residual | Q/K | 4.109 | 60.9 |
| MLP phase residual | Q/K | 4.131 | 62.2 |

### Interpretation

- **Full linear logit bias remains best**, but corrected factorized linear
  (`D→r→D`, r=32) closes almost all of the gap (−0.048 vs −0.053 vs RoPE).
- Bottleneck nonlinearity does not help over the factorized linear map
  (4.084 vs 4.081); both beat the confounded Phase-1 "low_rank" and the older MLP.
- **Logit channel >> Q/K phase residual** under this recipe. Linear phase helps
  modestly (−0.020 vs RoPE); MLP phase is effectively tied with / slightly worse
  than RoPE.
- Rank-32 is enough to approximate the useful part of the full `D→D` logit map.
- Bias diagnostics: corrected low-rank curves have large per-layer abs-max
  (up to ~59) and positive mean; bottleneck_mlp tends toward negative mean with
  smaller abs-max. Q/K-only runs do not emit logit-bias diagnostics.

### Implication for next steps

Prefer staying on the logit channel (or combining channels) rather than
pursuing richer Q/K phase maps alone. Natural follow-ups: content-conditioned
logit bias (Inkling), param-matched FFN controls, and optionally
`logit linear + qk phase linear` together.

## Intended future work

The following were deliberately out of scope for the dual-channel refactor, but
we intend to pursue them:

1. **Inkling / content-conditioned banks**
   - Query-dependent mixtures over learned relative-distance profiles.
   - Table and CosNet/function-parameterized versions.
2. **Content-conditioned Q/K residual**
   - Explore a cheap residual such as
     `x + low_rank_MLP(concat(x, rope_features))`.
3. **Parameter-matched wider-FFN controls**
   - Spend the position module's extra parameters in the baseline FFN to test
     whether gains come from mechanism or parameter count.
4. **Replacing RoPE entirely**
   - Current phase residuals are zero-initialized deltas on top of standard RoPE.
   - Later experiments should test learned phase/position mechanisms without
     the fixed RoPE prior.

Additional later analyses:

- Full channel × feature-map × sharing sweeps.
- Attention entropy and length-extrapolation evaluation.
- Content-routing diagnostics once Inkling is implemented.

## Reproducibility pointers

- Design: `position_embedding_experiments.md`
- Model: `transformer.py`
- Trainer and diagnostics: `train_gpt.py`
- Sweep launcher: `launch_position_bias.sh`
- Phase-1b configs: `sweep_configs/`
- Run outputs: `model-output/`
- WandB project: `mlprope-position-bias`
