# Phase 57 protocol: recognized positional baselines

Status: frozen before GPU preflight or outcome inspection on 2026-09-15.

## Purpose

Phase 57 closes the missing canonical comparison table. It asks whether the
primary scalar pre-Q/K sinusoidal carrier plus standard RoPE remains competitive
with familiar static positional designs when every arm is trained from scratch
under the same selected optimizer recipe.

This is a comparison study, not another search over carrier designs. No Phase 57
result will trigger an in-cohort change to an initialization, partial-RoPE
fraction, ALiBi slope, or learning rate.

## Frozen training recipe

- OpenWebText canonical token cache and GPT-2 tokenizer;
- decoder width 768, depth 8, eight heads, GeGLU FFN multiplier 4;
- pre-attention LayerNorm and the repository's method-aware per-head Q/K RMS
  normalization;
- context 1,024, sequence batch 32, one GPU per run;
- 100,000 optimizer steps (3.276B target tokens);
- paired training and initialization seed 123;
- AdamW, peak learning rate `1.2e-3`, betas `(0.9, 0.98)`, weight decay
  `0.01`, 200-step warmup, linear decay, global gradient clipping at `1.0`;
- bf16 training and compiled model execution;
- 128-block development evaluation from validation block 0;
- one final 1,024-block evaluation on the previously uninspected validation
  window `[7168, 8191]`.

The `1.2e-3` learning rate was selected prospectively in Phase 56 using only
RoPE development NLL from the bounded `6e-4`/`1.2e-3` comparison. The final
window above was explicitly left uninspected by that phase.

## Frozen arms

| ID | Implementation | Trainable positional parameters |
|---|---|---:|
| `R` | standard full-width RoPE, `theta=10000` | 0 |
| `C+R` | one learned scalar per layer multiplying a fixed model-width sinusoid before both Q/K projections, followed by standard RoPE | 8 |
| `N` | no explicit position mechanism | 0 |
| `S` | fixed model-width sinusoid added once after the input projection; no RoPE | 0 |
| `L` | learned `[1024, 768]` absolute table added once after the input projection; normal initialization with standard deviation `0.02`; no RoPE | 786,432 |
| `P25` | RoPE on the leading 25% of every head (24 of 96 dimensions), leaving the other dimensions unrotated | 0 |
| `A` | ALiBi with the standard fixed eight-head geometric slope schedule and no RoPE | 0 |

`S` is the repository's controlled classic additive-sinusoid baseline. It does
not reproduce every detail of the original Transformer, because this decoder
has an input projection and no separate embedding-scale factor. `P25` is a
precisely specified partial-RoPE comparator, not a claim of exact reproduction
of every architecture called p-RoPE. `A` uses FlexAttention because ALiBi needs
an additive score bias; its measured throughput must be reported rather than
treated as if it used the fused SDPA path.

All other positional mechanisms are disabled in every arm. In particular,
there is no learned frequency, phase, per-frequency amplitude, dynamic mapper,
Q/K projection bias, AddRoPE channel, or residual carrier beyond the arm named
above.

## Endpoints and analysis

The primary reported endpoint is mean token NLL at step 100,000 on the frozen
1,024-block final window. Report for every arm:

1. mean NLL;
2. paired per-block NLL difference versus standard RoPE and versus `C+R`;
3. IID bootstrap and contiguous-block-32 bootstrap 95% intervals for those
   differences;
4. total and positional parameter counts;
5. target-token throughput, elapsed wall time, and peak CUDA allocation and
   reservation;
6. whether all recorded numeric training metrics remained finite.

The scalar-versus-RoPE contrast is also a fresh confirmation of the primary
method at the prospectively selected learning rate. Training seed count is one,
so bootstrap intervals quantify this held-out stream, not training-seed
variation. Results are descriptive for the other baselines and will not be
called exact external reproductions.

## Preflight and artifact policy

Every arm first receives a 100-step preflight with four validation blocks.
Preflight acceptance requires exit code zero, a complete marker, finite logged
metrics, and the configured endpoint. ALiBi must additionally execute its
actual FlexAttention score-modification path on GPU.

Main runs use one rolling recovery checkpoint every 10,000 steps with at most
one retained checkpoint per active run. With the hard two-run concurrency cap,
expected maximum recovery storage is roughly two checkpoints. A checkpoint's
only purpose is interruption recovery. Once a run has its exact completion
marker and final evaluation-detail file, the launcher removes that run's
recovery state immediately and records the exact path and reclaimed bytes.
No final model weights are saved. Configs, provenance, metrics, summaries,
evaluation losses, analysis products, and cleanup manifests are retained.

All GPU work must run through `gpu-claim` with owner `mlprope` and hard
concurrency two.

