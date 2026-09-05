# MLPRope current status

_Authoritative as of 2026-09-05. Older mechanisms and protocols are preserved
in git history; compact experimental evidence remains under `results/`._

## Bottom line

Two attention-local sinusoidal mechanisms remain scientifically interesting:

1. **AddRoPE:** an additive Fourier carrier on projected Q/K;
2. **pre-Q/K sinusoid:** add one tied, gated sinusoid to the inputs of the Q
   and K projections, then apply standard RoPE.

The clearest current result is the second method. At h768/d8 and 200k steps,
pre-Q/K + RoPE beat its paired fixed-RoPE baseline by `-0.062831` validation
loss. The carrier alone also helped, but standard RoPE contributed another
`-0.030773`, so the method is complementary to rather than a replacement for
RoPE.

The carrier should remain simple. Separate Q/K gains, per-pair amplitude,
phase, smooth spectral amplitude, and globally shared learned frequencies did
not improve the scalar anchor at mature horizon. Content-dependent RoPE,
cumulative clocks, and EMA/linear-RNN controllers are also closed.

## Strongest completed evidence

| Result | Protocol | Finding |
| --- | --- | ---: |
| AddRoPE amplitude 1.0 vs fixed RoPE | 30k, 3 paired seeds | `-0.076867` mean |
| AddRoPE amplitude 1.0 vs 0.3 | 30k, 3 paired seeds | `-0.014895` mean |
| pre-Q/K + RoPE vs fixed RoPE | 30k, 3 paired seeds | `-0.065235` mean |
| pre-Q/K + RoPE vs fixed RoPE | 200k, 1 paired seed | `-0.062831` |
| pre-Q/K + RoPE vs pre-Q/K + NoPE | 200k, 1 paired seed | `-0.030773` |
| split/pair amplitude/pair phase ladder | 200k, 1 paired seed | all within about `0.001` |
| shared log-frequency carrier vs fixed | 200k, 1 paired seed | `+0.000861`, null |
| horizon-frequency carrier vs fixed | 200k, 1 paired seed | `+0.001341`, worse |
| direct smooth amplitude vs scalar | 200k, 1 paired seed | `+0.000111`, null |
| exponential smooth amplitude vs scalar | 200k, 1 paired seed | `-0.000363`, null |
| pointwise content AddRoPE vs position-only | 30k, 3 paired seeds | `-0.010812` mean |
| AddRoPE scalar EMA vs pointwise | 15k, 1 paired seed | `-0.010626` step-matched; about `-0.0015` iso-wall-clock |

At 15k, AddRoPE and pre-Q/K were strongly sub-additive: their combination was
`+0.004934` worse than AddRoPE alone. This is evidence of overlapping function,
but the comparison is too early to support a mature exclusivity claim.

## Baseline architecture

The main h768/d8 model has approximately 153.4M parameters and uses:

- decoder-only causal self-attention with fused PyTorch SDPA;
- eight 96-dimensional heads;
- pre-norm residual blocks with LayerNorm;
- separate bias-free Q/K/V projections and a biased output projection;
- GeGLU feed-forwards at four times model width;
- per-head Q/K normalization;
- standard fixed split-half RoPE at context 1024;
- tied paired initialization and fixed data order for comparisons;
- AdamW with linear scheduling and bf16 autocast.

The promoted carrier uses method-aware Q/K RMSNorm: content and position are
combined before `W_q/W_k`, then each projected head is normalized once. This
controls the projected mixture, but it creates an important evidence gap:
robustness to disabling QKNorm has not yet been established.

## Why the closed refinements are genuinely closed

Phase 37 directly paired scalar, exponential smooth amplitude, and signed
direct smooth amplitude for 200k steps. The primary disjoint 1,024-example
holdout was null, even though both shape maps moved substantially and had
finite gradients, Adam states, updates, and carrier-function movement. This
rules out an obvious inactive-path explanation for their failure.

Phase 34 similarly showed that horizon-normalized frequency coordinates remove
the dangerous raw `p` multiplier from the endpoint derivative, but still do
not improve modeling loss. Faster direct frequency coordinates eventually
violated spectral ordering. The negative frequency result is therefore not
well explained by the one optimization pathology we originally identified.

## Evidence limitation

With batch 8 and sequence length 1024, each step consumes 8,192 tokens. The
h768/d8 model sees:

| Steps | Tokens | Tokens / parameter |
| ---: | ---: | ---: |
| 30k | 245.8M | 1.60 |
| 100k | 819.2M | 5.34 |
| 200k | 1.638B | 10.68 |

The three-seed result is reproducible but short; the mature result is only one
seed and is still below a conventional compute-optimal token budget. There is
also no architecture, scale, modality, or longer-context transfer result yet.
Those are now more valuable than another carrier-shape sweep.

## Active implementation

The runtime keeps:

- standard fixed RoPE and NoPE;
- the tied-scalar pre-Q/K carrier, initialized at gate 1.0;
- static AddRoPE and the pointwise content-conditioned AddRoPE reference;
- generic positional LR control and optimizer/function-step diagnostics;
- paired evaluation, provenance, resumable checkpoints, and fused SDPA.

It no longer implements learned carrier frequency, pre-Q/K smooth amplitude,
separate Q/K preprojection transforms, dynamic RoPE, clocks, EMA, residual
position writes, or attention-output writes. Enabled archived configurations
fail explicitly; disabled archived blocks canonicalize to an inert active form.

## Next evidence program

The next experiments should test the method, not search its local shape space:

1. **mature replication:** additional paired 200k seeds for fixed RoPE versus
   scalar pre-Q/K + RoPE;
2. **architecture robustness:** a paired QKNorm/normalization ablation;
3. **scale transfer:** the same fixed two-arm comparison at another model size;
4. **mechanism:** length-stratified loss plus attention entropy, attended
   distance, and position-correlation diagnostics from trained checkpoints.

The first three should use identical data order within each pair and disjoint
1,024-example final holdouts. No refinement arm is admitted unless a distinct,
predeclared hypothesis emerges.

## Repository and storage state

- Compact phase reports and analysis JSON remain in `results/`; complete
  historical configs remain in `sweep_configs/`.
- Historical source and deleted protocols remain recoverable from git.
- On 2026-09-05, 14 redundant completed endpoint checkpoints plus one smoke
  checkpoint were deleted after verification, reclaiming about 24 GiB. Final
  model weights, evaluations, metrics, configs, and provenance remain.
- `/workspace` is not a persistent Vast volume. Irreplaceable weights must be
  copied off-box before instance recycle or destruction.
