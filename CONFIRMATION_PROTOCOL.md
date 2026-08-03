# Phase-19 paired confirmation protocol

_Status: locked before launch. Any change after a result exists must be recorded as a protocol revision, not silently regenerated._

## Question

Does the phase-18 learned absolute additive Q/K result survive paired base initialization, multiple seeds, a disjoint large holdout, simpler additive and parameter-matched controls, and an unprofiled training loop?

## Fixed design

- Model: h1024/d12, 8 heads, sequence length 1024.
- Training: 30,000 steps, batch 8, seed-specific shuffled data, linear schedule, lr `4e-4`, AdamW betas `0.95/0.999`.
- Seeds: 123, 456, 789. Each seed is shared by every arm for both data order and `paired_initialization_seed`.
- Development evaluation: first 25 validation batches every 5,000 steps.
- Final confirmation evaluation: 1,024 batch-size-one examples beginning at validation batch 2,048, disjoint from development evaluation. This is 1,047,552 causal targets per arm.
- Final weights and per-example final losses are saved.
- W&B and per-ten-step CUDA profiling are disabled. Local artifacts are authoritative.

## Arms

1. Standard RoPE.
2. Mapped AddRoPE, amplitude anchor 0.3: the simpler additive control from phase 9.
3. Position-only carrier hypernetwork: the phase-18 candidate.
4. Content+position carrier hypernetwork: tests the claimed negligible content contribution.
5. RoPE plus a close usable-capacity control: FFNs widened from 4096 to 4160 in 9 evenly distributed layers. This adds 1,770,624 parameters, 3.3% above position-only's 1,714,176.

## Locked primary analysis

For each contrast, compute the per-seed difference in final holdout loss, then report its mean, all three individual differences, and a confidence interval from the paired per-example losses within each seed. Do not substitute the development result if it looks better.

Primary contrasts, in order:

1. Position-only minus standard RoPE: confirmation of the headline effect.
2. Position-only minus mapped AddRoPE: incremental value of the hypernetwork over simple additive Q/K.
3. Position-only minus matched-FFN RoPE: positional mechanism versus comparable useful capacity.
4. Content+position minus position-only: incremental value of content conditioning.

Training throughput is secondary. It will be read from structured summaries, but an isolated same-GPU microbenchmark remains necessary before making an exact iso-wallclock claim.

## Decision rule

Call position-only confirmed only if it beats standard RoPE in all three seeds and the mean advantage remains practically material on the final holdout. Call the hypernetwork-specific mechanism supported only if it also consistently beats mapped AddRoPE. Differences below `0.01` are reported as unresolved unless the paired intervals and seed consistency are unusually decisive.
