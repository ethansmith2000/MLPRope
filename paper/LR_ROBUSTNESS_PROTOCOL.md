# Phase 55 symmetric learning-rate robustness protocol

Status: frozen before training on 2026-09-13. Launch is conditional on the
Phase-54 small-origin-offset rule.

## Question

Is the scalar pre-Q/K carrier's advantage over RoPE an artifact of evaluating
both methods only at the shared peak AdamW learning rate `3e-4`?

## Frozen matrix

Train exactly four new seed-123 runs at the canonical h768/d8, batch-32,
context-1024, 100k-update OpenWebText recipe:

| Peak LR | RoPE | Scalar pre-Q/K + RoPE |
|---:|---|---|
| `1.5e-4` | new | new |
| `3.0e-4` | retained Phase-49 checkpoint | retained Phase-49 checkpoint |
| `6.0e-4` | new | new |

All other optimizer, initialization, data-order, model, scheduler, warmup, and
evaluation settings remain fixed. The scalar gate remains directly optimized,
initialized at one, and uses no method-specific LR multiplier.

The final endpoint is the previously unused validation window `[6144, 7167]`.
Re-evaluate both retained `3e-4` checkpoints on the same window. Development
evaluations remain monitoring diagnostics and do not choose the endpoint.

## Reporting and decisions

- Report every arm/LR NLL, throughput, peak memory, and finite-training check.
- At each LR report paired scalar-minus-RoPE deltas with IID and contiguous
  block-32 intervals.
- Report best-RoPE versus best-scalar across the three-point grid. This is a
  descriptive robustness comparison, not a claim that three values fully tune
  either method.
- Strong robustness: scalar-minus-RoPE is negative at both new matched LRs and
  best scalar beats best RoPE.
- A reversal at either outer LR weakens robustness but does not erase the
  replicated canonical result. A non-finite arm is an optimization failure and
  is reported rather than silently replaced by another LR.

Training seed is not replicated here; the experiment audits hyperparameter
sensitivity around an already replicated central setting.

## Artifact policy

Each four-hour run may keep one rolling recovery checkpoint at 10k intervals,
solely to survive interruption. The launcher resumes only from a
completion-marked checkpoint. After successful final evaluation and analysis,
the exact recovery directories are deleted and the cleanup is recorded. No
final model weights or milestone checkpoints are saved; compact configs,
provenance, metrics, optimizer diagnostics, and per-block final losses remain.
