# MLPRope repository consolidation

_Decision record, 2026-09-05. Git history is the archive for superseded code
and protocols; `results/` is the durable evidence layer._

## Scientific focus

The local design search is complete. The active question is now:

> Does a tied scalar sinusoid injected before Q/K projection provide a robust,
> transferable improvement when composed with standard fixed RoPE?

AddRoPE remains a second interesting mechanism. Frequency, phase, smooth
spectral amplitude, separate Q/K preprojection transforms, dynamic RoPE,
cumulative clocks, and EMA/scan controllers did not earn continued active
runtime support.

## Keep active

- standard fixed RoPE and NoPE;
- frozen Fourier basis utilities;
- tied-scalar pre-Q/K injection;
- static AddRoPE and its replicated pointwise content-conditioned reference;
- method-aware and legacy Q/K normalization controls;
- position-specific LR groups and optimizer/function-step diagnostics;
- fused SDPA, paired initialization, disjoint evaluation, checkpointing, and
  provenance.

The AddRoPE framework remains generic for now because simplifying it risks the
best historical mechanism and is orthogonal to the immediate evidence runs.

## Removed in this pass

- the shared learned carrier-frequency module and all frequency-specific
  clipping, optimizer groups, diagnostics, and basis overrides;
- exponential and direct smooth pre-Q/K amplitude modes;
- phase, split-Q/K, and free-pair pre-Q/K modes already retired earlier;
- phase-35/36/37 launch and preparation scripts used only by closed modes;
- their cases in the active CUDA smoke test;
- superseded root-level design briefs and protocols whose results are already
  summarized in phase reports and the experiment journal.

The active pre-Q/K implementation now contains one learnable scalar per layer.
Compatibility validators recognize historical disabled blocks and raise an
explicit error for enabled removed modes. Historical configs remain under
`sweep_configs/`; historical implementations remain recoverable by commit.

## Evidence retained

The key closed-branch results remain in:

- `results/phase33_static_qkpre_200k/`;
- `results/phase34_shared_frequency_200k/`;
- `results/phase35_smooth_carrier_20k/`;
- `results/phase36_direct_carrier_20k/`;
- `results/phase37_direct_amplitude_200k/`.

Each contains a narrative report and machine-readable analysis. The complete
chronology and design rationale remain in `EXPERIMENT_JOURNAL.md`.

## Storage cleanup

Two verified cleanup passes have been performed:

1. 2026-09-03: 50 intermediate checkpoints removed, reclaiming 87 GiB;
2. 2026-09-05: 14 redundant completed step-200k resume states plus one smoke
   checkpoint removed, reclaiming about 24 GiB.

Before the second pass, every research checkpoint's parent had a completion
marker, final standalone weights, training summary, metrics, provenance, and
21 context-1024 evaluation files. Only optimizer/scheduler/sampler/RNG resume
state at already-completed endpoints was deleted. Final weights and compact
evidence remain.

`/workspace` is not a persistent Vast volume. Final weights are a separate
retention decision and should be copied off-box before recycling the instance.

## Next cleanup boundary

Do not simplify `position/channels.py` until the next AddRoPE evidence decision.
After the new pre-Q/K robustness experiments, reassess:

- whether the pointwise content-conditioned AddRoPE reference still merits
  active support;
- whether historical v1 config upgrades and presets can be retired;
- whether completed phase-specific analyzers are better kept in-tree or
  recoverable only from git.

No model artifact should be deleted as a side effect of code cleanup.
