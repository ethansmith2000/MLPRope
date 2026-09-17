# Attention-local sinusoidal carrier paper

This directory contains the working paper and its frozen experimental plan.

- `main.tex`: paper scaffold, method equations, current evidence, and planned
  analyses.
- `EXPERIMENT_PLAN.md`: candidate set, ablations, architecture, scales, token
  budgets, generalization axes, statistics, and execution order.
- `EVIDENCE_AUDIT.md`: claim-by-claim map from completed experiments to
  defensible wording and remaining requirements.
- `MECHANISM_PROTOCOL.md`: frozen Phase-53 checkpoint-analysis protocol.
- `LR_ROBUSTNESS_PROTOCOL.md`: completed Phase-55 symmetric LR audit.
- `LR_BOUNDARY_PROTOCOL.md`: frozen Phase-56 one-point LR close-out.
- `POSITIONAL_BASELINE_PROTOCOL.md`: frozen Phase-57 recognized positional
  baseline comparison.
- `references.bib`: bibliography.
- `../results/phase53_paper_mechanism/`: machine-readable mechanism results,
  compact arrays, CSV exports, and rendered report.
- `../results/phase55_lr_robustness/`: completed LR-robustness report and
  paired per-block evidence.
- `../results/phase56_lr_boundary/`: completed one-point LR close-out and
  frozen prospective recipe decision.

Build with:

```bash
cd paper
latexmk -pdf -interaction=nonstopmode main.tex
```

The checked-in style is currently the ICLR 2025 template and is only a working
layout. Update it after choosing a target venue.

The legacy plotting/collection utilities in this directory are not sources of
truth for this manuscript and must be audited before reuse. Current evidence
lives in `../results/`; resolved run provenance and the retained weights needed
for the completed checkpoint analyses live in `../model-output/`.
