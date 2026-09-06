# Attention-local sinusoidal carrier paper

This directory contains the working paper and its frozen experimental plan.

- `main.tex`: paper scaffold, method equations, current evidence, and planned
  analyses.
- `EXPERIMENT_PLAN.md`: candidate set, ablations, architecture, scales, token
  budgets, generalization axes, statistics, and execution order.
- `references.bib`: bibliography.
- `figs/` and `tables/`: generated paper artifacts when available.

Build with:

```bash
cd paper
latexmk -pdf -interaction=nonstopmode main.tex
```

The checked-in style is currently the ICLR 2025 template and is only a working
layout. Update it after choosing a target venue.

The utility scripts in this directory came from an older paper scaffold and
must be audited against this repository's JSONL result format before use. The
source-of-truth evidence currently lives in `../results/` and the resolved run
artifacts in `../model-output/`.
