# Phase 41: batch throughput and memory benchmark

All probes use the canonical h768/d8 RoPE model, context 1024, bf16, compiled
SDPA, one RTX 5090, and 120 optimizer steps. Throughput and peak memory exclude
the first 20 optimizer steps. No paper holdout was used for model selection.

| Sequence batch | Tokens/update | Target tokens/s | Peak allocated | Peak reserved |
|---:|---:|---:|---:|---:|
| 8 | 8,192 | 187,843 | 5,066 MiB | 5,076 MiB |
| 16 | 16,384 | 207,515 | 8,136 MiB | 8,544 MiB |
| 32 | 32,768 | 215,031 | 14,277 MiB | 16,258 MiB |
| 64 | 65,536 | 221,831 | 26,558 MiB | 26,638 MiB |

Batch 32 is the selected paper operating point. Relative to batch 8 it raises
measured target-token throughput by 14.5% while retaining roughly half of the
device memory as headroom. Batch 64 adds only 3.2% throughput over batch 32 but
uses 86% more allocated memory, leaving too little margin for compilation,
minor architecture changes, or co-resident variability.

The selection is operational, not a claim that batch 32 is optimization-
optimal. The paper cohort therefore uses matched batch-32 controls and
candidates. Existing batch-8 models remain a separately labeled robustness
cohort and are never treated as matched controls for batch-32 runs.
