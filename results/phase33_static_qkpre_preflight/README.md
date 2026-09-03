# Phase 33 static pre-Q/K preflight

All six frozen h768/d8, context-1024 arms completed 50 optimizer steps through
`gpu-claim` on RTX 5090 GPUs. The source was clean commit `6aa859d`; every run
recorded its resolved config, parameter counts, package/CUDA/GPU state, dataset
fingerprints, and both canonical cache manifests.

| Arm | tokens/s | vs RoPE | reserved MiB | parameters | Q/K-pre params |
| --- | ---: | ---: | ---: | ---: | ---: |
| fixed RoPE | 190,783 | — | 5,076 | 153,495,936 | 0 |
| tied, no RoPE | 190,790 | +0.00% | 5,220 | 153,495,944 | 8 |
| tied + RoPE | 189,838 | -0.50% | 5,220 | 153,495,944 | 8 |
| split scalar + RoPE | 189,161 | -0.85% | 5,300 | 153,495,952 | 16 |
| pair amplitude + RoPE | 186,771 | -2.10% | 5,340 | 153,502,080 | 6,144 |
| pair amplitude+phase + RoPE | 185,639 | -2.70% | 5,360 | 153,508,224 | 12,288 |

Throughput excludes the first 20 optimizer steps, leaving a 30-step measured
window. At the six-arm mean, 200k optimizer steps correspond to about 2.41
compute hours per arm before compilation, validation, and checkpoint I/O. The
short window is sufficient for launch sizing, not an exact iso-wall-clock
claim.

The real Accelerate checkpoint format was also exercised end to end. Saves at
steps 1, 2, and 3 left only marked step 3; a new process then loaded the model,
optimizer, scheduler, sampler, and RNG states and resumed at step 3.

All arms emitted the same PyTorch compile-time warning that bf16 LayerNorm
inputs and fp32 weights could not use one fused implementation. Compilation
and training still completed, throughput remained high, and the warning is
not mechanism-specific. It is therefore logged as a possible future kernel
optimization rather than a blocker for the paired screen.

The step-50 losses are health checks only. They are too early and use only four
validation examples, so they are not evidence for ranking the mechanisms.
Machine-readable values are in `preflight_summary.json`.
