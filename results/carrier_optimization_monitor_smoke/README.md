# Carrier optimization monitor smoke

_Validated 2026-09-04 on one RTX 5090 through `gpu-claim`._

A 12-step h768/d8 compiled-bf16 run exercised a horizon-normalized pre-Q/K
carrier frequency bank with standard fixed RoPE. The resolved config contained
no learned-RoPE field. Optimizer samples were written at steps 1, 2, 4, 8, and
12; every numeric value was finite.

| Step | frequency grad L2 | parameter update L2 | endpoint phase max step | carrier function RMS step | descent/update-gradient cosine |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.002986 | 0 | 0 | 0 | n/a |
| 2 | 0.001506 | 0.002108 | 0.000153 | 0.0000438 | 0.612 |
| 4 | 0.000838 | 0.002761 | 0.000269 | 0.0000577 | 0.469 |
| 8 | 0.000277 | 0.000966 | 0.000125 | 0.0000207 | 0.229 |
| 12 | 0.000221 | 0.000139 | 0.000061 | 0.0000038 | 0.235 |

The zero step at step 1 is expected: the learning-rate warmup begins at zero.
The later decline in current-gradient/update alignment illustrates information
that is not visible in gradient magnitude alone. Static adapter carrier
movement was also nonzero from step 2 onward.

The consolidated CUDA matrix also passed all eleven retained backbone/carrier
combinations in both eager and compiled bf16 forward/backward, including
AddRoPE and pre-Q/K carriers with either backbone, and pre-Q/K + AddRoPE +
standard RoPE. A separate compiled CUDA forward/backward with gradient
checkpointing also passed.

Command:

```bash
gpu-claim run --owner mlprope --job carrier-optimization-monitor-smoke --wait -- \
  /venv/main/bin/python -u train_gpt.py \
  --override_json sweep_configs/carrier_optimization_smoke/carrier-horizon-s12.json

gpu-claim run --owner mlprope --job carrier-backbone-cuda-matrix --wait -- \
  /venv/main/bin/python -u scripts/position_v2_cuda_smoke.py
```

The disposable full output remains under
`model-output/carrier_optimization_smoke/carrier-horizon-s12/`; this compact
report is the durable evidence.
