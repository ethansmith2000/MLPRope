# MLPRope

Research code for attention-local sinusoidal position mechanisms. The active
runtime is intentionally narrow: standard fixed RoPE or NoPE, AddRoPE, and a
tied scalar sinusoid injected before the Q/K projections.

- [Current evidence and decisions](CURRENT_STATUS.md)
- [Active architectural policy](SINUSOID_INTERVENTION_POLICY.md)
- [Next evidence roadmap](NEXT_EXPERIMENT_ROADMAP.md)
- [Position configuration](POSITION_CONFIG.md)
- [Repository consolidation record](CONSOLIDATION_PLAN.md)
- [Literature review](LITERATURE_REVIEW.md)
- [Chronological experiment journal](EXPERIMENT_JOURNAL.md)
- [Compact phase reports](results/)

Historical implementations and superseded protocols remain recoverable from
git history; completed experiment configs remain under `sweep_configs/`.

CPU verification:

```bash
/venv/main/bin/python -m unittest \
  test_position_channels test_position_dynamics \
  test_position_playground test_position_results -v
```

GPU verification uses the shared claim protocol:

```bash
gpu-claim run --owner mlprope --job position-playground-smoke --wait -- \
  /venv/main/bin/python -u scripts/position_v2_cuda_smoke.py
```
