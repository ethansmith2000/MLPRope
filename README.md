# MLPRope

- [Current status and decisions](CURRENT_STATUS.md)
- [Active sinusoidal intervention policy](SINUSOID_INTERVENTION_POLICY.md)
- [Phase-35 smooth carrier protocol and conclusion](SMOOTH_CARRIER_PLAN.md)
- [Phase-35 paired results](results/phase35_smooth_carrier_20k/PHASE35_RESULTS.md)
- [Phase-36 direct amplitude/frequency results](results/phase36_direct_carrier_20k/PHASE36_RESULTS.md)
- [Phase-37 direct-amplitude confirmation plan](DIRECT_AMPLITUDE_CONFIRMATION_PLAN.md)
- [Next experiment roadmap](NEXT_EXPERIMENT_ROADMAP.md)
- [Active research and repository consolidation plan](CONSOLIDATION_PLAN.md)
- [Globally shared frequency experiment](SHARED_FREQUENCY_PLAN.md)
- [Experiment journal](EXPERIMENT_JOURNAL.md)
- [Position embedding design](position_embedding_experiments.md)
- [Position playground configuration](POSITION_CONFIG.md)
- [Phase-24 basis screen](results/phase24_rope_embed_basis/PHASE24_RESULTS.md)

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
