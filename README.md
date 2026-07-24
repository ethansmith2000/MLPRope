# MLPRope

- [Experiment journal](EXPERIMENT_JOURNAL.md)
- [Position embedding design](position_embedding_experiments.md)
- [Position playground configuration](POSITION_CONFIG.md)

CPU verification:

```bash
/venv/main/bin/python -m unittest \
  test_position_channels test_position_playground -v
```

GPU verification uses the shared claim protocol:

```bash
gpu-claim run --owner mlprope --job position-playground-smoke --wait -- \
  /venv/main/bin/python -u scripts/position_v2_cuda_smoke.py
```