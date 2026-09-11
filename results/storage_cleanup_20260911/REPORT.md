# Storage cleanup — 2026-09-11

## Reason and pre-cleanup state

- Workspace filesystem: 1.1 TiB total, 962 GiB used, 70 GiB available (94%).
- `model-output/`: 147 GiB.
- Completed-run resume checkpoints duplicated weights already present as
  standalone `pytorch_model.bin` files and additionally retained optimizer,
  scheduler, sampler, and RNG state.

## Authorized removal

Remove the 52 exact completed-run `step_*` directories listed in
`removed_resume_checkpoints.txt`. Their measured total size before deletion
was 99,875,665,697 bytes (93.02 GiB).

Every target passed all of these checks before deletion:

1. its parent run contained a valid `COMPLETED` marker;
2. its parent retained a standalone `pytorch_model.bin` final weight file;
3. it was not a checkpoint of a currently running GPU job;
4. the target was an exact `model-output/.../step_*` directory.

The deletion removes local exact-resume capability for those completed runs.
It preserves final inference/evaluation weights and all compact evidence:
configs, seeds, provenance, metrics, evaluation details, logs, summaries, and
completion markers. The operation is not locally reversible because the
workspace is not a versioned artifact store.

## Protected at cleanup time

The active Phase-50 checkpoints below were explicitly excluded:

- `phase50-scalar-ffnmatch-r32-seed456.../step_85000`
- `phase50-rope-seed456.../step_80000`

Phase 50's already-running retention policy was not changed. Its completed
resume checkpoints may be removed under the same validation rule after the
cohort and analyzer finish.

## Deliberately retained final weights

No standalone final weight was deleted in this pass. In particular, the
following weights have a stated near-term diagnostic purpose:

- Phase-42 RoPE seed 123: baseline checkpoint analyses;
- Phase-42 scalar pre-Q/K seed 123: scalar-carrier analyses;
- Phase-49 rank-32 seed 123: carrier counterfactuals and decomposition;
- Phase-50 rank-32 seed 456, and seed 789 once complete: mechanism replication.

Other standalone final weights will be reviewed only after the checkpoint
interventions establish which comparisons still require model access.

## Future artifact policy

- Default trainer behavior remains `checkpointing_steps=null` and
  `save_final_model=false`.
- If checkpointing is explicitly enabled without a retention override, the
  trainer now retains only the newest complete recovery state.
- A long confirmatory run may keep one rolling recovery checkpoint only when
  interruption recovery is explicitly worth its storage cost; remove it after
  successful completion and analysis.
- Save a final model only for a named downstream evaluation, intervention, or
  selected-state comparison. Record that purpose in the experiment protocol.
- Preserve compact evidence even when all weights are removed.

## Completion

All 52 manifest entries were revalidated immediately before deletion, including
the exact aggregate byte count above. They were then removed successfully; zero
manifest targets remained. Post-cleanup, `model-output/` was 54 GiB and the
workspace had 162 GiB available (85% used).

The two active Phase-50 jobs had advanced from their earlier 80k/85k snapshot to
fresh 85k rolling checkpoints by the post-cleanup check. Those live checkpoints
remain present inside the Phase-50 experiment root and were never manifest
targets.
