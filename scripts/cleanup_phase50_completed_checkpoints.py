#!/usr/bin/env python
"""Remove Phase-50 recovery states only after the full analysis completes."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PHASE_ROOT = (
    ROOT / "model-output" / "position_bias_phase50_training_seed_replication"
)
RESULT_ROOT = ROOT / "results" / "storage_cleanup_20260911"
PHASE_REPORT = ROOT / "results" / "phase50_training_seed_replication" / "REPORT.md"


def _active_phase50_trainers() -> list[int]:
    active = []
    for process_dir in Path("/proc").glob("[0-9]*"):
        try:
            command = (process_dir / "cmdline").read_bytes().replace(b"\0", b" ")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if (
            b"train_gpt.py" in command
            and b"phase50_training_seed_replication" in command
        ):
            active.append(int(process_dir.name))
    return active


def main() -> None:
    if not PHASE_REPORT.is_file():
        raise RuntimeError(f"Phase-50 analysis report is missing: {PHASE_REPORT}")
    active = _active_phase50_trainers()
    if active:
        raise RuntimeError(f"Refusing cleanup with active Phase-50 trainers: {active}")

    runs = sorted(path for path in PHASE_ROOT.iterdir() if path.is_dir())
    if len(runs) != 8:
        raise RuntimeError(f"Expected exactly 8 Phase-50 run directories, got {len(runs)}")
    targets = []
    for run_dir in runs:
        completed_path = run_dir / "COMPLETED"
        final_weights = run_dir / "pytorch_model.bin"
        if not completed_path.is_file() or not final_weights.is_file():
            raise RuntimeError(f"Incomplete evidence parent: {run_dir}")
        completed = json.loads(completed_path.read_text())
        if int(completed.get("completed_steps", -1)) != 100_000:
            raise RuntimeError(f"Unexpected completion marker: {completed_path}")
        for checkpoint in sorted(run_dir.glob("step_*")):
            resolved = checkpoint.resolve()
            if checkpoint.is_symlink() or resolved.parent != run_dir.resolve():
                raise RuntimeError(f"Unsafe checkpoint target: {checkpoint}")
            marker = checkpoint / "CHECKPOINT_COMPLETE.json"
            if not marker.is_file():
                raise RuntimeError(f"Checkpoint lacks completion marker: {checkpoint}")
            size = sum(
                entry.stat().st_size
                for entry in checkpoint.rglob("*")
                if entry.is_file()
            )
            targets.append((checkpoint, size))

    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    manifest = RESULT_ROOT / "phase50_removed_resume_checkpoints.txt"
    manifest.write_text(
        "".join(f"{path.relative_to(ROOT)}\n" for path, _ in targets)
    )
    total = sum(size for _, size in targets)
    for path, _ in targets:
        shutil.rmtree(path)
    remaining = [str(path) for path, _ in targets if path.exists()]
    if remaining:
        raise RuntimeError(f"Cleanup targets remain: {remaining}")

    report = [
        "# Phase-50 recovery-checkpoint cleanup",
        "",
        f"Completed at {datetime.now(timezone.utc).isoformat()}.",
        "",
        f"Removed {len(targets)} exact completed recovery directories totaling ",
        f"{total:,} bytes after the Phase-50 report was written and no Phase-50 ",
        "trainer processes remained.",
        "",
        "Every parent retains its standalone final model, completion marker, config,",
        "provenance, metrics, evaluation details, applicable optimization log, and summary.",
        "The rank-32 final weights for seeds 456 and 789 remain declared inputs to",
        "Phase 51. No standalone weight was removed.",
        "",
    ]
    (RESULT_ROOT / "PHASE50_COMPLETION.md").write_text("\n".join(report))
    print("\n".join(report))


if __name__ == "__main__":
    main()
