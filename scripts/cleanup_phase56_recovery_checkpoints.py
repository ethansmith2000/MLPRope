#!/usr/bin/env python
"""Delete only completed Phase-56 rolling recovery states after analysis."""

from __future__ import annotations

import json
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "sweep_configs" / "phase56_lr_boundary"
RESULT_ROOT = ROOT / "results" / "phase56_lr_boundary"


def main() -> None:
    if not (RESULT_ROOT / "summary.json").is_file():
        raise RuntimeError("Refusing cleanup before Phase-56 analysis completes")
    removed = []
    reclaimed = 0
    for config_path in sorted(CONFIG_ROOT.glob("*.json")):
        config = json.loads(config_path.read_text())
        run_dir = Path(config["output_dir"]).resolve()
        marker = json.loads((run_dir / "COMPLETED").read_text())
        if int(marker.get("completed_steps", -1)) != int(config["max_train_steps"]):
            raise RuntimeError(f"Incomplete Phase-56 run: {run_dir}")
        detail = run_dir / "evaluation_details" / "step_00100000_context_001024.json"
        if not detail.is_file():
            raise RuntimeError(f"Missing final evaluation: {detail}")
        for checkpoint in sorted(run_dir.glob("step_*")):
            resolved = checkpoint.resolve()
            if checkpoint.is_symlink() or resolved.parent != run_dir:
                raise RuntimeError(f"Unsafe checkpoint target: {checkpoint}")
            complete = checkpoint / "CHECKPOINT_COMPLETE.json"
            if not complete.is_file():
                raise RuntimeError(f"Refusing to remove incomplete checkpoint: {checkpoint}")
            size = sum(path.stat().st_size for path in checkpoint.rglob("*") if path.is_file())
            shutil.rmtree(checkpoint)
            reclaimed += size
            removed.append({"path": str(checkpoint.relative_to(ROOT)), "bytes": size})
    payload = {
        "scope": "phase56_completed_recovery_cleanup",
        "removed": removed,
        "reclaimed_bytes": reclaimed,
        "recoverable": False,
        "reason": "Both Phase-56 evaluations and aggregate analysis completed; rolling states had recovery-only purpose.",
    }
    (RESULT_ROOT / "checkpoint_cleanup.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    print(f"Removed {len(removed)} completed recovery checkpoints; reclaimed {reclaimed / 2**30:.2f} GiB")


if __name__ == "__main__":
    main()
