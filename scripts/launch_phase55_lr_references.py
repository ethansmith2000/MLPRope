#!/usr/bin/env python
"""Launch the two retained Phase-55 reference evaluations through gpu-claim."""

from __future__ import annotations

import subprocess
from pathlib import Path
import shutil
import sys


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results" / "phase55_lr_robustness"
LOG_ROOT = ROOT / "logs" / "phase55_lr_robustness"
ARMS = ("rope", "scalar-qkpre")


def main() -> int:
    claimer = shutil.which("gpu-claim")
    if claimer is None:
        raise SystemExit("gpu-claim is required; see /workspace/GPU_QUEUEING.md")
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    processes = []
    for arm in ARMS:
        output = RESULT_ROOT / f"reference_lr3e4_{arm}.json"
        if output.is_file():
            continue
        command = [
            claimer, "run", "--owner", "mlprope", "--job", f"phase55-ref-{arm}",
            "--gpu", "0,1,2,3,4,5,6,7", "--wait", "--",
            "/venv/main/bin/python", "-u", "scripts/evaluate_phase55_lr_references.py",
            "--arm", arm,
        ]
        handle = (LOG_ROOT / f"reference_{arm}.log").open("a")
        processes.append((subprocess.Popen(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT), arm, handle))
    failed = []
    for process, arm, handle in processes:
        code = process.wait()
        handle.close()
        print(f"reference {arm} rc={code}", flush=True)
        if code:
            failed.append((arm, code))
    if failed:
        for arm, code in failed:
            print(f"FAILED reference {arm} rc={code}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
