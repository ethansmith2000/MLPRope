#!/usr/bin/env python
"""Evaluate the three retained Phase-49 references through gpu-claim."""

from __future__ import annotations

import argparse
from collections import deque
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results" / "phase52_bias_anchor_closeout"
LOG_ROOT = ROOT / "logs" / "phase52_bias_anchor_closeout"
ARMS = ("rope", "scalar-qkpre", "qk-readout-r32")
HARD_CONCURRENCY_LIMIT = 2


def _complete(arm: str) -> bool:
    path = RESULT_ROOT / f"reference_{arm}.json"
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return (
        payload.get("arm") == arm
        and payload.get("evaluation_start_batch") == 5_120
        and payload.get("evaluation_blocks") == 1_024
        and len(payload.get("losses", ())) == 1_024
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--max-concurrent", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()
    if not 1 <= args.max_concurrent <= HARD_CONCURRENCY_LIMIT:
        raise SystemExit("max-concurrent must be 1 or 2")
    claimer = shutil.which("gpu-claim")
    if claimer is None:
        raise SystemExit("gpu-claim is required; see /workspace/GPU_QUEUEING.md")
    pending = deque(arm for arm in ARMS if not _complete(arm))
    for arm in ARMS:
        print(f"{'complete' if _complete(arm) else 'pending ':8s} {arm}", flush=True)
    processes: dict[subprocess.Popen, tuple[str, object]] = {}
    failures = []
    LOG_ROOT.mkdir(parents=True, exist_ok=True)

    def start_next() -> None:
        arm = pending.popleft()
        command = [
            claimer, "run", "--owner", "mlprope", "--job", f"phase52-reference-{arm}",
            "--gpu", args.gpu, "--wait", "--", "/venv/main/bin/python", "-u",
            "scripts/evaluate_phase52_references.py", "--arm", arm,
            "--batch-size", str(args.batch_size),
        ]
        handle = (LOG_ROOT / f"reference_{arm}.log").open("a")
        handle.write(
            f"\n=== launcher_start unix={time.time():.6f} "
            f"command={json.dumps(command)} ===\n"
        )
        handle.flush()
        process = subprocess.Popen(
            command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT
        )
        processes[process] = (arm, handle)
        print(f"queued    {arm} pid={process.pid}", flush=True)

    try:
        while pending or processes:
            while pending and len(processes) < args.max_concurrent and not failures:
                start_next()
            for process, (arm, handle) in list(processes.items()):
                return_code = process.poll()
                if return_code is None:
                    continue
                handle.close()
                processes.pop(process)
                print(f"finished  {arm} rc={return_code}", flush=True)
                if return_code:
                    failures.append((arm, return_code))
            if failures:
                for process in processes:
                    process.terminate()
                for process, (_, handle) in list(processes.items()):
                    process.wait()
                    handle.close()
                processes.clear()
                break
            if pending or processes:
                time.sleep(5)
    finally:
        for _, handle in processes.values():
            handle.close()
    for arm, return_code in failures:
        print(f"FAILED {arm} rc={return_code}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
