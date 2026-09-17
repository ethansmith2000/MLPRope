#!/usr/bin/env python
"""Run Phase-54 checkpoint audits through gpu-claim, at most two at once."""

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
RESULT_ROOT = ROOT / "results" / "phase54_position_origin"
LOG_ROOT = ROOT / "logs" / "phase54_position_origin"
ARMS = ("scalar-qkpre", "qk-readout-r32")
SEEDS = (123, 456, 789)
HARD_CONCURRENCY_LIMIT = 2


def _complete(seed: int, arm: str) -> bool:
    stem = RESULT_ROOT / f"seed{seed}_{arm}"
    return stem.with_suffix(".json").is_file() and stem.with_suffix(".npz").is_file()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--max-concurrent", type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.max_concurrent <= HARD_CONCURRENCY_LIMIT:
        raise SystemExit("max-concurrent must be 1 or 2")
    claimer = shutil.which("gpu-claim")
    if claimer is None:
        raise SystemExit("gpu-claim is required; see /workspace/GPU_QUEUEING.md")

    pending = deque(
        (seed, arm) for seed in SEEDS for arm in ARMS if not _complete(seed, arm)
    )
    processes: dict[subprocess.Popen, tuple[int, str, object]] = {}
    failures = []
    LOG_ROOT.mkdir(parents=True, exist_ok=True)

    def start_next() -> None:
        seed, arm = pending.popleft()
        job = f"phase54-origin-seed{seed}-{arm}"
        command = [
            claimer, "run", "--owner", "mlprope", "--job", job,
            "--gpu", args.gpu, "--wait", "--", "/venv/main/bin/python", "-u",
            "scripts/evaluate_phase54_position_origin.py",
            "--seed", str(seed), "--arm", arm,
        ]
        handle = (LOG_ROOT / f"seed{seed}_{arm}.log").open("a")
        handle.write(f"\n=== start {time.time():.6f} {json.dumps(command)} ===\n")
        handle.flush()
        process = subprocess.Popen(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
        processes[process] = (seed, arm, handle)
        print(f"queued seed={seed} arm={arm} pid={process.pid}", flush=True)

    try:
        while pending or processes:
            while pending and len(processes) < args.max_concurrent and not failures:
                start_next()
            for process, (seed, arm, handle) in list(processes.items()):
                return_code = process.poll()
                if return_code is None:
                    continue
                handle.close()
                processes.pop(process)
                print(f"finished seed={seed} arm={arm} rc={return_code}", flush=True)
                if return_code:
                    failures.append((seed, arm, return_code))
            if failures:
                for process in processes:
                    process.terminate()
                for process, (_, _, handle) in list(processes.items()):
                    process.wait()
                    handle.close()
                processes.clear()
                break
            if pending or processes:
                time.sleep(5)
    finally:
        for _, _, handle in processes.values():
            handle.close()
    if failures:
        for seed, arm, code in failures:
            print(f"FAILED seed={seed} arm={arm} rc={code}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
