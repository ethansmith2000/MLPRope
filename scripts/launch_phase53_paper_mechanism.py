#!/usr/bin/env python
"""Run nine retained-checkpoint mechanism jobs through gpu-claim, cap two."""

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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_phase53_paper_mechanism import ARMS, RESULT_ROOT, SEEDS


LOG_ROOT = ROOT / "logs" / "phase53_paper_mechanism"
HARD_CONCURRENCY_LIMIT = 2


def _complete(seed: int, arm: str) -> bool:
    stem = RESULT_ROOT / f"seed{seed}_{arm}"
    json_path = stem.with_suffix(".json")
    npz_path = stem.with_suffix(".npz")
    if not json_path.is_file() or not npz_path.is_file():
        return False
    try:
        payload = json.loads(json_path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return (
        payload.get("seed") == seed
        and payload.get("arm") == arm
        and payload.get("holdout") == {"start_batch": 4_096, "blocks": 1_024}
        and payload.get("attention_sample", {}).get("blocks") == 64
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--max-concurrent", type=int, default=2)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute complete outputs, for analysis-code audit corrections.",
    )
    args = parser.parse_args()
    if not 1 <= args.max_concurrent <= HARD_CONCURRENCY_LIMIT:
        raise SystemExit("max-concurrent must be 1 or 2")
    claimer = shutil.which("gpu-claim")
    if claimer is None:
        raise SystemExit("gpu-claim is required; see /workspace/GPU_QUEUEING.md")
    pending = deque(
        (seed, arm)
        for seed in SEEDS
        for arm in ARMS
        if args.overwrite or not _complete(seed, arm)
    )
    for seed in SEEDS:
        for arm in ARMS:
            print(
                f"{'pending ' if args.overwrite or not _complete(seed, arm) else 'complete':8s} "
                f"seed={seed} arm={arm}",
                flush=True,
            )
    processes: dict[subprocess.Popen, tuple[int, str, object]] = {}
    failures = []
    LOG_ROOT.mkdir(parents=True, exist_ok=True)

    def start_next() -> None:
        seed, arm = pending.popleft()
        job = f"phase53-mechanism-seed{seed}-{arm}"
        command = [
            claimer, "run", "--owner", "mlprope", "--job", job,
            "--gpu", args.gpu, "--wait", "--", "/venv/main/bin/python", "-u",
            "scripts/evaluate_phase53_paper_mechanism.py", "--seed", str(seed),
            "--arm", arm,
        ]
        if args.overwrite:
            command.append("--overwrite")
        handle = (LOG_ROOT / f"seed{seed}_{arm}.log").open("a")
        handle.write(
            f"\n=== launcher_start unix={time.time():.6f} "
            f"command={json.dumps(command)} ===\n"
        )
        handle.flush()
        process = subprocess.Popen(
            command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT
        )
        processes[process] = (seed, arm, handle)
        print(f"queued    seed={seed} arm={arm} pid={process.pid}", flush=True)

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
                print(
                    f"finished  seed={seed} arm={arm} rc={return_code}", flush=True
                )
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
    for seed, arm, return_code in failures:
        print(f"FAILED seed={seed} arm={arm} rc={return_code}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
