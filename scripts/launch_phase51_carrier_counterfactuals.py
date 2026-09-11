#!/usr/bin/env python
"""Run three checkpoint counterfactual evaluations through gpu-claim."""

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
RESULT_ROOT = ROOT / "results" / "phase51_carrier_mechanism"
HARD_CONCURRENCY_LIMIT = 2


def _complete(seed: int) -> bool:
    path = RESULT_ROOT / f"counterfactual_seed{seed}.json"
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return (
        payload.get("seed") == seed
        and set(payload.get("interventions", {}))
        == {
            "full",
            "direct_mean_only",
            "direct_mean_removed",
            "direct_zero",
            "scalar_zero",
            "all_zero",
        }
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--max-concurrent", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()
    if not 1 <= args.max_concurrent <= HARD_CONCURRENCY_LIMIT:
        raise SystemExit(
            f"max-concurrent must be between 1 and {HARD_CONCURRENCY_LIMIT}"
        )
    claimer = shutil.which("gpu-claim")
    if claimer is None:
        raise SystemExit("gpu-claim is required; see /workspace/GPU_QUEUEING.md")

    pending = deque(seed for seed in (123, 456, 789) if not _complete(seed))
    for seed in (123, 456, 789):
        print(f"{'complete' if _complete(seed) else 'pending ':8s} seed {seed}", flush=True)
    processes: dict[subprocess.Popen, tuple[int, object]] = {}
    failures = []
    log_dir = ROOT / "logs" / "phase51_carrier_mechanism"
    log_dir.mkdir(parents=True, exist_ok=True)

    def start_next() -> None:
        seed = pending.popleft()
        command = [
            claimer,
            "run",
            "--owner",
            "mlprope",
            "--job",
            f"phase51-carrier-counterfactual-seed{seed}",
            "--gpu",
            args.gpu,
            "--wait",
            "--",
            "/venv/main/bin/python",
            "-u",
            "scripts/evaluate_phase51_carrier_counterfactuals.py",
            "--seed",
            str(seed),
            "--batch-size",
            str(args.batch_size),
        ]
        handle = (log_dir / f"counterfactual_seed{seed}.log").open("a")
        handle.write(
            f"\n=== launcher_start unix={time.time():.6f} "
            f"command={json.dumps(command)} ===\n"
        )
        handle.flush()
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
        processes[process] = (seed, handle)
        print(f"queued seed {seed} pid={process.pid}", flush=True)

    try:
        while pending or processes:
            while pending and len(processes) < args.max_concurrent and not failures:
                start_next()
            for process, (seed, handle) in list(processes.items()):
                return_code = process.poll()
                if return_code is None:
                    continue
                handle.close()
                processes.pop(process)
                print(f"finished seed {seed} rc={return_code}", flush=True)
                if return_code != 0:
                    failures.append((seed, return_code))
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
    except BaseException:
        for process in processes:
            process.terminate()
        raise
    finally:
        for _, handle in processes.values():
            handle.close()

    if failures:
        for seed, return_code in failures:
            print(f"FAILED seed {seed} rc={return_code}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
