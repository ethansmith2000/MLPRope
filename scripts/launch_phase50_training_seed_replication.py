#!/usr/bin/env python
"""Run Phase 50 through gpu-claim with at most two live jobs."""

from __future__ import annotations

import argparse
from collections import deque
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path


REPO_DIR = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_DIR / "sweep_configs" / "phase50_training_seed_replication"
LOG_DIR = REPO_DIR / "logs" / "phase50_training_seed_replication"
EXPECTED_RUNS = 8
HARD_CONCURRENCY_LIMIT = 2


def _completed(config_path: Path) -> bool:
    config = json.loads(config_path.read_text())
    marker = Path(config["output_dir"]) / "COMPLETED"
    if not marker.is_file():
        return False
    try:
        completed = json.loads(marker.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return int(completed.get("completed_steps", -1)) >= int(
        config["max_train_steps"]
    )


def _command(*, claimer: str, config_path: Path, run_name: str, gpu: str) -> list[str]:
    return [
        claimer,
        "run",
        "--owner",
        "mlprope",
        "--job",
        run_name,
        "--gpu",
        gpu,
        "--wait",
        "--",
        "/venv/main/bin/python",
        "-u",
        "train_gpt.py",
        "--override_json",
        str(config_path),
    ]


def run(*, gpu: str, max_concurrent: int) -> int:
    if not 1 <= max_concurrent <= HARD_CONCURRENCY_LIMIT:
        raise SystemExit(
            f"max_concurrent must be between 1 and {HARD_CONCURRENCY_LIMIT}"
        )
    claimer = shutil.which("gpu-claim")
    if claimer is None:
        raise SystemExit("gpu-claim is required; see /workspace/GPU_QUEUEING.md")
    configs = sorted(CONFIG_DIR.glob("*.json"))
    if len(configs) != EXPECTED_RUNS:
        raise SystemExit(
            f"Expected {EXPECTED_RUNS} configs in {CONFIG_DIR}, found {len(configs)}"
        )

    pending = deque()
    for config_path in configs:
        config = json.loads(config_path.read_text())
        state = "complete" if _completed(config_path) else "pending"
        print(f"{state:8s} {config['run_name']}", flush=True)
        if state == "pending":
            pending.append((config_path, config["run_name"]))
    if not pending:
        return 0

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    processes: dict[subprocess.Popen, tuple[str, object]] = {}
    failures: list[tuple[str, int]] = []

    def start_next() -> None:
        config_path, run_name = pending.popleft()
        command = _command(
            claimer=claimer,
            config_path=config_path,
            run_name=run_name,
            gpu=gpu,
        )
        log_handle = (LOG_DIR / f"{run_name}.log").open("a")
        log_handle.write(
            f"\n=== launcher_start unix={time.time():.6f} "
            f"command={json.dumps(command)} ===\n"
        )
        log_handle.flush()
        process = subprocess.Popen(
            command,
            cwd=REPO_DIR,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
        processes[process] = (run_name, log_handle)
        print(
            f"queued   {run_name} pid={process.pid} "
            f"active_slots={len(processes)}/{max_concurrent}",
            flush=True,
        )

    try:
        while pending or processes:
            while pending and len(processes) < max_concurrent and not failures:
                start_next()
            for process, (run_name, log_handle) in list(processes.items()):
                return_code = process.poll()
                if return_code is None:
                    continue
                log_handle.close()
                processes.pop(process)
                print(f"finished {run_name} rc={return_code}", flush=True)
                if return_code != 0:
                    failures.append((run_name, return_code))
            if failures:
                for process in processes:
                    process.terminate()
                for process, (_, log_handle) in list(processes.items()):
                    process.wait()
                    log_handle.close()
                processes.clear()
                break
            if pending or processes:
                time.sleep(5)
    except BaseException:
        for process in processes:
            process.terminate()
        raise
    finally:
        for _, log_handle in processes.values():
            log_handle.close()

    if failures:
        for run_name, return_code in failures:
            print(f"FAILED {run_name} rc={return_code}", file=sys.stderr)
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--max-concurrent", type=int, default=2)
    args = parser.parse_args()
    return run(gpu=args.gpu, max_concurrent=args.max_concurrent)


if __name__ == "__main__":
    raise SystemExit(main())
