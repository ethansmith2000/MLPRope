#!/usr/bin/env python
"""Run Phase 46 through the shared lifetime GPU claimer."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path


REPO_DIR = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_DIR / "sweep_configs" / "phase46_linear_rank128"
LOG_DIR = REPO_DIR / "logs" / "phase46_linear_rank128"


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


def _run(*, preflight: bool, owner: str, gpu: str) -> int:
    claimer = shutil.which("gpu-claim")
    if claimer is None:
        raise SystemExit("gpu-claim is required; see /workspace/GPU_QUEUEING.md")
    config_dir = CONFIG_DIR / "preflight" if preflight else CONFIG_DIR
    configs = sorted(config_dir.glob("*.json"))
    if len(configs) != 2:
        raise SystemExit(f"Expected 2 configs in {config_dir}, found {len(configs)}")

    pending = []
    for config_path in configs:
        config = json.loads(config_path.read_text())
        state = "complete" if _completed(config_path) else "pending"
        print(f"{state:8s} {config['run_name']}")
        if state == "pending":
            pending.append((config_path, config["run_name"]))
    if not pending:
        return 0

    log_dir = LOG_DIR / ("preflight" if preflight else "main")
    log_dir.mkdir(parents=True, exist_ok=True)
    processes: dict[subprocess.Popen, tuple[str, object]] = {}
    for config_path, run_name in pending:
        command = [
            claimer,
            "run",
            "--owner",
            owner,
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
        log_handle = (log_dir / f"{run_name}.log").open("a")
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
        print(f"queued   {run_name} pid={process.pid}", flush=True)

    failures = []
    try:
        while processes:
            for process, (run_name, log_handle) in list(processes.items()):
                return_code = process.poll()
                if return_code is None:
                    continue
                log_handle.close()
                processes.pop(process)
                print(f"finished {run_name} rc={return_code}", flush=True)
                if return_code != 0:
                    failures.append((run_name, return_code))
            if processes:
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
    parser.add_argument("--preflight-first", action="store_true")
    parser.add_argument("--owner", default="mlprope")
    parser.add_argument("--gpu", default="0,1,2,3,4,5,6,7")
    args = parser.parse_args()
    if args.preflight_first:
        result = _run(preflight=True, owner=args.owner, gpu=args.gpu)
        if result != 0:
            return result
    return _run(preflight=False, owner=args.owner, gpu=args.gpu)


if __name__ == "__main__":
    raise SystemExit(main())
