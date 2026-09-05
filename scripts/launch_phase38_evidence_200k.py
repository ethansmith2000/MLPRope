#!/usr/bin/env python
"""Queue every incomplete Phase-38 run through the shared GPU claimer."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path


REPO_DIR = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_DIR / "sweep_configs" / "phase38_evidence_200k"
LOG_DIR = REPO_DIR / "logs" / "phase38_evidence_200k"


def _target(path: Path) -> tuple[str, Path, int]:
    payload = json.loads(path.read_text())
    return (
        str(payload["run_name"]),
        Path(payload["output_dir"]),
        int(payload["max_train_steps"]),
    )


def _completed(output_dir: Path, target_steps: int) -> bool:
    marker = output_dir / "COMPLETED"
    if not marker.is_file():
        return False
    try:
        payload = json.loads(marker.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return int(payload.get("completed_steps", -1)) >= target_steps


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--owner", default="mlprope")
    parser.add_argument("--gpu", default=None)
    parser.add_argument("--status-only", action="store_true")
    args = parser.parse_args()

    claimer = shutil.which("gpu-claim")
    if claimer is None:
        raise SystemExit("gpu-claim is required; see /workspace/GPU_QUEUEING.md")
    configs = sorted(CONFIG_DIR.glob("*.json"))
    if len(configs) != 8:
        raise SystemExit(f"Expected eight Phase-38 configs, found {len(configs)}")

    pending = []
    for config_path in configs:
        run_name, output_dir, steps = _target(config_path)
        state = "complete" if _completed(output_dir, steps) else "pending"
        print(f"{state:8s} {run_name}")
        if state == "pending":
            pending.append((config_path, run_name))
    if args.status_only or not pending:
        return 0

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    processes = {}
    logs = {}
    for config_path, run_name in pending:
        command = [
            claimer,
            "run",
            "--owner",
            args.owner,
            "--job",
            run_name,
        ]
        if args.gpu:
            command.extend(("--gpu", args.gpu))
        command.extend(
            (
                "--wait",
                "--",
                "/venv/main/bin/python",
                "-u",
                "train_gpt.py",
                "--override_json",
                str(config_path),
            )
        )
        log_path = LOG_DIR / f"{run_name}.log"
        log_handle = log_path.open("a")
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
        processes[process] = run_name
        logs[process] = log_handle
        print(f"queued   {run_name} pid={process.pid}", flush=True)

    failures = []
    try:
        while processes:
            for process, run_name in list(processes.items()):
                return_code = process.poll()
                if return_code is None:
                    continue
                logs.pop(process).close()
                processes.pop(process)
                print(f"finished {run_name} rc={return_code}", flush=True)
                if return_code != 0:
                    failures.append((run_name, return_code))
            if processes:
                time.sleep(5)
    except BaseException:
        for process in processes:
            process.terminate()
        for process in processes:
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
        raise
    finally:
        for handle in logs.values():
            handle.close()

    if failures:
        for run_name, return_code in failures:
            print(f"FAILED {run_name} rc={return_code}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
