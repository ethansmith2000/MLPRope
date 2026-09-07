#!/usr/bin/env python
"""Run the Phase-41 batch-throughput probes through the shared GPU claimer."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path


REPO_DIR = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_DIR / "sweep_configs" / "phase41_batch_benchmark"
LOG_DIR = REPO_DIR / "logs" / "phase41_batch_benchmark"


def main() -> int:
    claimer = shutil.which("gpu-claim")
    if claimer is None:
        raise SystemExit("gpu-claim is required; see /workspace/GPU_QUEUEING.md")
    configs = sorted(CONFIG_DIR.glob("b*.json"))
    if len(configs) != 4:
        raise SystemExit(f"Expected four batch benchmark configs, found {len(configs)}")

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    processes: dict[subprocess.Popen, tuple[str, object]] = {}
    for config_path in configs:
        payload = json.loads(config_path.read_text())
        run_name = payload["run_name"]
        command = [
            claimer,
            "run",
            "--owner",
            "mlprope",
            "--job",
            run_name,
            "--gpu",
            "1,2,3,4,5,6,7",
            "--wait",
            "--",
            "/venv/main/bin/python",
            "-u",
            "train_gpt.py",
            "--override_json",
            str(config_path),
        ]
        log_handle = (LOG_DIR / f"{run_name}.log").open("w")
        process = subprocess.Popen(
            command,
            cwd=REPO_DIR,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
        processes[process] = (run_name, log_handle)
        print(f"queued {run_name} pid={process.pid}", flush=True)

    failures = []
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
            time.sleep(2)

    if failures:
        for run_name, return_code in failures:
            print(f"FAILED {run_name} rc={return_code}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
