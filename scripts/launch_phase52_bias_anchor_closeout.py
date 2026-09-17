#!/usr/bin/env python
"""Run Phase 52 preflights and main arms through gpu-claim, cap two."""

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
CONFIG_ROOT = ROOT / "sweep_configs" / "phase52_bias_anchor_closeout"
LOG_ROOT = ROOT / "logs" / "phase52_bias_anchor_closeout"
HARD_CONCURRENCY_LIMIT = 2


def _completed(config_path: Path) -> bool:
    config = json.loads(config_path.read_text())
    marker = Path(config["output_dir"]) / "COMPLETED"
    if not marker.is_file():
        return False
    try:
        payload = json.loads(marker.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return int(payload.get("completed_steps", -1)) >= int(config["max_train_steps"])


def _run_stage(
    configs: list[Path], *, claimer: str, gpu: str, max_concurrent: int, stage: str
) -> int:
    pending = deque()
    for path in configs:
        config = json.loads(path.read_text())
        state = "complete" if _completed(path) else "pending"
        print(f"{stage:9s} {state:8s} {config['run_name']}", flush=True)
        if state == "pending":
            pending.append((path, config["run_name"]))
    processes: dict[subprocess.Popen, tuple[str, object]] = {}
    failures: list[tuple[str, int]] = []
    LOG_ROOT.mkdir(parents=True, exist_ok=True)

    def start_next() -> None:
        path, run_name = pending.popleft()
        command = [
            claimer, "run", "--owner", "mlprope", "--job", run_name,
            "--gpu", gpu, "--wait", "--", "/venv/main/bin/python", "-u",
            "train_gpt.py", "--override_json", str(path),
        ]
        handle = (LOG_ROOT / f"{run_name}.log").open("a")
        handle.write(
            f"\n=== launcher_start unix={time.time():.6f} "
            f"command={json.dumps(command)} ===\n"
        )
        handle.flush()
        process = subprocess.Popen(
            command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT
        )
        processes[process] = (run_name, handle)
        print(f"queued    {run_name} pid={process.pid}", flush=True)

    try:
        while pending or processes:
            while pending and len(processes) < max_concurrent and not failures:
                start_next()
            for process, (run_name, handle) in list(processes.items()):
                return_code = process.poll()
                if return_code is None:
                    continue
                handle.close()
                processes.pop(process)
                print(f"finished  {run_name} rc={return_code}", flush=True)
                if return_code:
                    failures.append((run_name, return_code))
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
    for run_name, return_code in failures:
        print(f"FAILED {run_name} rc={return_code}", file=sys.stderr)
    return 1 if failures else 0


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
    stages = (
        ("preflight", sorted((CONFIG_ROOT / "preflight").glob("*.json"))),
        ("main", sorted(CONFIG_ROOT.glob("*.json"))),
    )
    if any(len(configs) != 2 for _, configs in stages):
        raise SystemExit("Expected exactly two preflight and two main configs")
    for stage, configs in stages:
        result = _run_stage(
            configs,
            claimer=claimer,
            gpu=args.gpu,
            max_concurrent=args.max_concurrent,
            stage=stage,
        )
        if result:
            return result
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
