#!/usr/bin/env python
"""Run Phase-58 preflights or main jobs through gpu-claim, cap two."""

from __future__ import annotations

import argparse
from collections import deque
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "sweep_configs" / "phase58_gate_initialization"
LOG_ROOT = ROOT / "logs" / "phase58_gate_initialization"
HARD_CONCURRENCY_LIMIT = 2
EXPECTED_CONFIGS = 4


def _config(path: Path) -> dict:
    return json.loads(path.read_text())


def _complete(config_path: Path) -> bool:
    config = _config(config_path)
    marker = Path(config["output_dir"]) / "COMPLETED"
    if not marker.is_file():
        return False
    try:
        payload = json.loads(marker.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return int(payload.get("completed_steps", -1)) == int(config["max_train_steps"])


def _finite_metrics(config_path: Path) -> bool:
    metrics = Path(_config(config_path)["output_dir"]) / "metrics.jsonl"
    if not metrics.is_file():
        return False
    for line in metrics.read_text().splitlines():
        if not line:
            continue
        row = json.loads(line)
        for value in row.values():
            if isinstance(value, (int, float)) and not math.isfinite(float(value)):
                return False
    return True


def _endpoint_exists(config_path: Path, *, preflight: bool) -> bool:
    if preflight:
        return True
    config = _config(config_path)
    detail = (
        Path(config["output_dir"])
        / "evaluation_details"
        / f"step_{int(config['max_train_steps']):08d}_context_001024.json"
    )
    return detail.is_file()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--max-concurrent", type=int, default=2)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.max_concurrent <= HARD_CONCURRENCY_LIMIT:
        raise SystemExit("max-concurrent must be 1 or 2")
    claimer = shutil.which("gpu-claim")
    if claimer is None:
        raise SystemExit("gpu-claim is required; see /workspace/GPU_QUEUEING.md")

    config_dir = CONFIG_ROOT / "preflight" if args.preflight else CONFIG_ROOT
    configs = sorted(config_dir.glob("*.json"))
    if len(configs) != EXPECTED_CONFIGS:
        raise SystemExit(
            f"Expected {EXPECTED_CONFIGS} configs in {config_dir}, found {len(configs)}"
        )
    pending = deque(path for path in configs if not _complete(path))
    processes: dict[subprocess.Popen, tuple[Path, str, object]] = {}
    failures = []
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    log_prefix = "preflight-" if args.preflight else ""

    def start_next() -> None:
        path = pending.popleft()
        run_name = _config(path)["run_name"]
        command = [
            claimer,
            "run",
            "--owner",
            "mlprope",
            "--job",
            run_name,
            "--gpu",
            args.gpu,
            "--wait",
            "--",
            "/venv/main/bin/python",
            "-u",
            "train_gpt.py",
            "--override_json",
            str(path),
        ]
        handle = (LOG_ROOT / f"{log_prefix}{run_name}.log").open("a")
        handle.write(f"\n=== start {time.time():.6f} {json.dumps(command)} ===\n")
        handle.flush()
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
        processes[process] = (path, run_name, handle)
        print(f"queued {run_name} pid={process.pid}", flush=True)

    try:
        while pending or processes:
            while pending and len(processes) < args.max_concurrent and not failures:
                start_next()
            for process, (path, run_name, handle) in list(processes.items()):
                code = process.poll()
                if code is None:
                    continue
                handle.close()
                processes.pop(process)
                print(f"finished {run_name} rc={code}", flush=True)
                if code:
                    failures.append((run_name, code, "training"))
                elif not _complete(path):
                    failures.append((run_name, 1, "missing exact completion marker"))
                elif not _finite_metrics(path):
                    failures.append((run_name, 1, "missing or non-finite metrics"))
                elif not _endpoint_exists(path, preflight=args.preflight):
                    failures.append((run_name, 1, "missing final evaluation details"))
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
        for run_name, code, stage in failures:
            print(f"FAILED {run_name} rc={code} stage={stage}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
