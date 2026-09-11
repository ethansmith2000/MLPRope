#!/usr/bin/env python
"""Run Phase 48 through the shared lifetime GPU claimer."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path


REPO_DIR = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_DIR / "sweep_configs" / "phase48_empirical_rank32"
LOG_DIR = REPO_DIR / "logs" / "phase48_empirical_rank32"


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
    if len(configs) != 1:
        raise SystemExit(f"Expected 1 config in {config_dir}, found {len(configs)}")

    config_path = configs[0]
    config = json.loads(config_path.read_text())
    run_name = config["run_name"]
    if _completed(config_path):
        print(f"complete {run_name}")
        return 0

    log_dir = LOG_DIR / ("preflight" if preflight else "main")
    log_dir.mkdir(parents=True, exist_ok=True)
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
    with (log_dir / f"{run_name}.log").open("a") as log_handle:
        log_handle.write(
            f"\n=== launcher_start unix={time.time():.6f} "
            f"command={json.dumps(command)} ===\n"
        )
        log_handle.flush()
        return subprocess.run(
            command,
            cwd=REPO_DIR,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            check=False,
        ).returncode


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
