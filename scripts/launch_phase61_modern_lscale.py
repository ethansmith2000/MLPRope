#!/usr/bin/env python
"""Run Phase-61 preflights or main jobs through gpu-claim, cap two."""

from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "sweep_configs" / "phase61_modern_lscale"
LOG_ROOT = ROOT / "logs" / "phase61_modern_lscale"
RESULT_ROOT = ROOT / "results" / "phase61_modern_lscale"
OUTPUT_ROOT = ROOT / "model-output" / "position_bias_phase61_modern_lscale"
HARD_CONCURRENCY_LIMIT = 2
EXPECTED_CONFIGS = 2


def _config(path: Path) -> dict:
    return json.loads(path.read_text())


def _complete(path: Path) -> bool:
    config = _config(path)
    marker = Path(config["output_dir"]) / "COMPLETED"
    if not marker.is_file():
        return False
    try:
        payload = json.loads(marker.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return int(payload.get("completed_steps", -1)) == int(config["max_train_steps"])


def _finite_jsonl(path: Path) -> bool:
    if not path.is_file():
        return False
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    return bool(rows) and all(
        math.isfinite(float(value))
        for row in rows
        for value in row.values()
        if isinstance(value, (int, float))
    )


def _finite_logs(path: Path) -> bool:
    config = _config(path)
    output = Path(config["output_dir"])
    if not _finite_jsonl(output / "metrics.jsonl"):
        return False
    pre = config.get("qk_preprojection", {})
    return not (pre.get("enabled") and pre.get("learnable_gate")) or _finite_jsonl(
        output / "intervention_optimization.jsonl"
    )


def _endpoint(config: dict) -> Path:
    return (
        Path(config["output_dir"])
        / "evaluation_details"
        / f"step_{int(config['max_train_steps']):08d}_context_001024.json"
    )


def _directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _cleanup_recovery(path: Path) -> None:
    config = _config(path)
    run_dir = Path(config["output_dir"]).resolve()
    if run_dir.parent != OUTPUT_ROOT.resolve() or run_dir.is_symlink():
        raise RuntimeError(f"Unsafe Phase-61 run directory: {run_dir}")
    if not _complete(path) or not _endpoint(config).is_file():
        raise RuntimeError(f"Refusing cleanup before verified completion: {run_dir}")
    removed = []
    reclaimed = 0
    for checkpoint in sorted(run_dir.glob("step_*")):
        if checkpoint.is_symlink() or checkpoint.resolve().parent != run_dir:
            raise RuntimeError(f"Unsafe checkpoint target: {checkpoint}")
        if not (checkpoint / "CHECKPOINT_COMPLETE.json").is_file():
            raise RuntimeError(f"Refusing to remove incomplete checkpoint: {checkpoint}")
        size = _directory_bytes(checkpoint)
        shutil.rmtree(checkpoint)
        reclaimed += size
        removed.append({"path": str(checkpoint.relative_to(ROOT)), "bytes": size})
    event = {
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "run_name": config["run_name"],
        "reason": "verified completion and final evaluation; checkpoints were recovery-only",
        "recoverable": False,
        "removed": removed,
        "reclaimed_bytes": reclaimed,
    }
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    with (RESULT_ROOT / "checkpoint_cleanup_events.jsonl").open("a") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")
    print(
        f"cleanup {config['run_name']}: {len(removed)} checkpoints, "
        f"{reclaimed / 2**30:.2f} GiB",
        flush=True,
    )


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
        raise SystemExit(f"Expected {EXPECTED_CONFIGS} configs, found {len(configs)}")
    if not args.preflight:
        for path in configs:
            if _complete(path):
                _cleanup_recovery(path)
    pending = deque(path for path in configs if not _complete(path))
    processes: dict[subprocess.Popen, tuple[Path, str, object]] = {}
    failures = []
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    prefix = "preflight-" if args.preflight else ""

    def start_next() -> None:
        path = pending.popleft()
        run_name = _config(path)["run_name"]
        command = [
            claimer, "run", "--owner", "mlprope", "--job", run_name,
            "--gpu", args.gpu, "--wait", "--", "/venv/main/bin/python", "-u",
            "train_gpt.py", "--override_json", str(path),
        ]
        handle = (LOG_ROOT / f"{prefix}{run_name}.log").open("a")
        handle.write(f"\n=== start {time.time():.6f} {json.dumps(command)} ===\n")
        handle.flush()
        process = subprocess.Popen(
            command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT
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
                elif not _finite_logs(path):
                    failures.append((run_name, 1, "missing or non-finite logs"))
                elif not args.preflight and not _endpoint(_config(path)).is_file():
                    failures.append((run_name, 1, "missing final evaluation details"))
                elif not args.preflight:
                    try:
                        _cleanup_recovery(path)
                    except Exception as error:
                        failures.append((run_name, 1, f"checkpoint cleanup: {error}"))
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
