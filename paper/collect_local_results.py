#!/usr/bin/env python3
"""Collect local paper-run metrics into machine-readable summary files."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOTS = (
    REPO_ROOT / "model-output" / "paper_runs",
    REPO_ROOT / "model-output" / "focused",
    REPO_ROOT / "model-output" / "diffusion_paper",
    REPO_ROOT / "model-output" / "imagenet_paper",
    # Keep the historical ImageNet location readable during migration.
    REPO_ROOT / "imagenet_output" / "paper_runs",
)


def _number(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _read_metrics(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    value = json.loads(path.read_text())
    return value if isinstance(value, dict) else {}


def _last(rows: list[dict[str, str]], key: str) -> float | None:
    for row in reversed(rows):
        value = _number(row.get(key))
        if value is not None:
            return value
    return None


def _median_tail(rows: list[dict[str, str]], key: str, fraction: float = 0.8) -> float | None:
    values = [value for row in rows if (value := _number(row.get(key))) is not None]
    if not values:
        return None
    start = min(len(values) - 1, int(len(values) * (1.0 - fraction)))
    return statistics.median(values[start:])


def _first_number(*values: Any) -> float | None:
    for value in values:
        number = _number(value)
        if number is not None:
            return number
    return None


def summarize_run(run_dir: Path, domain: str) -> dict[str, Any]:
    rows = _read_metrics(run_dir / "metrics.csv")
    result = _read_json(run_dir / "all_results.json")
    summary = _read_json(run_dir / "run_summary.json")
    config = _read_json(run_dir / "config.json")
    recipe = summary.get("recipe_metadata") or config.get("recipe_metadata") or {}
    perplexity = _number(result.get("perplexity"))
    eval_loss = _last(rows, "eval_loss")
    if eval_loss is None and perplexity is not None and perplexity > 0:
        eval_loss = math.log(perplexity)

    return {
        "domain": domain,
        "run": run_dir.name,
        "path": str(run_dir),
        "final_step": _last(rows, "step"),
        "eval_loss": eval_loss,
        "perplexity": perplexity,
        "top1_accuracy": _first_number(
            _last(rows, "val/top1"), summary.get("final_top1")
        ),
        "top5_accuracy": _first_number(
            _last(rows, "val/top5"), summary.get("final_top5")
        ),
        "ema_top1_accuracy": _first_number(
            _last(rows, "ema_val/top1"), summary.get("final_ema_top1")
        ),
        "ema_top5_accuracy": _first_number(
            _last(rows, "ema_val/top5"), summary.get("final_ema_top5")
        ),
        "fid_50k": _last(rows, "eval/fid_50k"),
        "inception_score": _last(rows, "eval/inception_score"),
        "wall_time_seconds": _first_number(
            _last(rows, "wall_time_seconds"), summary.get("elapsed_time_seconds")
        ),
        "tokens_per_second": _median_tail(rows, "throughput/tokens_per_second"),
        "examples_per_second": _first_number(
            _median_tail(rows, "throughput/examples_per_second"),
            _median_tail(rows, "throughput/images_per_second"),
            summary.get("examples_per_second"),
        ),
        "sequences_per_second": _median_tail(rows, "throughput/sequences_per_second"),
        "max_allocated_gb": _first_number(
            _last(rows, "memory/max_allocated_gb"),
            _last(rows, "memory/max_allocated_gib"),
        ),
        "max_reserved_gb": _first_number(
            _last(rows, "memory/max_reserved_gb"),
            _last(rows, "memory/max_reserved_gib"),
        ),
        "peak_cuda_allocated_bytes": _first_number(
            _last(rows, "memory/peak_cuda_allocated_bytes"),
            summary.get("peak_cuda_allocated_bytes"),
        ),
        "peak_cuda_reserved_bytes": _first_number(
            _last(rows, "memory/peak_cuda_reserved_bytes"),
            summary.get("peak_cuda_reserved_bytes"),
        ),
        "optimizer_state_bytes": _first_number(
            _last(rows, "optimizer/state_bytes"),
            summary.get("optimizer_state_bytes"),
        ),
        "recipe_name": recipe.get("recipe_name"),
        "recipe_version": recipe.get("recipe_version"),
        "paper_phase": recipe.get("paper_phase"),
        "candidate": recipe.get("candidate"),
        "seed": summary.get("seed", config.get("seed")),
        "epochs": summary.get("epochs", config.get("epochs")),
        "model": summary.get("model", config.get("model")),
        "optimizer": summary.get("optimizer", config.get("optimizer")),
    }


def _domain_for_root(root: Path) -> str:
    name = root.name
    if name == "paper_runs" and "model-output" in root.parts:
        return "llm"
    if name in {"imagenet_paper", "paper_runs"} and "imagenet_output" in root.parts:
        return "imagenet"
    if name == "imagenet_paper":
        return "imagenet"
    if name == "diffusion_paper":
        return "diffusion"
    if name == "focused":
        return "llm"
    raise ValueError(
        f"Cannot infer result domain from root {root}; expected paper_runs, focused, "
        "diffusion_paper, imagenet_paper, or imagenet_output/paper_runs"
    )


def collect(roots: list[Path]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for root in roots:
        if not root.exists():
            continue
        domain = _domain_for_root(root)
        for metrics_path in sorted(root.glob("*/metrics.csv")):
            summaries.append(summarize_run(metrics_path.parent, domain))
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots", nargs="*", type=Path, default=list(DEFAULT_ROOTS))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "data",
    )
    args = parser.parse_args()

    summaries = collect(args.roots)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "local_results.json"
    csv_path = args.output_dir / "local_results.csv"
    json_path.write_text(json.dumps(summaries, indent=2, sort_keys=True) + "\n")

    fields = list(summaries[0]) if summaries else [
        "domain",
        "run",
        "path",
        "final_step",
        "eval_loss",
    ]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summaries)
    print(f"Wrote {len(summaries)} runs to {json_path} and {csv_path}")


if __name__ == "__main__":
    main()
