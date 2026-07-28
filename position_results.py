#!/usr/bin/env python3
"""Collect and reshape local MLPRope metrics across selected runs."""

from __future__ import annotations

import argparse
import csv
import fnmatch
import io
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Iterable


CORE_METRICS = ("eval_loss", "eval_loss/context_*", "perplexity")
QK_HEALTH_COLUMNS = (
    "qk_addend_rms_max",
    "qk_addend_abs_max",
    "qk_to_content_p95_max",
    "content_combined_cosine_min",
    "rotary_scale_abs_max",
    "additive_gain_mean",
)
HYPER_HEALTH_COLUMNS = (
    "hyper_phase_delta_rms_max",
    "hyper_phase_delta_p95_max",
    "hyper_log_gain_delta_rms_max",
    "hyper_log_gain_delta_p95_max",
    "hyper_effective_gain_max",
    "hyper_amplitude_delta_rms_max",
    "hyper_amplitude_delta_p95_max",
    "hyper_effective_amplitude_max",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select local runs, filter their metric history by step, and emit "
            "summary, Q/K-health, or raw metric views."
        )
    )
    parser.add_argument(
        "roots",
        nargs="*",
        type=Path,
        default=[Path("model-output")],
        help="Files or directories to scan recursively for metrics.jsonl.",
    )
    parser.add_argument(
        "--run-glob",
        action="append",
        default=[],
        help="Include run names matching this shell glob; repeatable.",
    )
    parser.add_argument("--run-regex", help="Include run names matching this regex.")
    parser.add_argument(
        "--exclude-glob",
        action="append",
        default=[],
        help="Exclude run names matching this shell glob; repeatable.",
    )
    parser.add_argument("--step-min", type=int)
    parser.add_argument("--step-max", type=int)
    parser.add_argument(
        "--every",
        type=int,
        help="Keep steps divisible by this interval after min/max filtering.",
    )
    parser.add_argument(
        "--history",
        action="store_true",
        help="Emit one row per retained step instead of one summary row per run.",
    )
    parser.add_argument(
        "--preset",
        choices=("core", "qk-health", "hyper-health", "all"),
        default="core",
        help="Post-processing view for retained metric records.",
    )
    parser.add_argument(
        "--metric",
        action="append",
        default=[],
        help="Metric shell glob to include; repeatable. Extends the preset.",
    )
    parser.add_argument(
        "--format",
        choices=("table", "markdown", "csv", "json", "jsonl"),
        default="table",
    )
    parser.add_argument(
        "--sort",
        default="final_eval_loss",
        help="Output column used for sorting; prefix with - for descending.",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--output",
        type=Path,
        help="Write rendered output to this file instead of stdout.",
    )
    return parser.parse_args()


def discover_metric_files(roots: Iterable[Path]) -> list[Path]:
    files: set[Path] = set()
    for root in roots:
        if root.is_file() and root.name == "metrics.jsonl":
            files.add(root.resolve())
        elif root.is_dir():
            files.update(path.resolve() for path in root.rglob("metrics.jsonl"))
    return sorted(files)


def selected_run(
    run_name: str,
    *,
    include_globs: list[str],
    include_regex: str | None,
    exclude_globs: list[str],
) -> bool:
    if include_globs and not any(
        fnmatch.fnmatch(run_name, pattern) for pattern in include_globs
    ):
        return False
    if include_regex and re.search(include_regex, run_name) is None:
        return False
    return not any(fnmatch.fnmatch(run_name, pattern) for pattern in exclude_globs)


def load_history(
    path: Path,
    *,
    step_min: int | None,
    step_max: int | None,
    every: int | None,
) -> list[dict[str, Any]]:
    by_step: dict[int, dict[str, Any]] = {}
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            step = record.get("step")
            if not isinstance(step, int):
                continue
            if step_min is not None and step < step_min:
                continue
            if step_max is not None and step > step_max:
                continue
            if every is not None and step % every != 0:
                continue
            # Final evaluation is sometimes logged twice at the same step.
            by_step[step] = record
    return [by_step[step] for step in sorted(by_step)]


def status_for_run(run_dir: Path) -> str:
    for marker, status in (
        ("COMPLETED", "completed"),
        ("FAILED", "failed"),
        ("RUNNING", "running"),
    ):
        if (run_dir / marker).exists():
            return status
    return "unknown"


def numeric_values(record: dict[str, Any], suffixes: tuple[str, ...]) -> list[float]:
    return [
        float(value)
        for key, value in record.items()
        if isinstance(value, (int, float))
        and not isinstance(value, bool)
        and key.endswith(suffixes)
        and math.isfinite(float(value))
    ]


def qk_health(record: dict[str, Any]) -> dict[str, float | None]:
    addend_rms = numeric_values(
        record,
        ("/addend_q/rms", "/addend_k/rms"),
    )
    addend_abs = numeric_values(
        record,
        ("/addend_q/abs_max", "/addend_k/abs_max"),
    )
    ratios = numeric_values(
        record,
        ("addend_q_to_q_ratio_p95", "addend_k_to_k_ratio_p95"),
    )
    cosines = numeric_values(
        record,
        ("q_content_combined_cosine_mean", "k_content_combined_cosine_mean"),
    )
    scales = numeric_values(
        record,
        ("/scale_q/abs_max", "/scale_k/abs_max"),
    )
    gains = numeric_values(
        record,
        ("additive_gain_q_mean", "additive_gain_k_mean"),
    )
    return {
        "qk_addend_rms_max": max(addend_rms, default=None),
        "qk_addend_abs_max": max(addend_abs, default=None),
        "qk_to_content_p95_max": max(ratios, default=None),
        "content_combined_cosine_min": min(cosines, default=None),
        "rotary_scale_abs_max": max(scales, default=None),
        "additive_gain_mean": (
            sum(gains) / len(gains) if gains else None
        ),
    }


def hyper_health(record: dict[str, Any]) -> dict[str, float | None]:
    phase_rms = numeric_values(
        record,
        ("/hyper_phase_delta_q/rms", "/hyper_phase_delta_k/rms"),
    )
    phase_p95 = numeric_values(
        record,
        (
            "/hyper_phase_delta_q/p95_abs",
            "/hyper_phase_delta_k/p95_abs",
        ),
    )
    log_gain_rms = numeric_values(
        record,
        (
            "/hyper_log_gain_delta_q/rms",
            "/hyper_log_gain_delta_k/rms",
        ),
    )
    log_gain_p95 = numeric_values(
        record,
        (
            "/hyper_log_gain_delta_q/p95_abs",
            "/hyper_log_gain_delta_k/p95_abs",
        ),
    )
    effective_gain = numeric_values(
        record,
        (
            "/hyper_effective_gain_q/max",
            "/hyper_effective_gain_k/max",
        ),
    )
    amplitude_rms = numeric_values(
        record,
        (
            "/hyper_amplitude_delta_q/rms",
            "/hyper_amplitude_delta_k/rms",
        ),
    )
    amplitude_p95 = numeric_values(
        record,
        (
            "/hyper_amplitude_delta_q/p95_abs",
            "/hyper_amplitude_delta_k/p95_abs",
        ),
    )
    effective_amplitude = numeric_values(
        record,
        (
            "/hyper_effective_amplitude_q/max",
            "/hyper_effective_amplitude_k/max",
        ),
    )
    return {
        "hyper_phase_delta_rms_max": max(phase_rms, default=None),
        "hyper_phase_delta_p95_max": max(phase_p95, default=None),
        "hyper_log_gain_delta_rms_max": max(log_gain_rms, default=None),
        "hyper_log_gain_delta_p95_max": max(log_gain_p95, default=None),
        "hyper_effective_gain_max": max(effective_gain, default=None),
        "hyper_amplitude_delta_rms_max": max(amplitude_rms, default=None),
        "hyper_amplitude_delta_p95_max": max(amplitude_p95, default=None),
        "hyper_effective_amplitude_max": max(
            effective_amplitude,
            default=None,
        ),
    }


def matching_metrics(
    record: dict[str, Any],
    patterns: Iterable[str],
) -> dict[str, Any]:
    return {
        key: value
        for key, value in record.items()
        if any(fnmatch.fnmatch(key, pattern) for pattern in patterns)
    }


def build_rows(
    metric_files: Iterable[Path],
    *,
    include_globs: list[str],
    include_regex: str | None,
    exclude_globs: list[str],
    step_min: int | None,
    step_max: int | None,
    every: int | None,
    history: bool,
    preset: str,
    metric_patterns: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in metric_files:
        run_dir = path.parent
        run_name = run_dir.name
        if not selected_run(
            run_name,
            include_globs=include_globs,
            include_regex=include_regex,
            exclude_globs=exclude_globs,
        ):
            continue
        records = load_history(
            path,
            step_min=step_min,
            step_max=step_max,
            every=every,
        )
        if not records:
            continue
        chosen_records = records if history else [records[-1]]
        best_record = min(
            (record for record in records if isinstance(record.get("eval_loss"), (int, float))),
            key=lambda record: record["eval_loss"],
            default=None,
        )
        for record in chosen_records:
            row: dict[str, Any] = {
                "run": run_name,
                "status": status_for_run(run_dir),
                "step": record["step"],
            }
            if not history:
                row["points"] = len(records)
                row["first_step"] = records[0]["step"]
                row["final_eval_loss"] = record.get("eval_loss")
                row["best_eval_loss"] = (
                    None if best_record is None else best_record["eval_loss"]
                )
                row["best_step"] = None if best_record is None else best_record["step"]
            patterns = list(metric_patterns)
            if preset in {"core", "qk-health", "hyper-health"}:
                patterns.extend(CORE_METRICS)
            elif preset == "all":
                patterns.append("*")
            row.update(matching_metrics(record, patterns))
            if preset in {"qk-health", "hyper-health"}:
                row.update(qk_health(record))
            if preset == "hyper-health":
                row.update(hyper_health(record))
            rows.append(row)
    return rows


def sort_rows(rows: list[dict[str, Any]], sort_spec: str) -> list[dict[str, Any]]:
    descending = sort_spec.startswith("-")
    key = sort_spec[1:] if descending else sort_spec
    present = [row for row in rows if row.get(key) is not None]
    missing = [row for row in rows if row.get(key) is None]
    return sorted(
        present,
        key=lambda row: row[key],
        reverse=descending,
    ) + missing


def columns_for_rows(rows: list[dict[str, Any]]) -> list[str]:
    preferred = [
        "run",
        "status",
        "step",
        "points",
        "first_step",
        "final_eval_loss",
        "best_eval_loss",
        "best_step",
        *QK_HEALTH_COLUMNS,
        *HYPER_HEALTH_COLUMNS,
        "eval_loss",
        "perplexity",
    ]
    present = {key for row in rows for key in row}
    return [key for key in preferred if key in present] + sorted(
        present - set(preferred)
    )


def display_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def render_delimited(
    rows: list[dict[str, Any]],
    columns: list[str],
    *,
    delimiter: str,
) -> str:
    output = io.StringIO()
    writer = csv.writer(output, delimiter=delimiter, lineterminator="\n")
    writer.writerow(columns)
    for row in rows:
        writer.writerow([display_value(row.get(column)) for column in columns])
    return output.getvalue()


def render_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    rendered = [
        [display_value(row.get(column)) for column in columns] for row in rows
    ]
    widths = [
        max(len(column), *(len(row[index]) for row in rendered))
        for index, column in enumerate(columns)
    ]
    lines = [
        "  ".join(column.ljust(widths[index]) for index, column in enumerate(columns)),
        "  ".join("-" * width for width in widths),
    ]
    lines.extend(
        "  ".join(value.ljust(widths[index]) for index, value in enumerate(row))
        for row in rendered
    )
    return "\n".join(lines) + "\n"


def render_markdown(rows: list[dict[str, Any]], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    body = [
        "| "
        + " | ".join(display_value(row.get(column)) for column in columns)
        + " |"
        for row in rows
    ]
    return "\n".join([header, divider, *body]) + "\n"


def render(rows: list[dict[str, Any]], output_format: str) -> str:
    columns = columns_for_rows(rows)
    if output_format == "json":
        return json.dumps(rows, indent=2, sort_keys=True) + "\n"
    if output_format == "jsonl":
        return "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    if not rows:
        return ""
    if output_format == "csv":
        return render_delimited(rows, columns, delimiter=",")
    if output_format == "markdown":
        return render_markdown(rows, columns)
    return render_table(rows, columns)


def main() -> int:
    args = parse_args()
    if args.every is not None and args.every <= 0:
        raise ValueError("--every must be positive")
    metric_files = discover_metric_files(args.roots)
    rows = build_rows(
        metric_files,
        include_globs=args.run_glob,
        include_regex=args.run_regex,
        exclude_globs=args.exclude_glob,
        step_min=args.step_min,
        step_max=args.step_max,
        every=args.every,
        history=args.history,
        preset=args.preset,
        metric_patterns=args.metric,
    )
    rows = sort_rows(rows, args.sort)
    if args.limit is not None:
        rows = rows[: args.limit]
    rendered = render(rows, args.format)
    if args.output is None:
        sys.stdout.write(rendered)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
