from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from position_results import (
    build_rows,
    discover_metric_files,
    load_history,
    render,
)


class PositionResultsTest(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.run = self.root / "family" / "example-seed123"
        self.run.mkdir(parents=True)
        (self.run / "COMPLETED").touch()
        records = [
            {"step": 500, "eval_loss": 5.0},
            {"step": 1000, "eval_loss": 4.8},
            {"step": 1000, "eval_loss": 4.7},
            {
                "step": 1500,
                "eval_loss": 4.75,
                "position/layer_00/qk/addend_q/rms": 0.2,
                "position/layer_01/qk/addend_k/rms": 0.3,
                "position/layer_00/qk/addend_q/abs_max": 0.8,
                "position/layer_00/qk/addend_q_to_q_ratio_p95": 0.24,
                "position/layer_00/qk/q_content_combined_cosine_mean": 0.98,
                "position/layer_00/qk/additive_gain_q_mean": 0.21,
                "position/layer_00/qk/hyper_phase_delta_q/rms": 0.04,
                "position/layer_01/qk/hyper_phase_delta_k/p95_abs": 0.12,
                "position/layer_00/qk/hyper_log_gain_delta_q/rms": 0.03,
                "position/layer_01/qk/hyper_log_gain_delta_k/p95_abs": 0.09,
                "position/layer_01/qk/hyper_effective_gain_k/max": 1.15,
            },
        ]
        self.metrics_path = self.run / "metrics.jsonl"
        self.metrics_path.write_text(
            "".join(json.dumps(record) + "\n" for record in records)
        )

    def tearDown(self):
        self.tempdir.cleanup()

    def test_discovers_and_deduplicates_filtered_steps(self):
        self.assertEqual(discover_metric_files([self.root]), [self.metrics_path])
        history = load_history(
            self.metrics_path,
            step_min=750,
            step_max=1500,
            every=500,
        )
        self.assertEqual([record["step"] for record in history], [1000, 1500])
        self.assertEqual(history[0]["eval_loss"], 4.7)

    def test_summary_and_qk_health_postprocessing(self):
        rows = build_rows(
            [self.metrics_path],
            include_globs=["example-*"],
            include_regex="seed123",
            exclude_globs=[],
            step_min=None,
            step_max=None,
            every=None,
            history=False,
            preset="qk-health",
            metric_patterns=[],
        )
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["status"], "completed")
        self.assertEqual(row["points"], 3)
        self.assertEqual(row["best_eval_loss"], 4.7)
        self.assertEqual(row["best_step"], 1000)
        self.assertEqual(row["qk_addend_rms_max"], 0.3)
        self.assertEqual(row["qk_to_content_p95_max"], 0.24)
        self.assertEqual(row["content_combined_cosine_min"], 0.98)

    def test_hyper_health_postprocessing_is_compact(self):
        rows = build_rows(
            [self.metrics_path],
            include_globs=[],
            include_regex=None,
            exclude_globs=[],
            step_min=None,
            step_max=None,
            every=None,
            history=False,
            preset="hyper-health",
            metric_patterns=[],
        )
        row = rows[0]
        self.assertEqual(row["hyper_phase_delta_rms_max"], 0.04)
        self.assertEqual(row["hyper_phase_delta_p95_max"], 0.12)
        self.assertEqual(row["hyper_log_gain_delta_rms_max"], 0.03)
        self.assertEqual(row["hyper_log_gain_delta_p95_max"], 0.09)
        self.assertEqual(row["hyper_effective_gain_max"], 1.15)
        self.assertEqual(row["qk_addend_rms_max"], 0.3)

    def test_history_and_machine_readable_rendering(self):
        rows = build_rows(
            [self.metrics_path],
            include_globs=[],
            include_regex=None,
            exclude_globs=[],
            step_min=1000,
            step_max=None,
            every=None,
            history=True,
            preset="core",
            metric_patterns=[],
        )
        self.assertEqual(len(rows), 2)
        rendered = json.loads(render(rows, "json"))
        self.assertEqual(rendered[0]["step"], 1000)
        self.assertIn("example-seed123", render(rows, "markdown"))


if __name__ == "__main__":
    unittest.main()
