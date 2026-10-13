import json
from pathlib import Path
import tempfile
import unittest

from tools.run_attention_ab import file_hash
from tools.summarize_attention_ab import summarize


class AttentionSummaryTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.plan = self.root / "plan.json"
        self.plan.write_text(json.dumps({"job_count_per_variant": 2,
                                         "pairs": [{"index": 1, "scene": 1, "seed": 42},
                                                   {"index": 2, "scene": 2, "seed": 99}]}))
        (self.root / "provenance.json").write_text(json.dumps({"plan_sha256": file_hash(self.plan)}))
        self.rows = [{"index": 1, "seed": 42, "status": "completed", "render_seconds": 10},
                     {"index": 2, "seed": 99, "status": "completed", "render_seconds": 20}]
        self.write_results("sage_on", self.rows)
        self.write_results("sage_off", [dict(row, render_seconds=row["render_seconds"] * 2)
                                        for row in self.rows])

    def write_results(self, name, rows):
        directory = self.root / name
        directory.mkdir(exist_ok=True)
        (directory / "results.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))

    def test_pairs_are_matched_by_index_not_completion_order(self):
        self.write_results("sage_off", [dict(row, render_seconds=row["render_seconds"] * 2)
                                        for row in reversed(self.rows)])
        summary = summarize(self.plan, self.root)
        self.assertEqual(summary["paired_timing_count"], 2)
        self.assertEqual(summary["median_paired_sage_speedup"], 2)

    def test_missing_and_failed_results_never_imply_speedup(self):
        self.write_results("sage_off", [dict(self.rows[0], status="error")])
        summary = summarize(self.plan, self.root)
        self.assertEqual(summary["paired_timing_count"], 0)
        self.assertIsNone(summary["median_paired_sage_speedup"])
        self.assertEqual(summary["completed_jobs"]["sage_off"], 0)

    def test_changed_plan_is_rejected(self):
        self.plan.write_text(self.plan.read_text() + " ")
        with self.assertRaisesRegex(ValueError, "different plan"):
            summarize(self.plan, self.root)

    def test_mismatched_sampling_settings_are_rejected(self):
        self.write_results("sage_off", [dict(self.rows[0], steps=4)])
        with self.assertRaisesRegex(ValueError, "settings differ"):
            summarize(self.plan, self.root)

    def test_duplicate_results_are_rejected(self):
        self.write_results("sage_off", [self.rows[0], self.rows[0]])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            summarize(self.plan, self.root)
