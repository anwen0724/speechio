"""阶段结论不能由模型有文字输出推定。"""

import json
import tempfile
import unittest
from pathlib import Path

from evals.p0.report import phase_status
from evals.p0.saved_report import rebuild_report
from speechio.errors import ASRError


class ReportTests(unittest.TestCase):
    def test_missing_corpus_snapshot_cannot_rebuild_report(self):
        with tempfile.TemporaryDirectory() as folder:
            Path(folder, "records.jsonl").write_text("", encoding="utf-8")
            with self.assertRaises(ASRError) as caught:
                rebuild_report(folder)
            self.assertEqual(caught.exception.code, "RECORD_INCOMPLETE")

    def test_partial_draft_run_neither_passes_nor_produces_quality_scores(self):
        samples = [
            {
                "sample_id": "cn",
                "categories": ["C01"],
                "annotation_checked": False,
                "language": "zh",
                "reference_text": "原始正文",
                "corpus_version": "v1",
            },
            {
                "sample_id": "en",
                "categories": ["C07"],
                "annotation_checked": False,
                "language": "en",
                "reference_text": "original text",
                "corpus_version": "v1",
            },
        ]
        corpus = {
            "corpus_version": "v1",
            "samples": samples,
            "representative_sample_id": {"C01": "cn", "C07": "en"},
        }
        records = [
            {"record_type": "corpus", **corpus},
            *[{"record_type": "sample", **item} for item in samples],
            {
                "record_type": "trial_result",
                "trial_id": "trial",
                "matrix_item": "M01",
                "sample_id": "cn",
                "status": "ok",
                "evaluation_role": "diagnostic",
                "config_id": "config",
                "scenario": None,
                "trial_index": 1,
                "result": {"text": "返回正文"},
                "exploratory": True,
            },
        ]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder, "records.jsonl")
            content = "".join(
                json.dumps({"schema_version": 1, "run_id": "run", **item}) + "\n"
                for item in records
            )
            path.write_text(content, encoding="utf-8")
            report = rebuild_report(folder)
            self.assertEqual(path.read_text(encoding="utf-8"), content)
            self.assertEqual(report["status"], "blocked")
            self.assertEqual(report["scores"], [])
            self.assertEqual(
                next(item for item in report["acceptance"] if item["acceptance_id"] == "A02")[
                    "status"
                ],
                "blocked",
            )

    def test_unreviewed_or_missing_core_is_blocked(self):
        self.assertEqual(phase_status(reviewed=False, core=["ok"], checks=["passed"]), "blocked")
        self.assertEqual(
            phase_status(reviewed=True, core=["skipped"], checks=["passed"]), "blocked"
        )

    def test_failure_and_optional_limits_are_separate(self):
        self.assertEqual(phase_status(reviewed=True, core=["failed"], checks=["passed"]), "failed")
        self.assertEqual(
            phase_status(reviewed=True, core=["ok"], checks=["passed"], limits=["GPU unavailable"]),
            "passed_with_limit",
        )
