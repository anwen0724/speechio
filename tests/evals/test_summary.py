"""重复和中间文本不能进入主评分分母。"""

import unittest

from evals.p0.evaluation.summary import grouped_scores


class SummaryTests(unittest.TestCase):
    def test_only_first_complete_primary_result_enters_primary_group(self):
        sample = {"sample_id": "cn", "language": "zh", "reference_text": "你好世界"}
        common = {
            "config_id": "cpu-whole",
            "sample_id": "cn",
            "status": "ok",
            "input_path": "whole",
            "language_mode": "fixed",
            "corpus_version": "v1",
            "output_kind": "whole_final",
        }
        trials = [
            {**common, "trial_index": 1, "evaluation_role": "primary", "result": {"text": "你好"}},
            {
                **common,
                "trial_index": 2,
                "evaluation_role": "repeat",
                "result": {"text": "你好世界"},
            },
            {
                **common,
                "trial_index": 1,
                "evaluation_role": "diagnostic",
                "output_kind": "snapshot_partial",
                "result": {"text": "你好世界"},
            },
        ]
        groups = grouped_scores(trials, [sample])
        primary = next(group for group in groups if group["evaluation_role"] == "primary")
        self.assertEqual((primary["N"], primary["E"], primary["error_rate"]), (4, 2, 0.5))
        self.assertEqual(primary["successful_trials"], 1)
        self.assertEqual(len(groups), 2)
