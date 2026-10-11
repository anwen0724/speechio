"""评分预期使用 Spec 示例和独立已知样例。"""

import unittest

from evals.p0.evaluation.scoring import score


class ScoringTests(unittest.TestCase):
    def test_chinese_deletions_follow_spec_example(self):
        result = score("你好世界", "你好", "zh")
        self.assertEqual(
            (result["N"], result["E"], result["D"], result["error_rate"]), (4, 2, 2, 0.5)
        )

    def test_nfkc_punctuation_and_english_word_boundaries_are_fixed(self):
        self.assertEqual(score("Ａ B，你好！", "ab你好", "zh")["E"], 0)
        self.assertEqual(score("Don't STOP.", "don t stop", "en")["E"], 0)
        self.assertEqual(score("one two", "one three four", "en")["E"], 2)

    def test_empty_hypothesis_is_deletion_but_empty_reference_has_no_rate(self):
        self.assertEqual(score("你好", "", "zh")["D"], 2)
        self.assertIsNone(score("", "误转写", "none")["error_rate"])
