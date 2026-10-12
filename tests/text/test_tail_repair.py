"""尾部修复从公开文本接口进入，预期使用独立正文样例。"""

import unittest

import realtime_asr as sdk


class TailRepairTests(unittest.TestCase):
    def test_exact_anchor_discards_untrusted_tail_prefix(self):
        result = sdk.repair_tail(
            "we use speech to text system", "boundary noise speech to text system today"
        )
        self.assertEqual(result.text, "we use speech to text system today")
        self.assertEqual((result.status, result.matched), ("exact", True))

    def test_partial_word_repair_requires_following_exact_evidence(self):
        result = sdk.repair_tail("we use spee to text system", "speech to text system today")
        self.assertEqual(result.text, "we use speech to text system today")
        self.assertEqual((result.status, result.used_fuzzy_match), ("fuzzy", True))

    def test_unstable_last_word_uses_three_word_anchor(self):
        result = sdk.repair_tail("we use speech to text syst", "speech to text system today")
        self.assertEqual(result.text, "we use speech to text system today")
        self.assertEqual(result.status, "fuzzy")

    def test_one_long_word_edit_is_repaired_without_using_tail_prefix(self):
        result = sdk.repair_tail("we use speech to tezt", "noise speech to text today")
        self.assertEqual(result.text, "we use speech to text today")
        self.assertEqual((result.status, result.distance), ("fuzzy", 1))

    def test_chinese_exact_anchor_and_short_overlap_have_different_results(self):
        exact = sdk.repair_tail("今天测试语音系统", "语音系统效果很好")
        self.assertEqual(exact.text, "今天测试语音系统效果很好")
        self.assertEqual(exact.status, "exact")
        short = sdk.repair_tail("今天测试语音系统", "系统已经修改")
        self.assertEqual((short.text, short.status), ("今天测试语音系统", "alignment_failed"))

    def test_empty_live_result_keeps_explicit_tail_scope(self):
        result = sdk.repair_tail("", "only this tail")
        self.assertEqual(
            (result.text, result.status, result.matched), ("only this tail", "tail_only", False)
        )
