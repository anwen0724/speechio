"""句末策略只给出提示与真实词切点，使用语言规则的代表场景。"""

import unittest

from realtime_asr import RecognitionResult, WordTimestamp
from realtime_asr.text.endpoint import SentenceBoundaryPolicy


class SentencePolicyTests(unittest.TestCase):
    def test_confirmed_raw_period_can_use_stable_characters_that_omitted_punctuation(self):
        policy = SentenceBoundaryPolicy()
        stable = "今天天气好我们继续说话"
        raw = "今天天气好。我们继续说话"
        self.assertIsNone(policy.observe(stable, 32000, observation_text=raw))
        hint = policy.observe(stable, 40000, observation_text=raw)
        self.assertIsNotNone(hint)
        self.assertEqual(hint.mark, "。")
        # 时间模型只负责提供左侧锚点位置，不需要再次输出标点或右侧词。
        timed = RecognitionResult(
            words=(WordTimestamp("今天", 0, 5000), WordTimestamp("天气好", 5000, 12345)),
            timestamp_status="reported",
        )
        self.assertEqual(policy.locate(hint, timed), 12345)

    def test_chinese_hint_requires_progress_and_aligns_to_reported_word(self):
        policy = SentenceBoundaryPolicy()
        self.assertIsNone(policy.observe("今天天气好。我们继续说话", 32000))
        self.assertIsNone(policy.observe("今天天气好。我们继续说话", 32000))
        hint = policy.observe("今天天气好。我们继续说话", 40000)
        self.assertIsNotNone(hint)
        result = RecognitionResult(
            text="今天天气好。我们继续说话",
            words=(
                WordTimestamp("今天", 0, 5000),
                WordTimestamp("天气好。", 5000, 12345),
                WordTimestamp("我们继续说话", 12345, 25000),
            ),
            timestamp_status="reported",
        )
        self.assertEqual(policy.locate(hint, result), 12345)
        self.assertIsNone(policy.locate(hint, RecognitionResult(text=result.text)))
        self.assertIsNone(
            policy.locate(
                hint,
                RecognitionResult(
                    words=(WordTimestamp("不同的结果。我们继续说话", 0, 16000),),
                    timestamp_status="reported",
                ),
            )
        )

    def test_numbers_abbreviations_experimental_marks_and_short_context_do_not_split(self):
        for text in (
            "value is 3.14 and growing",
            "please see Dr. Smith again",
            "one two three... four five",
            "one two three, four five six",
            "好。我们继续",
            "今天天气好。A1",
        ):
            with self.subTest(text=text):
                policy = SentenceBoundaryPolicy()
                self.assertIsNone(policy.observe(text, 32000))
                self.assertIsNone(policy.observe(text, 40000))
