"""通过公开合并器验证正文基准和极速尾部，不依赖执行器实现。"""

import unittest

import speechio as sdk
from speechio.stt.text.realtime_merge import RealtimeMerger
from speechio.stt.text.stabilizer import TextObservation


class RealtimeMergeTests(unittest.TestCase):
    def observation(self, unit, sequence, text):
        return TextObservation(unit, sequence, text, sequence * 16000, float(sequence))

    def test_fast_waits_for_accurate_and_only_extends_after_exact_anchor(self):
        unit = sdk.TurnRef("merge", 0, 1)
        merger = RealtimeMerger(unit)
        fast = TextObservation(unit, 1, "we use speech system today", 32000, 1.0)
        waiting = merger.observe("ultrafast", fast)
        self.assertEqual((waiting.status, waiting.text), ("waiting_for_accurate", ""))
        merged = merger.observe(
            "accurate", TextObservation(unit, 1, "We use speech system", 16000, 2.0)
        )
        self.assertEqual(merged.text, "We use speech system today")
        self.assertEqual(merged.ultrafast_suffix, "today")
        self.assertTrue(merged.matched)
        self.assertEqual((merged.accurate_sequence, merged.ultrafast_sequence), (1, 1))
        limited = merger.observe(
            "ultrafast",
            self.observation(unit, 2, "we use speech system today tomorrow next week soon later"),
        )
        self.assertEqual(limited.ultrafast_suffix, "today tomorrow next week soon")

    def test_old_unit_sequence_or_audio_cannot_change_current_merge(self):
        unit = sdk.TurnRef("current", 1, 2)
        merger = RealtimeMerger(unit)
        merger.observe("accurate", self.observation(unit, 1, "speech system"))
        current = merger.observe("ultrafast", self.observation(unit, 1, "speech system today"))
        invalid = (
            self.observation(sdk.TurnRef("current", 0, 1), 2, "old unit"),
            self.observation(unit, 1, "old sequence"),
            TextObservation(unit, 2, "old audio", 8000, 2.0),
        )
        for observation in invalid:
            with self.subTest(observation=observation):
                result = merger.observe("ultrafast", observation)
                self.assertFalse(result.accepted)
                self.assertIsNotNone(result.ignored_reason)
                self.assertEqual(merger.snapshot(), current)

    def test_retained_text_keeps_its_real_observation_coverage(self):
        unit = sdk.TurnRef("coverage", 0, 1)
        merger = RealtimeMerger(unit)
        merger.observe("accurate", self.observation(unit, 1, "speech system"))
        merger.observe("ultrafast", self.observation(unit, 1, "speech system today"))
        blank = merger.observe("accurate", self.observation(unit, 2, ""))
        self.assertEqual((blank.accurate_sequence, blank.accurate_audio_end_sample), (1, 16000))
        held = merger.observe("ultrafast", self.observation(unit, 2, "speech system tomorrow"))
        self.assertEqual(
            (held.ultrafast_suffix, held.suffix_sequence, held.suffix_audio_end_sample),
            ("today", 1, 16000),
        )

    def test_chinese_anchor_keeps_tail_until_accurate_content_advances(self):
        unit = sdk.TurnRef("chinese", 0, 1)
        merger = RealtimeMerger(unit)
        merger.observe("accurate", self.observation(unit, 1, "今天测试语音系统"))
        unrelated = merger.observe("ultrafast", self.observation(unit, 1, "今天讨论别的内容"))
        self.assertEqual(unrelated.text, "今天测试语音系统")
        merged = merger.observe("ultrafast", self.observation(unit, 2, "今天测试语音系统效果很好"))
        self.assertEqual(merged.text, "今天测试语音系统效果很好")
        conflict = merger.observe(
            "ultrafast", self.observation(unit, 3, "今天测试语音系统效果不错")
        )
        self.assertEqual(
            (conflict.text, conflict.status), ("今天测试语音系统效果很好", "held_fast_conflict")
        )
        advanced = merger.observe("accurate", self.observation(unit, 2, "今天测试识别接口"))
        self.assertEqual((advanced.text, advanced.accurate_generation), ("今天测试识别接口", 2))
        self.assertEqual(advanced.ultrafast_suffix, "")

    def test_fuzzy_anchor_waits_for_requested_confirmation(self):
        unit = sdk.TurnRef("fuzzy", 0, 1)
        merger = RealtimeMerger(unit, sdk.RealtimeMergeConfig(fuzzy_confirmations=2))
        merger.observe("accurate", self.observation(unit, 1, "use speech system"))
        pending = merger.observe("ultrafast", self.observation(unit, 1, "use speach system today"))
        self.assertEqual(
            (pending.status, pending.text), ("held_fuzzy_pending", "use speech system")
        )
        confirmed = merger.observe(
            "ultrafast", self.observation(unit, 2, "use speach system today")
        )
        self.assertEqual((confirmed.status, confirmed.text), ("fuzzy", "use speech system today"))
        self.assertTrue(confirmed.used_fuzzy_match)

    def test_monotonic_frontier_requires_two_strong_supports(self):
        unit = sdk.TurnRef("frontier", 0, 1)
        merger = RealtimeMerger(unit)
        merger.observe("accurate", self.observation(unit, 1, "we do audio conversion"))
        merged = merger.observe(
            "ultrafast", self.observation(unit, 1, "we do extra audio live conversion today")
        )
        self.assertEqual(
            (merged.status, merged.text), ("soft_frontier", "we do audio conversion today")
        )
        weak = RealtimeMerger(unit)
        weak.observe("accurate", self.observation(unit, 1, "we cat"))
        conflict = weak.observe("ultrafast", self.observation(unit, 1, "we bat today"))
        self.assertEqual(conflict.text, "we cat")
        self.assertFalse(conflict.matched)

    def test_mixed_text_uses_word_evidence_for_its_english_tail(self):
        unit = sdk.TurnRef("mixed", 0, 1)
        merger = RealtimeMerger(unit)
        merger.observe("accurate", self.observation(unit, 1, "今天讨论 speech system"))
        merged = merger.observe("ultrafast", self.observation(unit, 1, "speech system today"))
        self.assertEqual(merged.text, "今天讨论 speech system today")
        self.assertEqual(merged.anchor_length, 2)
