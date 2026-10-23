"""稳定文本使用公开观察接口；时序和正文来自独立样例。"""

import unittest

from speechio import TurnRef
from speechio.stt.text.stabilizer import StabilizationConfig, TextObservation, TextStabilizer


class StabilizerTests(unittest.TestCase):
    def test_audio_progress_and_final_conflict(self):
        unit = TurnRef("session", 0, 1)
        stabilizer = TextStabilizer(unit)
        first = stabilizer.observe(TextObservation(unit, 1, "你好世界", 16000, 1.0))
        self.assertEqual(first.stable_text, "")
        repeated = stabilizer.observe(TextObservation(unit, 2, "你好世界", 16000, 2.0))
        self.assertEqual(repeated.stable_text, "")
        confirmed = stabilizer.observe(TextObservation(unit, 3, "你好世界呀", 32000, 2.1))
        self.assertEqual(confirmed.stable_text, "你好世界")
        self.assertEqual(confirmed.stable_delta, "你好世界")
        self.assertEqual(confirmed.unstable_text, "呀")
        final = stabilizer.finalize("你好地球")
        self.assertEqual(final.final_text, "你好地球")
        self.assertFalse(final.agrees_with_stable_prefix)
        self.assertEqual(final.mismatch_reason, "stable-prefix-mismatch")
        self.assertFalse(
            stabilizer.observe(TextObservation(unit, 4, "你好世界", 48000, 3)).accepted
        )

    def test_english_space_and_punctuation_need_right_context(self):
        unit = TurnRef("session", 0, 1)
        stabilizer = TextStabilizer(unit)
        for sequence, time in enumerate((0, 0.9, 1.2, 1.5), 1):
            result = stabilizer.observe(
                TextObservation(unit, sequence, "Hello, world again", sequence * 16000, time)
            )
        self.assertEqual(result.stable_text, "Hello, world again")
        self.assertTrue(stabilizer.finalize("HELLO, world again!").agrees_with_stable_prefix)

    def test_stale_wrong_unit_outlier_and_bounded_history(self):
        unit = TurnRef("session", 0, 1)
        stabilizer = TextStabilizer(unit, StabilizationConfig(initial_prefix_span=1000))
        stabilizer.observe(TextObservation(unit, 1, "你好世界", 16000, 1))
        self.assertEqual(
            stabilizer.observe(TextObservation(unit, 1, "错误", 32000, 2)).ignored_reason,
            "stale-sequence",
        )
        self.assertEqual(
            stabilizer.observe(
                TextObservation(TurnRef("session", 1, 1), 2, "错误", 32000, 2)
            ).ignored_reason,
            "wrong-unit",
        )
        self.assertEqual(
            stabilizer.observe(TextObservation(unit, 2, "abcdef", 32000, 2)).ignored_reason,
            "outlier",
        )
        for sequence in range(3, 160):
            result = stabilizer.observe(
                TextObservation(unit, sequence, "你好世界", sequence * 16000, 2 + sequence / 100)
            )
        self.assertEqual(result.retained_observations, 128)
        self.assertEqual(result.stable_text, "")
