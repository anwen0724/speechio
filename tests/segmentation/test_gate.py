"""分句接口以标准音频和检测记录为输入，核对可定位的实际音频。"""

import unittest

from realtime_asr.core.audio.conversion import AudioFrame
from realtime_asr.core.segmentation.gate import SegmentationConfig, SegmentationGate
from realtime_asr.core.segmentation.vad import VoiceActivity


class SegmentationTests(unittest.TestCase):
    def test_timestamp_cut_keeps_unaligned_right_samples(self):
        gate = SegmentationGate(SegmentationConfig(pre_roll_duration=0), max_segment_samples=480000)
        left = gate.feed(
            AudioFrame(b"\x01\x00" * 1024, 0, 1024), VoiceActivity(True, True, True, True, 1)
        )[0]
        self.assertTrue(hasattr(gate, "split_at"), "门控尚未提供有时间证据的左右切分")
        right = gate.split_at(777)
        gate.feed(
            AudioFrame(b"\x02\x00" * 100, 1024, 1124), VoiceActivity(False, True, True, True, 2)
        )
        gate.seal(reason="input_finished")
        self.assertEqual((left.boundary.end_sample, right.boundary.start_sample), (777, 777))
        self.assertEqual(bytes(left.pcm + right.pcm), b"\x01\x00" * 1024 + b"\x02\x00" * 100)

    def feed(self, gate, count, *, speech, onset=False):
        segments = []
        for _ in range(count):
            start = gate.position
            segments.extend(
                gate.feed(
                    AudioFrame(bytes(1024), start, start + 512),
                    VoiceActivity(onset, speech, speech, onset, 0),
                )
            )
        return segments

    def test_short_pause_keeps_one_segment_and_confirmed_tail_seals(self):
        gate = SegmentationGate(SegmentationConfig(), max_segment_samples=480000)
        self.feed(gate, 5, speech=False)
        segment = self.feed(gate, 20, speech=True, onset=True)[0]
        self.feed(gate, 10, speech=False)
        self.assertIsNone(segment.boundary)
        self.feed(gate, 1, speech=True)
        self.feed(gate, 23, speech=False)
        self.assertIsNone(segment.boundary)
        self.feed(gate, 1, speech=False)
        self.assertIsNotNone(segment.boundary, "连续静音达到阈值后仍未封口")
        self.assertEqual(segment.boundary.end_sample, 60 * 512)
        self.assertEqual(segment.boundary.reason, "silence")

    def test_capacity_split_preserves_adjacent_samples(self):
        gate = SegmentationGate(SegmentationConfig(pre_roll_duration=0), max_segment_samples=1000)
        segments = self.feed(gate, 4, speech=True, onset=True)
        gate.seal(reason="input_finished")
        self.assertEqual(
            [(s.boundary.start_sample, s.boundary.end_sample) for s in segments],
            [(0, 1000), (1000, 2000), (2000, 2048)],
        )
        self.assertEqual(b"".join(bytes(s.pcm) for s in segments), bytes(4096))

    def test_backdate_uses_one_cut_for_left_audio_and_next_preroll(self):
        gate = SegmentationGate(
            SegmentationConfig(backdate_duration=0.2), max_segment_samples=480000
        )
        first = self.feed(gate, 20, speech=True, onset=True)[0]
        self.feed(gate, 24, speech=False)
        self.assertEqual(first.boundary.end_sample, 44 * 512 - 3200)
        second = self.feed(gate, 1, speech=True, onset=True)[0]
        gate.seal()
        self.assertEqual(second.boundary.start_sample, first.boundary.end_sample)
        self.assertEqual(len(first.pcm) + len(second.pcm), 45 * 1024)

    def test_conservative_trim_keeps_quiet_leadin_and_guard(self):
        gate = SegmentationGate(SegmentationConfig(trim_pre_roll=True), max_segment_samples=480000)
        self.feed(gate, 25, speech=False)
        for _ in range(2):
            start = gate.position
            gate.feed(
                AudioFrame(b"\x64\x00" * 512, start, start + 512),
                VoiceActivity(False, False, False, False, 100),
            )
        segment = self.feed(gate, 1, speech=True, onset=True)[0]
        self.assertEqual(segment.pre_roll_reason, "trimmed")
        self.assertEqual(segment.start_sample, 4096)
        self.assertEqual(bytes(segment.pcm[-3072:-1024]), b"\x64\x00" * 1024)

    def test_minimum_gap_applies_between_segments_and_keeps_waiting_onset(self):
        gate = SegmentationGate(SegmentationConfig(min_gap=0.25), max_segment_samples=480000)
        first = self.feed(gate, 1, speech=True, onset=True)
        self.assertEqual(len(first), 1, "首段之前没有录音间隔")
        gate.seal()
        self.assertEqual(self.feed(gate, 1, speech=True, onset=True), [])
        self.assertEqual(self.feed(gate, 6, speech=False), [])
        second = self.feed(gate, 1, speech=False)[0]
        self.assertEqual(second.start_sample, 512)
        self.assertEqual(len(second.pcm) // 2, 4096)

    def test_trim_does_not_remove_speech_waiting_for_minimum_gap(self):
        gate = SegmentationGate(
            SegmentationConfig(min_gap=0.8, trim_pre_roll=True), max_segment_samples=480000
        )
        self.feed(gate, 1, speech=True, onset=True)
        gate.seal()
        self.feed(gate, 1, speech=True, onset=True)
        self.feed(gate, 22, speech=False)
        self.feed(gate, 1, speech=True, onset=True)
        second = self.feed(gate, 1, speech=False)[0]
        self.assertEqual(second.start_sample, 512, "等待间隔时已经出现的首部不能被裁剪")

    def test_backdated_silence_still_obeys_preroll_budget(self):
        gate = SegmentationGate(
            SegmentationConfig(pre_roll_duration=0.1, backdate_duration=0.2),
            max_segment_samples=480000,
        )
        self.feed(gate, 20, speech=True, onset=True)
        self.feed(gate, 24, speech=False)
        self.assertEqual(gate.history_samples, 1600)
