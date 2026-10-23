"""通过公开转换接口验证信号与位置，不读取内部缓冲。"""

import unittest
from unittest.mock import patch

import numpy as np
import soxr

from speechio.errors import STTError
from speechio.stt.input import AudioConverter, AudioFormat


class ConversionTests(unittest.TestCase):
    def test_resampling_keeps_one_second_and_matches_independent_hq(self):
        for rate in (48000, 44100):
            with self.subTest(rate=rate):
                source = (0.4 * np.sin(np.arange(rate) * 2 * np.pi * 440 / rate)).astype(np.float32)
                expected = (
                    np.clip(soxr.resample(source, rate, 16000, quality="HQ"), -1, 1) * 32767
                ).astype(np.int16)
                for chunk_size in (rate, 160, 1000):
                    converter = AudioConverter(AudioFormat(rate, sample_type="float32"))
                    frames = []
                    for start in range(0, rate, chunk_size):
                        frames += converter.feed(source[start : start + chunk_size])
                    frames += converter.finish()
                    actual = np.frombuffer(b"".join(f.pcm for f in frames), dtype="<i2")
                    self.assertEqual(len(actual), 16000)
                    self.assertLessEqual(np.max(np.abs(actual.astype(int) - expected)), 1)

    def test_standard_pcm_keeps_values_and_short_tail_positions(self):
        source = np.arange(525, dtype=np.int16)
        converter = AudioConverter(AudioFormat())
        frames = converter.feed(source) + converter.finish()
        self.assertEqual(b"".join(frame.pcm for frame in frames), source.astype("<i2").tobytes())
        self.assertEqual([(f.start_sample, f.end_sample) for f in frames], [(0, 512), (512, 525)])

    def test_float_mix_and_clip_follow_known_quantization(self):
        source = np.array([[-1, 1], [-0.5, 0], [0, 0.5], [1, 1], [-2, -2]], np.float64)
        converter = AudioConverter(AudioFormat(channels=2, sample_type="float32"))
        frames = converter.feed(source) + converter.finish()
        expected = np.array([0, -8191, 8191, 32767, -32767], dtype="<i2").tobytes()
        self.assertEqual(b"".join(f.pcm for f in frames), expected)

    def test_byte_fragments_preserve_source_frames_and_representation(self):
        raw = np.array([[1000, -1000], [101, 100]], dtype="<i2").tobytes()
        converter = AudioConverter(AudioFormat(channels=2), frame_samples=1)
        frames = converter.feed(b"") + converter.feed(raw[:3])
        with self.assertRaises(STTError) as caught:
            converter.feed(np.zeros((1, 2), np.int16))
        self.assertEqual(caught.exception.code, "INVALID_AUDIO")
        frames += converter.feed(bytearray(raw[3:])) + converter.finish()
        self.assertEqual(b"".join(f.pcm for f in frames), b"\x00\x00\x64\x00")
        self.assertEqual([(f.start_sample, f.end_sample) for f in frames], [(0, 1), (1, 2)])

    def test_rejected_float_block_does_not_consume_byte_remainder(self):
        raw = np.array([0.5], dtype="<f4").tobytes()
        converter = AudioConverter(AudioFormat(sample_type="float32"))
        converter.feed(raw[:2])
        with self.assertRaises(STTError) as caught:
            converter.feed(raw[2:] + np.array([np.nan], dtype="<f4").tobytes())
        self.assertEqual(caught.exception.code, "INVALID_AUDIO")
        frames = converter.feed(raw[2:]) + converter.finish()
        self.assertEqual(b"".join(f.pcm for f in frames), b"\xff\x3f")

    def test_truncated_finish_fails_and_close_rejects_further_input(self):
        converter = AudioConverter(AudioFormat())
        converter.feed(b"\x01")
        with self.assertRaises(STTError) as caught:
            converter.finish()
        self.assertEqual(caught.exception.code, "AUDIO_INPUT_TRUNCATED")
        self.assertEqual(converter.state, "failed")
        converter.close()
        converter.close()
        with self.assertRaises(STTError) as caught:
            converter.feed(b"\x00")
        self.assertEqual(caught.exception.code, "CONVERTER_CLOSED")

    def test_shape_failure_can_be_repaired_and_finish_is_idempotent(self):
        converter = AudioConverter(AudioFormat(channels=2))
        with self.assertRaises(STTError):
            converter.feed(np.array([1, 2], np.int16))
        frames = converter.feed(np.array([[100, 100]], np.int16)) + converter.finish()
        self.assertEqual(frames[0].pcm, b"\x64\x00")
        self.assertEqual(converter.finish(), [])
        with self.assertRaises(STTError) as caught:
            converter.feed(np.array([[1, 1]], np.int16))
        self.assertEqual(caught.exception.code, "CONVERTER_FINISHED")

    def test_invalid_format_is_rejected_at_construction(self):
        for options, code in [
            ({"sample_rate": True}, "INVALID_SAMPLE_RATE"),
            ({"channels": 0}, "INVALID_AUDIO_FORMAT"),
        ]:
            with self.subTest(options=options):
                with self.assertRaises(STTError) as caught:
                    AudioConverter(AudioFormat(**options))
                self.assertEqual(caught.exception.code, code)

    def test_native_resample_failure_disallows_finish_and_new_blocks(self):
        converter = AudioConverter(AudioFormat(48000, sample_type="float32"))
        with patch("soxr.ResampleStream.resample_chunk", side_effect=RuntimeError("滤波失败")):
            with self.assertRaises(STTError) as caught:
                converter.feed(np.zeros(480, np.float32))
        self.assertEqual(caught.exception.code, "AUDIO_CONVERSION_FAILED")
        self.assertEqual(converter.state, "failed")
        with self.assertRaises(STTError):
            converter.finish()
        converter.close()
