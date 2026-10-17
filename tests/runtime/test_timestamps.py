"""词时间验证使用真实 SDK 和 Whisper 适配器，替换外部模型返回。"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from realtime_asr import AudioFormat, ModelConfig, Transcriber
from tests.runtime import test_transcriber


class TimestampTests(unittest.TestCase):
    def setUp(self):
        test_transcriber.RecognizerTests.setUp(self)

    def test_reported_word_samples_and_disabled_default(self):
        word = SimpleNamespace(word="你好", start=0.125, end=0.5, probability=0.8)
        segment = SimpleNamespace(text="你好", start=0, end=0.5, words=[word])
        info = SimpleNamespace(language="zh", language_probability=0.9)
        with patch(
            "faster_whisper.BatchedInferencePipeline.transcribe",
            return_value=(iter([segment]), info),
        ):
            with Transcriber(ModelConfig(model=self.path)) as recognizer:
                result = recognizer.transcribe(
                    np.ones(16000, np.int16), format=AudioFormat(), word_timestamps=True
                )
        self.assertEqual(result.timestamp_status, "reported")
        self.assertEqual((result.words[0].start_sample, result.words[0].end_sample), (2000, 8000))
        self.assertEqual(result.words[0].text, "你好")

    def test_out_of_input_word_time_is_unavailable_without_clamping(self):
        word = SimpleNamespace(word="hello", start=0.1, end=2.0, probability=0.8)
        info = SimpleNamespace(language="en", language_probability=0.9)
        with patch(
            "faster_whisper.BatchedInferencePipeline.transcribe",
            return_value=(
                iter([SimpleNamespace(text="hello", start=0, end=1, words=[word])]),
                info,
            ),
        ):
            with Transcriber(ModelConfig(model=self.path)) as recognizer:
                result = recognizer.transcribe(
                    np.ones(16000, np.int16), format=AudioFormat(), word_timestamps=True
                )
        self.assertEqual(result.text, "hello")
        self.assertEqual(result.timestamp_status, "unavailable")
        self.assertIsNone(result.words)
