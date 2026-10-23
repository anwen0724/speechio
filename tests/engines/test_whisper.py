"""分段迭代器是运行库边界；结果不得止于第一段。"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from speechio.errors import STTError
from tests.support import whisper_engine


class WhisperTests(unittest.TestCase):
    def test_all_lazy_segments_and_real_language_are_preserved(self):
        segments = iter(
            [
                SimpleNamespace(text=" 你好", start=0.1, end=0.4),
                SimpleNamespace(text=" 世界", start=0.4, end=0.8),
            ]
        )
        info = SimpleNamespace(language="zh", language_probability=0.9)
        with whisper_engine() as engine:
            with patch(
                "faster_whisper.BatchedInferencePipeline.transcribe", return_value=(segments, info)
            ):
                result = engine.transcribe([0.1], language="zh")
        self.assertEqual(result.text, "你好  世界")
        self.assertEqual([item["text"] for item in result.segments], [" 你好", " 世界"])
        self.assertEqual(result.reported_language, "zh")
        self.assertEqual(result.language_probability, 0.9)
        self.assertEqual(result.language_source, "detected")

    def test_late_iterator_failure_is_not_empty_success(self):
        def segments():
            yield SimpleNamespace(text="第一段", start=0, end=1)
            raise RuntimeError("后续分段失败")

        info = SimpleNamespace(language=None, language_probability=None)
        with whisper_engine() as engine:
            with patch(
                "faster_whisper.BatchedInferencePipeline.transcribe",
                return_value=(segments(), info),
            ):
                with self.assertRaises(STTError) as caught:
                    engine.transcribe([0.1])
        self.assertEqual(caught.exception.code, "INFERENCE_FAILED")
