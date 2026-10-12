"""通过公开运行时事件检查实际任务的提交和执行时间。"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from realtime_asr import ASRRuntime, RecognizerConfig
from tests.runtime import test_recognizer


class TaskTimingTests(unittest.TestCase):
    def setUp(self):
        test_recognizer.RecognizerTests.setUp(self)

    def test_final_reports_submission_and_execution_times(self):
        result = (
            iter([SimpleNamespace(text="result", start=0, end=0.032)]),
            SimpleNamespace(language="en", language_probability=1),
        )
        with patch("faster_whisper.BatchedInferencePipeline.transcribe", return_value=result):
            with ASRRuntime(RecognizerConfig(model_path=self.path)) as runtime:
                session = runtime.create_session()
                session.start()
                session.feed_audio(b"\x01\x00" * 512)
                session.finish()
                event = session.get_event(timeout=5)
                self.assertEqual(event.kind, "final")
                self.assertIsNotNone(getattr(event, "submitted_at_monotonic", None))
                self.assertLessEqual(event.submitted_at_monotonic, event.started_at_monotonic)
                self.assertLessEqual(event.started_at_monotonic, event.completed_at_monotonic)
                session.close()
