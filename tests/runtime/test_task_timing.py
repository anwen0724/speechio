"""通过公开运行时事件检查实际任务的提交和执行时间。"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from speechio import ModelConfig
from speechio.runtime.asr import ASRRuntime
from tests.runtime import test_transcriber


class TaskTimingTests(unittest.TestCase):
    def setUp(self):
        test_transcriber.RecognizerTests.setUp(self)

    def test_final_reports_submission_and_execution_times(self):
        result = (
            iter([SimpleNamespace(text="result", start=0, end=0.032)]),
            SimpleNamespace(language="en", language_probability=1),
        )
        with patch("faster_whisper.BatchedInferencePipeline.transcribe", return_value=result):
            with ASRRuntime(ModelConfig(model=self.path)) as runtime:
                session = runtime.create_session()
                session.start()
                session.feed_audio(b"\x01\x00" * 512)
                session.end_turn()
                event = session.next_event(timeout=5)
                self.assertEqual(event.kind, "final")
                self.assertIsNotNone(getattr(event, "submitted_at_monotonic", None))
                self.assertLessEqual(event.submitted_at_monotonic, event.started_at_monotonic)
                self.assertLessEqual(event.started_at_monotonic, event.completed_at_monotonic)
                session.close()
