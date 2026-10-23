"""默认完成使用流式结束结果，失败与失效结果不触发额外识别。"""

import unittest
from threading import Event

from speechio import RealtimeConfig, SessionConfig, STTError, Transcript
from speechio.stt.runtime.asr import STTRuntime
from tests.runtime.test_engine_injection import (
    RecordingEngine,
    RecordingStream,
    RecordingStreamingEngine,
)


class StreamCompletionTests(unittest.TestCase):
    def open_session(self, stream):
        class Engine(RecordingStreamingEngine):
            def create_streaming_session(self, language=None):
                self.streams.append(stream)
                return stream

        main = RecordingEngine()
        runtime = STTRuntime.from_adapter(
            adapter=main,
            realtime=RealtimeConfig.from_adapter(
                mode="streaming", adapter=Engine(), initial_delay_seconds=0
            ),
        )
        self.addCleanup(runtime.close)
        runtime.prepare()
        session = runtime.create_session(SessionConfig(realtime=True))
        session.start()
        session.feed_audio(bytes(1050))
        return runtime, session, main

    def test_empty_success_is_final_without_retry(self):
        class EmptyStream(RecordingStream):
            def finish(self):
                return Transcript(text="")

        _, session, main = self.open_session(EmptyStream())
        session.end_turn()
        while (event := session.next_event(timeout=5)).kind == "realtime":
            pass
        self.assertEqual((event.kind, event.result.text), ("final", ""))
        self.assertEqual(session.next_event(timeout=5).kind, "completed")
        self.assertEqual(main.inputs, [])

    def test_failure_before_or_during_finish_never_falls_back(self):
        for stage in ("decode", "finish"):
            with self.subTest(stage=stage):

                class FailedStream(RecordingStream):
                    def decode(self):
                        if stage == "decode":
                            raise STTError("INFERENCE_FAILED", "decode", "服务断开")

                    def finish(self):
                        raise STTError("REMOTE_TIMEOUT", "finish", "结束确认超时")

                runtime, session, main = self.open_session(FailedStream())
                if stage == "decode":
                    self.assertEqual(session.next_event(timeout=5).kind, "realtime_failed")
                session.end_turn()
                while (event := session.next_event(timeout=5)).kind.startswith("realtime"):
                    pass
                self.assertEqual(event.kind, "failed")
                self.assertIsNone(session.next_event())
                self.assertEqual(main.inputs, [])
                self.assertEqual(runtime.close().status, "ok")

    def test_cancel_or_reset_discards_late_finish(self):
        for action in ("cancel_turn", "reset"):
            with self.subTest(action=action):
                entered, release = Event(), Event()

                class BlockingStream(RecordingStream):
                    def finish(self):
                        entered.set()
                        if not release.wait(5):
                            raise TimeoutError("测试没有释放流式结束调用")
                        return Transcript(text="不能交付的旧结果")

                runtime, session, main = self.open_session(BlockingStream())
                task = session.end_turn()
                try:
                    self.assertTrue(entered.wait(5))
                    self.assertGreater(runtime.pending_for(session.session_id), 0)
                    if action == "cancel_turn":
                        self.assertTrue(session.cancel_turn(task.turn))
                        self.assertEqual(session.next_event(timeout=5).kind, "cancelled")
                    else:
                        self.assertEqual(len(session.reset().cancelled), 1)
                finally:
                    release.set()
                with runtime.coordination:
                    self.assertTrue(
                        runtime.coordination.wait_for(
                            lambda: runtime.pending_for(session.session_id) == 0, timeout=5
                        )
                    )
                self.assertIsNone(session.next_event())
                self.assertEqual(main.inputs, [])
