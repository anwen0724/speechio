"""默认完成使用流式结束结果，失败与失效结果不触发额外识别。"""

import unittest
from threading import Event

from speechio import RealtimeConfig, SessionConfig, SpeechEndConfig, STTError, Transcript
from speechio.stt.runtime.asr import STTRuntime
from tests.runtime.test_engine_injection import (
    RecordingEngine,
    RecordingStream,
    RecordingStreamingEngine,
)
from tests.test_api import SpeechDetector


class StreamCompletionTests(unittest.TestCase):
    def test_stream_only_adapter_completes_manual_and_automatic_sessions(self):
        for automatic in (False, True):
            with self.subTest(automatic=automatic):
                engine = RecordingStreamingEngine()
                # 契约允许不提供整段方法；完成不能依赖隐藏的整段调用。
                engine.transcribe = None
                with STTRuntime.from_adapter(
                    engine, realtime=RealtimeConfig(mode="streaming", initial_delay_seconds=0)
                ) as runtime:
                    config = SessionConfig(realtime=True)
                    session = (
                        runtime.create_auto_session_from_detector(SpeechDetector, config=config)
                        if automatic
                        else runtime.create_session(config)
                    )
                    if not automatic:
                        session.start()
                    session.feed_audio(b"\x01\x00" * 16000)
                    if automatic:
                        session.end_input()
                    else:
                        session.end_turn()
                    finals = []
                    while True:
                        event = session.next_event(timeout=2)
                        self.assertIsNotNone(event)
                        self.assertNotIn(event.kind, ("failed", "realtime_failed"))
                        if event.kind == "final":
                            finals.append(event)
                        if event.kind == "completed":
                            break
                    self.assertEqual(len(finals), 1)
                    self.assertEqual(finals[0].source, "streaming")
                    self.assertEqual(finals[0].result.text, "完整实时结果")
                    self.assertEqual(finals[0].audio_samples, 16000)
                    self.assertEqual(engine.inputs, [])
                self.assertEqual(engine.closes, 1)

    def test_stream_only_adapter_rejects_whole_audio_completion_before_input(self):
        engine = RecordingStreamingEngine()
        with STTRuntime.from_adapter(engine, realtime=RealtimeConfig(mode="streaming")) as runtime:
            for config in (
                SessionConfig(),
                SessionConfig(realtime=True, completion_mode="final"),
            ):
                with self.subTest(config=config):
                    with self.assertRaises(STTError) as caught:
                        runtime.create_session(config)
                    self.assertEqual(caught.exception.code, "BACKEND_INCOMPATIBLE")
            self.assertEqual(runtime.capacity.open_sessions, 0)
            self.assertEqual(engine.inputs, [])

    def test_stream_only_adapter_rejects_early_whole_audio_recognition(self):
        with STTRuntime.from_adapter(
            RecordingStreamingEngine(), realtime=RealtimeConfig(mode="streaming")
        ) as runtime:
            with self.assertRaises(STTError) as caught:
                runtime.create_auto_session_from_detector(
                    SpeechDetector,
                    config=SessionConfig(realtime=True),
                    endpoint=SpeechEndConfig(early_transcription_silence=0.1),
                )
            self.assertEqual(caught.exception.code, "BACKEND_INCOMPATIBLE")
            self.assertEqual(runtime.capacity.open_sessions, 0)

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
