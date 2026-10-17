"""双实时走正式会话与执行器，替身仅代表外部整段及流式模型。"""

import unittest
from threading import Event

import numpy as np

import speechio as sdk
from speechio.runtime.asr import ASRRuntime
from tests.runtime.test_engine_injection import RecordingEngine, RecordingStream


class RoleEngine(RecordingEngine):
    """外部模型边界保留实际输入，各流返回不同角色的已知正文。"""

    def __init__(self, text):
        super().__init__()
        self.text = text
        self.streams = []

    @property
    def capabilities(self):
        return sdk.EngineCapabilities(engine="external-role", streaming=True)

    def create_streaming_session(self, language=None):
        text = self.text

        class Stream(RecordingStream):
            def get_result(self):
                return sdk.Transcript(text=text)

            def finish(self):
                self.finishes += 1
                return self.get_result()

        stream = Stream()
        self.streams.append(stream)
        return stream


class DualRealtimeTests(unittest.TestCase):
    def test_reset_and_close_keep_both_physical_calls_until_they_return(self):
        class BlockingRole(RoleEngine):
            def __init__(self, text):
                super().__init__(text)
                self.entered, self.release = Event(), Event()

            def create_streaming_session(self, language=None):
                stream = super().create_streaming_session(language)
                accept = stream.accept_audio

                def blocked(audio, sample_rate=None):
                    self.entered.set()
                    if not self.release.wait(5):
                        raise TimeoutError("外部流没有释放")
                    accept(audio, sample_rate)

                stream.accept_audio = blocked
                return stream

        accurate, fast = BlockingRole("speech system"), BlockingRole("speech system today")
        runtime = ASRRuntime(
            engine=accurate,
            realtime=sdk.RealtimeConfig(mode="native", initial_delay=0, max_units=1),
            ultrafast=sdk.RealtimeConfig(mode="native", engine=fast, initial_delay=0, max_units=1),
        )
        self.addCleanup(runtime.close)
        runtime.prepare()
        session = runtime.create_session(
            sdk.SessionConfig(realtime=True, merge=sdk.RealtimeMergeConfig())
        )
        session.start()
        session.feed_audio(np.ones(512, np.int16))
        try:
            self.assertTrue(accurate.entered.wait(5))
            self.assertTrue(fast.entered.wait(5))
            self.assertEqual(session.reset().pending_tasks, 2)
            self.assertEqual(runtime.capacity.realtime_units, 2)
            with self.assertRaises(sdk.ASRError) as caught:
                session.start()
            self.assertEqual(caught.exception.code, "REALTIME_CAPACITY_EXCEEDED")
            pending = runtime.close(timeout=0)
            self.assertEqual((pending.status, pending.pending_tasks), ("pending", 2))
        finally:
            accurate.release.set()
            fast.release.set()
        self.assertEqual(runtime.close(timeout=5).status, "ok")
        self.assertIsNone(session.next_event())
        self.assertEqual((accurate.closes, fast.closes), (1, 1))

    def test_one_failed_role_does_not_replace_accurate_text_or_fail_final(self):
        class FailedRole(RoleEngine):
            def create_streaming_session(self, language=None):
                stream = super().create_streaming_session(language)

                def failed():
                    raise ConnectionError("极速服务中断")

                stream.get_result = failed
                return stream

        accurate, fast = RoleEngine("speech system"), FailedRole("unused")
        with ASRRuntime(
            engine=accurate,
            realtime=sdk.RealtimeConfig(mode="native", initial_delay=0),
            ultrafast=sdk.RealtimeConfig(mode="native", engine=fast, initial_delay=0),
        ) as runtime:
            session = runtime.create_session(
                sdk.SessionConfig(realtime=True, merge=sdk.RealtimeMergeConfig())
            )
            session.start()
            session.feed_audio(np.ones(525, np.int16))
            session.end_turn()
            events = []
            while not events or events[-1].kind != "completed":
                event = session.next_event(timeout=5)
                self.assertIsNotNone(event)
                events.append(event)
            failure = next(event for event in events if event.kind == "realtime_failed")
            self.assertEqual(
                (failure.realtime_role, failure.error.code), ("ultrafast", "INFERENCE_FAILED")
            )
            final = next(event for event in events if event.kind == "final")
            self.assertEqual(final.result.text, "收到完整音频")
            self.assertEqual(final.merge.text, "speech system")

    def test_start_rejects_all_roles_together_when_fast_capacity_is_full(self):
        accurate, fast = RoleEngine("speech system"), RoleEngine("speech system today")
        with ASRRuntime(
            engine=accurate,
            realtime=sdk.RealtimeConfig(mode="native", max_units=2),
            ultrafast=sdk.RealtimeConfig(mode="native", engine=fast, max_units=1),
        ) as runtime:
            config = sdk.SessionConfig(realtime=True, merge=sdk.RealtimeMergeConfig())
            first, second = runtime.create_session(config), runtime.create_session(config)
            first.start()
            self.assertEqual(runtime.capacity.realtime_units, 2)
            with self.assertRaises(sdk.ASRError) as caught:
                second.start()
            self.assertEqual(caught.exception.code, "REALTIME_CAPACITY_EXCEEDED")
            self.assertEqual(runtime.capacity.realtime_units, 2)
            self.assertEqual(second.state, "idle")

    def test_each_unit_drains_both_roles_and_final_remains_independent(self):
        accurate, fast = RoleEngine("speech system"), RoleEngine("speech system today")
        runtime = ASRRuntime(
            engine=accurate,
            realtime=sdk.RealtimeConfig(mode="native", initial_delay=0),
            ultrafast=sdk.RealtimeConfig(mode="native", engine=fast, initial_delay=0),
        )
        self.addCleanup(runtime.close)
        runtime.prepare()
        session = runtime.create_session(
            sdk.SessionConfig(realtime=True, merge=sdk.RealtimeMergeConfig())
        )
        sources = (np.arange(525, dtype=np.int16), -np.arange(513, dtype=np.int16))
        for source in sources:
            session.start()
            session.feed_audio(source[:512])
            session.feed_audio(source[512:])
            session.end_turn()
        completions, drained, merges, finals = 0, {"accurate": set(), "ultrafast": set()}, [], []
        while completions < 2:
            event = session.next_event(timeout=5)
            self.assertIsNotNone(event)
            if event.kind == "realtime" and event.stream_finished:
                drained[event.realtime_role].add(event.turn.turn_id)
                merges.append(event.merge.text)
            if event.kind == "final":
                finals.append(event.result.text)
            completions += int(event.kind == "completed")
        self.assertEqual(drained, {"accurate": {1, 2}, "ultrafast": {1, 2}})
        self.assertIn("speech system today", merges)
        self.assertEqual(finals, ["收到完整音频", "收到完整音频"])
        for engine in (accurate, fast):
            for stream, source in zip(engine.streams, sources, strict=True):
                np.testing.assert_array_equal(
                    np.concatenate(stream.inputs), source.astype(np.float32) / 32768
                )
                self.assertEqual((stream.finishes, stream.closes), (1, 1))
        self.assertEqual(runtime.close().status, "ok")
        self.assertEqual((accurate.preparations, fast.preparations), (1, 1))
        self.assertEqual((accurate.closes, fast.closes), (1, 1))
