"""自定义识别引擎是外部边界，音频转换和会话仍执行真实实现。"""

import unittest
from threading import Event

import numpy as np

from speechio import (
    ASRError,
    AudioFormat,
    ModelConfig,
    SessionConfig,
    Transcriber,
    Transcript,
)
from speechio.runtime.asr import ASRRuntime


class RecordingEngine:
    """记录模型边界实际收到的音频，不替换核心转换和调度。"""

    def __init__(self):
        self.preparations = 0
        self.closes = 0
        self.inputs = []

    @property
    def capabilities(self):
        from speechio import EngineCapabilities

        return EngineCapabilities(engine="remote-test", device="remote", snapshot=True)

    def prepare(self):
        self.preparations += 1

    def transcribe(self, audio, language=None, *, role="final", word_timestamps=False):
        self.inputs.append((audio.copy(), language, role, word_timestamps))
        return Transcript(text="收到完整音频", reported_language=language)

    def close(self):
        self.closes += 1


class EngineInjectionTests(unittest.TestCase):
    def test_builtin_engine_names_accept_local_options_and_preserve_config_conflicts(self):
        for name in ("whisper", "nemotron"):
            with self.subTest(engine=name):
                recognizer = Transcriber(
                    model_path="test-model", engine=name, compute_type="default"
                )
                self.addCleanup(recognizer.close)
                self.assertEqual(recognizer.config.engine, name)
                self.assertEqual(recognizer.state, "unprepared")
        with self.assertRaises(ASRError) as caught:
            Transcriber(ModelConfig(engine="whisper", model_path="tiny"), engine="whisper")
        self.assertEqual(caught.exception.code, "INVALID_CONFIGURATION")

    def test_custom_engine_receives_complete_audio_and_owns_one_lifecycle(self):
        engine = RecordingEngine()
        recognizer = Transcriber(engine=engine)
        self.addCleanup(recognizer.close)
        source = np.arange(525, dtype=np.int16)
        result = recognizer.transcribe_chunks(
            (source[:511], source[511:]),
            format=AudioFormat(),
            language="zh-CN",
            word_timestamps=True,
        )
        recognizer.prepare()
        self.assertEqual(result.text, "收到完整音频")
        self.assertEqual(result.requested_language, "zh-CN")
        self.assertEqual(result.timestamp_status, "unavailable")
        np.testing.assert_array_equal(engine.inputs[0][0], source.astype(np.float32) / 32768)
        self.assertEqual(engine.inputs[0][1:], ("zh-CN", "final", False))
        self.assertEqual(engine.preparations, 1)
        first = recognizer.close()
        self.assertIs(recognizer.close(), first)
        self.assertEqual(engine.closes, 1)

    def test_custom_stream_receives_new_audio_and_drains_each_independent_unit(self):
        from speechio import RealtimeConfig

        final = RecordingEngine()
        streaming = RecordingStreamingEngine()
        runtime = ASRRuntime(
            engine=final,
            realtime=RealtimeConfig(
                mode="native", engine=streaming, initial_delay=0, interval=0.01
            ),
        )
        self.addCleanup(runtime.close)
        runtime.prepare()
        session = runtime.create_session(SessionConfig(realtime=True))
        sources = (np.arange(525, dtype=np.int16), -np.arange(513, dtype=np.int16))
        for source in sources:
            session.start()
            session.feed_audio(source[:512])
            session.feed_audio(source[512:])
            session.end_turn()
        completed = 0
        drained = set()
        while completed < 2:
            event = session.next_event(timeout=5)
            self.assertIsNotNone(event)
            if event.kind == "realtime" and event.stream_finished:
                drained.add(event.turn.turn_id)
            if event.kind == "completed":
                completed += 1
        self.assertEqual(drained, {1, 2})
        self.assertEqual(len(streaming.streams), 2)
        for stream, source in zip(streaming.streams, sources, strict=True):
            np.testing.assert_array_equal(
                np.concatenate(stream.inputs), source.astype(np.float32) / 32768
            )
            self.assertEqual(stream.finishes, 1)
            self.assertEqual(stream.closes, 1)
        self.assertEqual(runtime.close().status, "ok")
        self.assertEqual((final.closes, streaming.closes), (1, 1))

    def test_custom_snapshot_uses_fixed_prefix_then_complete_final_input(self):
        from speechio import RealtimeConfig

        engine = RecordingEngine()
        runtime = ASRRuntime(engine=engine, realtime=RealtimeConfig(initial_delay=0))
        self.addCleanup(runtime.close)
        runtime.prepare()
        session = runtime.create_session(SessionConfig(realtime=True, language="zh-CN"))
        source = np.arange(1027, dtype=np.int16)
        session.start()
        session.feed_audio(source[:1024])
        preview = session.next_event(timeout=5)
        self.assertEqual(
            (preview.kind, preview.source, preview.audio_samples), ("realtime", "snapshot", 1024)
        )
        session.feed_audio(source[1024:])
        session.end_turn()
        self.assertEqual(session.next_event(timeout=5).kind, "final")
        self.assertEqual(session.next_event(timeout=5).kind, "completed")
        self.assertEqual(engine.inputs[0][1:], ("zh-CN", "snapshot", False))
        np.testing.assert_array_equal(engine.inputs[0][0], source[:1024].astype(np.float32) / 32768)
        self.assertEqual(engine.inputs[-1][1:], ("zh-CN", "final", False))
        np.testing.assert_array_equal(engine.inputs[-1][0], source.astype(np.float32) / 32768)

    def test_same_injected_instance_is_shared_and_closed_once(self):
        from speechio import RealtimeConfig

        class SharedEngine(RecordingStreamingEngine):
            @property
            def capabilities(self):
                from speechio import EngineCapabilities

                return EngineCapabilities(engine="shared", streaming=True)

        engine = SharedEngine()
        runtime = ASRRuntime(
            engine=engine,
            realtime=RealtimeConfig(mode="native", engine=engine, initial_delay=0),
        )
        self.addCleanup(runtime.close)
        runtime.prepare()
        session = runtime.create_session(SessionConfig(realtime=True))
        session.start()
        source = np.arange(525, dtype=np.int16)
        session.feed_audio(source)
        session.end_turn()
        while True:
            event = session.next_event(timeout=5)
            self.assertIsNotNone(event)
            if event.kind == "completed":
                break
        self.assertEqual(engine.preparations, 1)
        self.assertEqual(len(engine.inputs), 1)
        self.assertEqual(engine.streams[0].finishes, 1)
        np.testing.assert_array_equal(engine.inputs[0][0], source.astype(np.float32) / 32768)
        self.assertEqual(runtime.close().status, "ok")
        self.assertEqual(engine.closes, 1)

    def test_reset_discards_late_result_and_retains_physical_input_until_return(self):
        entered, release = Event(), Event()

        class BlockingEngine(RecordingEngine):
            def transcribe(self, audio, language=None, *, role="final", word_timestamps=False):
                if not self.inputs:
                    entered.set()
                    if not release.wait(5):
                        raise TimeoutError("测试没有释放外部识别调用")
                return super().transcribe(
                    audio, language, role=role, word_timestamps=word_timestamps
                )

        engine = BlockingEngine()
        runtime = ASRRuntime(engine=engine)
        self.addCleanup(runtime.close)
        runtime.prepare()
        session = runtime.create_session()
        source = np.arange(525, dtype=np.int16)
        session.start()
        session.feed_audio(source)
        session.end_turn()
        try:
            self.assertTrue(entered.wait(5))
            change = session.reset()
            self.assertEqual(change.pending_tasks, 1)
            self.assertEqual(runtime.capacity.tasks, 1)
            source.fill(999)
            session.start()
            session.feed_audio(np.full(525, 7000, np.int16))
            session.end_turn()
            self.assertEqual(runtime.capacity.tasks, 2)
        finally:
            release.set()
        events = [session.next_event(timeout=5), session.next_event(timeout=5)]
        self.assertEqual([event.kind for event in events], ["final", "completed"])
        self.assertTrue(all(event.turn.generation == 1 for event in events))
        np.testing.assert_array_equal(engine.inputs[0][0], np.arange(525, dtype=np.float32) / 32768)
        np.testing.assert_array_equal(engine.inputs[1][0], np.full(525, 7000, np.float32) / 32768)
        self.assertEqual(runtime.close().status, "ok")

    def test_invalid_provider_result_is_a_task_failure_instead_of_successful_empty_text(self):
        class InvalidEngine(RecordingEngine):
            def transcribe(self, audio, language=None, *, role="final", word_timestamps=False):
                return "厂商原始字符串"

        runtime = ASRRuntime(engine=InvalidEngine())
        self.addCleanup(runtime.close)
        runtime.prepare()
        session = runtime.create_session()
        session.start()
        session.feed_audio(b"\x01\x00")
        session.end_turn()
        event = session.next_event(timeout=5)
        self.assertEqual(event.kind, "failed")
        self.assertEqual(event.error.code, "BACKEND_INCOMPATIBLE")
        self.assertIsNone(event.result)
        self.assertIsNone(session.next_event())

    def test_external_prepare_failure_closes_owner_and_preserves_cause(self):
        class FailedEngine(RecordingEngine):
            def prepare(self):
                raise ConnectionError("外部服务无法连接")

        engine = FailedEngine()
        runtime = ASRRuntime(engine=engine)
        with self.assertRaises(ASRError) as caught:
            runtime.prepare()
        self.assertIsInstance(caught.exception.__cause__, ConnectionError)
        self.assertEqual(runtime.state, "closed")
        self.assertEqual(runtime.close().status, "ok")
        self.assertEqual(engine.closes, 1)


class RecordingStream:
    """外部流边界记录输入与排空，实时调度仍使用正式实现。"""

    def __init__(self):
        self.inputs = []
        self.finishes = 0
        self.closes = 0

    def accept_audio(self, audio, sample_rate=None):
        self.inputs.append(audio.copy())

    def decode(self):
        pass

    def get_result(self):
        return Transcript(text="实时结果")

    def finish(self):
        self.finishes += 1
        return Transcript(text="完整实时结果")

    def close(self):
        self.closes += 1


class RecordingStreamingEngine(RecordingEngine):
    """只支持原生流式的提供方，可作为独立实时角色。"""

    def __init__(self):
        super().__init__()
        self.streams = []

    @property
    def capabilities(self):
        from speechio import EngineCapabilities

        return EngineCapabilities(engine="remote-stream", transcription=False, streaming=True)

    def create_streaming_session(self, language=None):
        stream = RecordingStream()
        self.streams.append(stream)
        return stream
