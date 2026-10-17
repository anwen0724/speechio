"""公开实时会话行为；替身仅用于外部模型和时钟边界。"""

import unittest
from threading import Event, Lock
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from realtime_asr import (
    ASRError,
    AudioFormat,
    ModelConfig,
    RealtimeConfig,
    RuntimeLimits,
    SessionConfig,
)
from realtime_asr.runtime.asr import ASRRuntime
from realtime_asr.runtime.models.manifest import create_manifest
from realtime_asr.runtime.models.pinned import NEMOTRON_FILES
from tests.engines.test_stream import RuntimeRecognizer, RuntimeStream
from tests.runtime import test_transcriber


class RealtimeTests(unittest.TestCase):
    def setUp(self):
        test_transcriber.RecognizerTests.setUp(self)

    def test_acoustic_followup_observes_same_audio_without_stabilizing_duplicate_evidence(self):
        from realtime_asr import StabilizationConfig

        voice = np.sin(2 * np.pi * 180 * np.arange(3200) / 16000)
        audio = (
            np.concatenate((np.zeros(1600), voice * 0.2, voice[:320] * 0.005, voice * 0.2)) * 32767
        ).astype(np.int16)
        result = SimpleNamespace(text="测试文本", start=0, end=0.5)
        info = SimpleNamespace(language="zh", language_probability=1)
        with patch(
            "faster_whisper.BatchedInferencePipeline.transcribe",
            side_effect=lambda *a, **kw: (iter([result]), info),
        ):
            with ASRRuntime(
                ModelConfig(model=self.path),
                realtime=RealtimeConfig(
                    initial_delay=0,
                    interval=60,
                    acoustic_boundaries=True,
                    boundary_followup_delays=(0.01, 0.05),
                    boundary_fallback_interval=60,
                ),
            ) as runtime:
                session = runtime.create_session(
                    SessionConfig(
                        realtime=True, stabilization=StabilizationConfig(initial_prefix_span=0)
                    )
                )
                session.start()
                session.feed_audio(audio)
                first = session.next_event(timeout=5)
                followup = session.next_event(timeout=5)
                self.assertEqual(first.trigger_reason, "acoustic-boundary")
                self.assertEqual(followup.trigger_reason, "boundary-followup")
                self.assertEqual(first.audio_version, followup.audio_version)
                self.assertEqual(followup.stabilization.stable_text, "")
                self.assertGreater(followup.observation_sequence, first.observation_sequence)
                session.cancel_turn()

    def nemotron_fixture(self, native):
        """临时模型使用合成文件身份，替换外部 Sherpa 工厂，运行真实流适配器。"""
        folder = self.path / "nemotron"
        folder.mkdir()
        files = {name: None for name in NEMOTRON_FILES}
        for name in files:
            (folder / name).write_bytes(b"test-model")
        identity = patch.dict(NEMOTRON_FILES, files, clear=True)
        identity.start()
        self.addCleanup(identity.stop)
        create_manifest(folder, backend="nemotron", model_id="test", source="test", revision="1")
        factory = patch("sherpa_onnx.OnlineRecognizer.from_transducer", return_value=native)
        factory.start()
        self.addCleanup(factory.stop)
        return ModelConfig(model=folder, engine="nemotron")

    def test_native_only_new_samples_and_short_tail_then_separate_whole_final(self):
        streams = []

        class Native(RuntimeRecognizer):
            def create_stream(self):
                stream = RuntimeStream()
                streams.append(stream)
                return stream

        config = self.nemotron_fixture(Native())
        with ASRRuntime(
            config, realtime=RealtimeConfig(mode="native", initial_delay=0, interval=10)
        ) as runtime:
            session = runtime.create_session(SessionConfig(realtime=True, language="zh-CN"))
            session.start()
            session.feed_audio(np.arange(512, dtype=np.int16))
            first = session.next_event(timeout=1)
            self.assertIsNotNone(first, "原生流没有实时观察")
            self.assertEqual((first.source, first.audio_samples), ("native", 512))
            session.feed_audio(np.array([1001, 2002, 3003], np.int16))
            task = session.end_turn()
            self.assertEqual(session.end_turn(), task)
            ended = session.next_event(timeout=5)
            self.assertTrue(ended.stream_finished)
            self.assertEqual(ended.audio_samples, 515)
            final = session.next_event(timeout=5)
            self.assertEqual((final.kind, final.audio_samples), ("final", 515))
            self.assertEqual(session.next_event(timeout=5).kind, "completed")
            self.assertEqual(len(streams), 2)
            expected = np.concatenate((np.arange(512), [1001, 2002, 3003])) / 32768
            for stream in streams:
                self.assertTrue(stream.ended)
                np.testing.assert_array_equal(stream.samples, expected)

    def test_native_finish_includes_resampler_flush_and_preserves_whole_audio(self):
        streams = []

        class Native(RuntimeRecognizer):
            def create_stream(self):
                stream = RuntimeStream()
                streams.append(stream)
                return stream

        with ASRRuntime(
            self.nemotron_fixture(Native()),
            realtime=RealtimeConfig(mode="native", initial_delay=10),
        ) as runtime:
            session = runtime.create_session(
                SessionConfig(realtime=True, format=AudioFormat(48000, 2, "float32"))
            )
            session.start()
            audio = np.full((4800, 2), 0.1, np.float32)
            for offset in range(0, len(audio), 317):
                session.feed_audio(audio[offset : offset + 317])
            session.end_turn()
            ended = session.next_event(timeout=5)
            self.assertTrue(ended.stream_finished)
            self.assertEqual(ended.audio_samples, 1600)
            self.assertEqual(session.next_event(timeout=5).kind, "final")
            self.assertEqual(session.next_event(timeout=5).kind, "completed")
            self.assertEqual([len(s.samples) for s in streams], [1600, 1600])
            np.testing.assert_array_equal(streams[0].samples, streams[1].samples)

    def test_independent_realtime_model_progresses_while_final_is_blocked(self):
        entered, release = Event(), Event()
        self.addCleanup(release.set)

        def native(audio, **kwargs):
            if kwargs["beam_size"] == 5:
                entered.set()
                release.wait(5)
            return iter([SimpleNamespace(text="text", start=0, end=1)]), SimpleNamespace(
                language="en", language_probability=1
            )

        with patch("faster_whisper.BatchedInferencePipeline.transcribe", side_effect=native):
            with ASRRuntime(
                ModelConfig(model=self.path),
                realtime=RealtimeConfig(model=ModelConfig(model=self.path), initial_delay=0),
            ) as runtime:
                final_session = runtime.create_session()
                final_session.start()
                final_session.feed_audio(np.ones(1024, np.int16))
                final_session.end_turn()
                self.assertTrue(entered.wait(5))
                live = runtime.create_session(SessionConfig(realtime=True))
                live.start()
                live.feed_audio(np.ones(1024, np.int16))
                observation = live.next_event(timeout=1)
                release.set()
                self.assertIsNotNone(observation, "独立实时模型被最终角色阻塞")
                self.assertEqual(observation.kind, "realtime")

    def test_cancelled_snapshot_retains_physical_capacity_until_call_returns(self):
        entered, release = Event(), Event()
        self.addCleanup(release.set)

        def native(audio, **kwargs):
            entered.set()
            release.wait(5)
            return iter([]), SimpleNamespace(language=None, language_probability=None)

        with patch("faster_whisper.BatchedInferencePipeline.transcribe", side_effect=native):
            runtime = ASRRuntime(
                ModelConfig(model=self.path),
                realtime=RealtimeConfig(initial_delay=0, max_units=1, max_audio_samples=2048),
            )
            runtime.prepare()
            self.addCleanup(runtime.close)
            session = runtime.create_session(SessionConfig(realtime=True, max_audio_samples=2048))
            unit = session.start()
            session.feed_audio(np.ones(1024, np.int16))
            self.assertTrue(entered.wait(5))
            change = session.reset()
            self.assertEqual(change.pending_tasks, 1)
            self.assertEqual(runtime.capacity.realtime_units, 1)
            self.assertEqual(runtime.capacity.realtime_reserved_samples, 2048)
            with self.assertRaises(ASRError) as caught:
                session.start()
            self.assertEqual(caught.exception.code, "REALTIME_CAPACITY_EXCEEDED")
            closed = runtime.close(timeout=0)
            self.assertEqual((closed.status, closed.pending_tasks), ("pending", 1))
            release.set()
            self.assertEqual(runtime.close(timeout=5).status, "ok")
            self.assertEqual(runtime.capacity.realtime_units, 0)
            self.assertEqual(change.cancelled[0].turn, unit)
            self.assertIsNone(session.next_event())

    def test_native_finish_during_decode_preserves_new_increment_and_closes_stream(self):
        entered, release = Event(), Event()
        streams = []
        self.addCleanup(release.set)

        class Stream(RuntimeStream):
            closed = False

            def close(self):
                self.closed = True

        class Native(RuntimeRecognizer):
            def create_stream(self):
                stream = Stream()
                streams.append(stream)
                return stream

            def get_result_all(self, stream):
                if stream is streams[0] and not stream.ended:
                    entered.set()
                    release.wait(5)
                return super().get_result_all(stream)

        config = self.nemotron_fixture(Native())
        with ASRRuntime(config, realtime=RealtimeConfig(mode="native", initial_delay=0)) as runtime:
            session = runtime.create_session(SessionConfig(realtime=True))
            session.start()
            session.feed_audio(np.ones(512, np.int16))
            self.assertTrue(entered.wait(5))
            session.feed_audio(np.array([1001, 2002, 3003], np.int16))
            session.end_turn()
            release.set()
            events = []
            while not events or events[-1].kind != "completed":
                event = session.next_event(timeout=5)
                self.assertIsNotNone(event)
                events.append(event)
            self.assertTrue(any(e.stream_finished and e.audio_samples == 515 for e in events))
            self.assertEqual(len(streams[0].samples), 515)
            self.assertTrue(all(s.closed for s in streams))
            self.assertEqual(runtime.capacity.realtime_units, 0)

    def test_realtime_error_does_not_end_input_or_hide_ordinary_final(self):
        def native(audio, **kwargs):
            if kwargs["beam_size"] == 3:
                raise RuntimeError("外部快照推理失败")
            return iter([SimpleNamespace(text="final", start=0, end=1)]), SimpleNamespace(
                language="en", language_probability=1
            )

        with patch("faster_whisper.BatchedInferencePipeline.transcribe", side_effect=native):
            with ASRRuntime(
                ModelConfig(model=self.path), realtime=RealtimeConfig(initial_delay=0)
            ) as runtime:
                session = runtime.create_session(SessionConfig(realtime=True))
                session.start()
                session.feed_audio(np.ones(1024, np.int16))
                failure = session.next_event(timeout=5)
                self.assertEqual(
                    (failure.kind, failure.error.code), ("realtime_failed", "INFERENCE_FAILED")
                )
                self.assertEqual(session.state, "receiving")
                session.feed_audio(np.ones(512, np.int16))
                session.end_turn()
                self.assertEqual(session.next_event(timeout=5).result.text, "final")
                self.assertEqual(session.next_event(timeout=5).kind, "completed")

    def test_second_model_prepare_failure_returns_to_unprepared_and_can_retry(self):
        model = SimpleNamespace(feature_extractor=SimpleNamespace(chunk_length=30))
        with patch(
            "faster_whisper.WhisperModel",
            side_effect=[model, RuntimeError("第二模型失败"), model, model],
        ):
            runtime = ASRRuntime(
                ModelConfig(model=self.path),
                realtime=RealtimeConfig(model=ModelConfig(model=self.path)),
            )
            self.addCleanup(runtime.close)
            with self.assertRaises(ASRError) as caught:
                runtime.prepare()
            self.assertEqual(caught.exception.code, "MODEL_LOAD")
            self.assertEqual(runtime.state, "unprepared")
            runtime.prepare()
            self.assertEqual(runtime.state, "ready")

    def test_snapshot_time_threshold_and_no_duplicate_version(self):
        clock = [0.0]
        native = (iter([]), SimpleNamespace(language=None, language_probability=None))
        with (
            patch("realtime_asr.runtime.realtime.monotonic", side_effect=lambda: clock[0]),
            patch(
                "faster_whisper.BatchedInferencePipeline.transcribe",
                side_effect=lambda *a, **k: (iter([]), native[1]),
            ),
        ):
            with ASRRuntime(ModelConfig(model=self.path), realtime=RealtimeConfig()) as runtime:
                session = runtime.create_session(SessionConfig(realtime=True))
                session.start()
                session.feed_audio(np.ones(512, np.int16))
                self.assertIsNone(session.next_event(timeout=0.02))

                session.feed_audio(np.ones(512, np.int16))
                self.assertIsNone(session.next_event(timeout=0.02))
                clock[0] = 0.2
                session.feed_audio(b"")
                first = session.next_event(timeout=5)
                self.assertEqual(first.started_at_monotonic, 0.2)
                clock[0] = 0.3
                session.feed_audio(np.ones(512, np.int16))
                self.assertIsNone(session.next_event(timeout=0.02))
                clock[0] = 0.4
                session.feed_audio(b"")
                second = session.next_event(timeout=5)
                self.assertEqual(second.audio_samples, 1536)
                clock[0] = 1
                session.feed_audio(b"")
                self.assertIsNone(session.next_event(timeout=0.02))

    def test_native_sealed_retry_does_not_repeat_stream_end(self):
        entered, release = Event(), Event()
        self.addCleanup(release.set)
        streams = []

        class Native(RuntimeRecognizer):
            def create_stream(self):
                stream = RuntimeStream()
                streams.append(stream)
                return stream

            def get_result_all(self, stream):
                if stream is streams[0]:
                    entered.set()
                    release.wait(5)
                return super().get_result_all(stream)

        config = self.nemotron_fixture(Native())
        with ASRRuntime(
            config,
            limits=RuntimeLimits(max_tasks=1),
            realtime=RealtimeConfig(mode="native", initial_delay=0),
        ) as runtime:
            blocker = runtime.create_session()
            blocker.start()
            blocker.feed_audio(np.ones(512, np.int16))
            blocker.end_turn()
            self.assertTrue(entered.wait(5))
            live = runtime.create_session(SessionConfig(realtime=True))
            unit = live.start()
            live.feed_audio(np.ones(515, np.int16))
            with self.assertRaises(ASRError) as caught:
                live.end_turn()
            self.assertEqual(caught.exception.code, "TASK_CAPACITY_EXCEEDED")
            self.assertEqual(live.state, "sealed")
            release.set()
            self.assertEqual(blocker.next_event(timeout=5).kind, "final")
            ended = live.next_event(timeout=5)
            self.assertEqual((ended.turn, ended.stream_finished), (unit, True))
            live.end_turn()
            self.assertEqual(live.next_event(timeout=5).kind, "final")
            self.assertEqual(live.next_event(timeout=5).kind, "completed")
            self.assertEqual(len(streams), 3)
            self.assertEqual([len(s.samples) for s in streams], [512, 515, 515])

    def test_native_failure_preserves_inference_error_and_records_cleanup_failure(self):
        streams = []

        class FailingStream(RuntimeStream):
            def close(self):
                raise RuntimeError("原生关闭失败")

        class Native(RuntimeRecognizer):
            def create_stream(self):
                stream = FailingStream() if not streams else RuntimeStream()
                streams.append(stream)
                return stream

            def get_result_all(self, stream):
                if stream is streams[0]:
                    raise RuntimeError("结果读取失败")
                return super().get_result_all(stream)

        runtime = ASRRuntime(
            self.nemotron_fixture(Native()), realtime=RealtimeConfig(mode="native", initial_delay=0)
        )
        runtime.prepare()
        self.addCleanup(runtime.close)
        session = runtime.create_session(SessionConfig(realtime=True))
        session.start()
        session.feed_audio(np.ones(512, np.int16))
        error = session.next_event(timeout=5)
        self.assertEqual((error.kind, error.error.operation), ("realtime_failed", "get_result"))
        session.end_turn()
        self.assertEqual(session.next_event(timeout=5).kind, "final")
        self.assertEqual(session.next_event(timeout=5).kind, "completed")
        fresh = session.start()
        session.feed_audio(np.ones(512, np.int16))
        observation = session.next_event(timeout=5)
        self.assertEqual((observation.kind, observation.turn), ("realtime", fresh))
        closed = runtime.close()
        self.assertEqual(closed.status, "failed")
        self.assertEqual(closed.errors[0].code, "CLEANUP_FAILED")

    def test_native_capacity_refusal_preserves_next_unit_id_and_recovers_after_drain(self):
        config = self.nemotron_fixture(RuntimeRecognizer())
        with ASRRuntime(
            config,
            realtime=RealtimeConfig(
                mode="native", initial_delay=0, max_units=1, max_audio_samples=1024
            ),
        ) as runtime:
            first = runtime.create_session(SessionConfig(realtime=True, max_audio_samples=1024))
            second = runtime.create_session(SessionConfig(realtime=True, max_audio_samples=1024))
            first.start()
            with self.assertRaises(ASRError) as caught:
                second.start()
            self.assertEqual(caught.exception.code, "REALTIME_CAPACITY_EXCEEDED")
            self.assertEqual(second.state, "idle")
            first.feed_audio(np.ones(515, np.int16))
            first.end_turn()
            while (event := first.next_event(timeout=5)).kind != "completed":
                self.assertNotEqual(event.kind, "failed")
            self.assertEqual(runtime.capacity.realtime_units, 0)
            self.assertEqual(second.start().turn_id, 1)

    def test_slow_consumer_keeps_only_latest_observation_and_protects_final_pair(self):
        third_entered, release = Event(), Event()
        self.addCleanup(release.set)
        # 原生边界的不同累计范围驱动三次观察，无需猜测后台进度。
        streams = []
        first_entered, second_entered = Event(), Event()
        first_release, second_release = Event(), Event()
        self.addCleanup(first_release.set)
        self.addCleanup(second_release.set)

        class Native(RuntimeRecognizer):
            def create_stream(self):
                stream = RuntimeStream()
                streams.append(stream)
                return stream

            def get_result_all(self, stream):
                if stream is streams[0] and not stream.ended:
                    gate = {
                        512: (first_entered, first_release),
                        1024: (second_entered, second_release),
                        1536: (third_entered, release),
                    }.get(len(stream.samples))
                    if gate:
                        gate[0].set()
                        gate[1].wait(5)
                return super().get_result_all(stream)

        with ASRRuntime(
            self.nemotron_fixture(Native()),
            realtime=RealtimeConfig(mode="native", initial_delay=0, interval=0.001),
        ) as runtime:
            live = runtime.create_session(SessionConfig(realtime=True))
            live.start()
            live.feed_audio(np.ones(512, np.int16))
            self.assertTrue(first_entered.wait(5))
            live.feed_audio(np.ones(512, np.int16))
            first_release.set()
            self.assertTrue(second_entered.wait(5))
            live.feed_audio(np.ones(512, np.int16))
            second_release.set()
            self.assertTrue(third_entered.wait(5))
            latest = live.next_event(timeout=5)
            self.assertEqual((latest.audio_samples, latest.sequence), (1024, 1))
            live.end_turn()
            release.set()
            events = []
            while not events or events[-1].kind != "final":
                events.append(live.next_event(timeout=5))
            live.start()
            live.feed_audio(np.ones(512, np.int16))
            self.assertEqual(live.next_event(timeout=5).kind, "completed")

    def test_shared_model_runs_waiting_final_before_next_periodic_snapshot(self):
        snapshot_entered, snapshot_release, final_entered, final_release = (
            Event() for _ in range(4)
        )
        self.addCleanup(snapshot_release.set)
        self.addCleanup(final_release.set)
        boundary = Lock()
        received = []

        def native(audio, **kwargs):
            if not boundary.acquire(blocking=False):
                raise RuntimeError("同一外部模型被并发访问")
            try:
                received.append(kwargs["beam_size"])
                if len(received) == 1:
                    snapshot_entered.set()
                    snapshot_release.wait(5)
                elif kwargs["beam_size"] == 5:
                    final_entered.set()
                    final_release.wait(5)
                return iter([]), SimpleNamespace(language=None, language_probability=None)
            finally:
                boundary.release()

        with patch("faster_whisper.BatchedInferencePipeline.transcribe", side_effect=native):
            with ASRRuntime(
                ModelConfig(model=self.path), realtime=RealtimeConfig(initial_delay=0)
            ) as runtime:
                session = runtime.create_session(SessionConfig(realtime=True))
                first = session.start()
                session.feed_audio(np.ones(1024, np.int16))
                self.assertTrue(snapshot_entered.wait(5))
                session.end_turn()
                second = session.start()
                session.feed_audio(np.ones(1024, np.int16))
                snapshot_release.set()
                self.assertTrue(final_entered.wait(5))
                self.assertIsNone(session.next_event(timeout=0.02))
                final_release.set()
                self.assertEqual(session.next_event(timeout=5).turn, first)
                self.assertEqual(session.next_event(timeout=5).kind, "completed")
                observation = session.next_event(timeout=5)
                self.assertEqual((observation.turn, observation.kind), (second, "realtime"))
                self.assertEqual(received, [3, 5, 3])

    def test_slow_snapshot_keeps_fixed_audio_and_skips_obsolete_pending_versions(self):
        entered, release, second_entered, second_release = (Event() for _ in range(4))
        self.addCleanup(release.set)
        self.addCleanup(second_release.set)
        received = []

        def native(audio, **kwargs):
            received.append((len(audio), kwargs["beam_size"]))
            if len(received) == 1:
                entered.set()
                release.wait(5)
            elif kwargs["beam_size"] == 3:
                second_entered.set()
                second_release.wait(5)
            return iter([SimpleNamespace(text=str(len(audio)), start=0, end=1)]), SimpleNamespace(
                language="en", language_probability=1
            )

        with patch("faster_whisper.BatchedInferencePipeline.transcribe", side_effect=native):
            with ASRRuntime(
                ModelConfig(model=self.path),
                realtime=RealtimeConfig(initial_delay=0, interval=0.001),
            ) as runtime:
                session = runtime.create_session(SessionConfig(realtime=True))
                session.start()
                session.feed_audio(np.ones(1024, np.int16))
                self.assertTrue(entered.wait(5))
                for _ in range(3):
                    session.feed_audio(np.ones(1024, np.int16))
                release.set()
                self.assertTrue(second_entered.wait(5))
                self.assertEqual(session.next_event(timeout=5).audio_samples, 1024)
                # 结束时使在途快照失效，普通最终识别仍读取完整音频。
                session.end_turn()
                second_release.set()
                final = session.next_event(timeout=5)
                self.assertEqual((final.kind, final.audio_samples), ("final", 4096))
                self.assertEqual(session.next_event(timeout=5).kind, "completed")
                self.assertIsNone(session.next_event())
                self.assertEqual(received, [(1024, 3), (4096, 3), (4096, 5)])

    def test_snapshot_observation_precedes_independent_final_with_distinct_parameters(self):
        received = []

        def native(audio, **kwargs):
            received.append((audio.copy(), kwargs["beam_size"]))
            text = "partial" if kwargs["beam_size"] == 3 else "final correction"
            return iter([SimpleNamespace(text=text, start=0, end=1)]), SimpleNamespace(
                language="en", language_probability=1
            )

        with patch("faster_whisper.BatchedInferencePipeline.transcribe", side_effect=native):
            with ASRRuntime(
                ModelConfig(model=self.path),
                realtime=RealtimeConfig(initial_delay=0, interval=10),
            ) as runtime:
                session = runtime.create_session(SessionConfig(realtime=True, language="en"))
                unit = session.start()
                session.feed_audio(np.full(1024, 100, np.int16))
                observation = session.next_event(timeout=1)
                self.assertIsNotNone(observation, "接收期间没有实时观察")
                self.assertEqual((observation.kind, observation.source), ("realtime", "snapshot"))
                self.assertEqual((observation.turn, observation.audio_samples), (unit, 1024))
                self.assertEqual(observation.result.text, "partial")
                self.assertGreaterEqual(
                    observation.completed_at_monotonic, observation.started_at_monotonic
                )
                session.feed_audio(np.array([200, 300, 400], np.int16))
                session.end_turn()
                final = session.next_event(timeout=5)
                self.assertEqual((final.kind, final.source), ("final", "final"))
                self.assertEqual(final.result.text, "final correction")
                self.assertEqual(final.audio_samples, 1027)
                self.assertEqual(session.next_event(timeout=5).kind, "completed")
                self.assertEqual([beam for _, beam in received], [3, 5])
                np.testing.assert_array_equal(
                    received[-1][0][-3:], np.array([200, 300, 400]) / 32768
                )
