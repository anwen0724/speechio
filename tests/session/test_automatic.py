"""自动会话使用真实检测、分句和调度，只在外部模型边界控制数据。"""

import unittest
from pathlib import Path
from threading import Event, Thread
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from speechio import (
    AudioFormat,
    LocalModelConfig,
    RealtimeConfig,
    RuntimeLimits,
    SegmentationConfig,
    SessionConfig,
    STTError,
    VadConfig,
)
from speechio.stt.input import AudioConverter
from speechio.stt.options import TranscriptionOptions
from speechio.stt.runtime.asr import STTRuntime
from tests.engines.test_stream import RuntimeRecognizer, RuntimeStream
from tests.runtime import test_realtime, test_transcriber


class AutomaticTests(unittest.TestCase):
    def setUp(self):
        test_transcriber.RecognizerTests.setUp(self)
        self.vad = VadConfig(
            model_path=Path(".venv/Lib/site-packages/faster_whisper/assets/silero_vad_v6.onnx"),
            warmup=False,
        )
        web = patch(
            "webrtcvad.Vad.is_speech",
            side_effect=lambda pcm, rate: bool(np.any(np.frombuffer(pcm, "<i2"))),
        )
        web.start()
        self.addCleanup(web.stop)
        graph = patch(
            "onnxruntime.InferenceSession.run",
            side_effect=lambda names, feeds: (np.array([0.9], np.float32), feeds["h"], feeds["c"]),
        )
        graph.start()
        self.addCleanup(graph.stop)
        self.final_audio = []

        def final(audio, **kwargs):
            self.final_audio.append(audio.copy())
            return iter([SimpleNamespace(text="result", start=0, end=len(audio) / 16000)]), (
                SimpleNamespace(language="en", language_probability=1)
            )

        native = patch("faster_whisper.BatchedInferencePipeline.transcribe", side_effect=final)
        native.start()
        self.addCleanup(native.stop)

    def test_continuous_input_creates_owned_segments_and_preserves_short_tail(self):
        audio = np.concatenate(
            (
                np.zeros(512 * 4, np.int16),
                np.full(512 * 20, 500, np.int16),
                np.zeros(512 * 30, np.int16),
                np.full(515, 700, np.int16),
            )
        )
        with STTRuntime(LocalModelConfig(adapter="whisper", model_path=self.path)) as runtime:
            session = runtime.create_auto_session(vad=self.vad)
            for start in range(0, len(audio), 317):
                session.feed_audio(audio[start : start + 317])
            session.end_input()
            finals = []
            while session.pending_segments:
                event = session.next_event(timeout=5)
                self.assertIsNotNone(event, "自动封口没有交付事件")
                if event.kind == "final":
                    finals.append(event)
            self.assertEqual(len(finals), 2)
            self.assertEqual(finals[-1].boundary.end_sample, len(audio))
            for event, actual in zip(finals, self.final_audio, strict=True):
                boundary = event.boundary
                np.testing.assert_array_equal(
                    actual, audio[boundary.start_sample : boundary.end_sample] / 32768
                )

    def test_cancel_current_segment_keeps_listening_without_recreating_cancelled_audio(self):
        with STTRuntime(LocalModelConfig(adapter="whisper", model_path=self.path)) as runtime:
            session = runtime.create_auto_session(vad=self.vad)
            session.feed_audio(np.full(1024, 500, np.int16))
            self.assertTrue(session.cancel_turn())
            self.assertEqual(session.next_event(timeout=1).kind, "cancelled")
            session.feed_audio(np.full(515, 700, np.int16))
            session.end_input()
            final = session.next_event(timeout=5)
            self.assertEqual(final.kind, "final")
            self.assertEqual(final.boundary.start_sample, 1024)
            np.testing.assert_array_equal(self.final_audio[-1], np.full(515, 700) / 32768)

    def test_detection_failure_fails_active_unit_and_reset_recovers(self):
        with STTRuntime(LocalModelConfig(adapter="whisper", model_path=self.path)) as runtime:
            session = runtime.create_auto_session(vad=self.vad)
            session.feed_audio(np.full(1024, 500, np.int16))
            with patch(
                "onnxruntime.InferenceSession.run", side_effect=RuntimeError("graph failed")
            ):
                with self.assertRaises(STTError) as caught:
                    session.feed_audio(np.full(512, 500, np.int16))
            self.assertEqual(caught.exception.code, "VAD_FAILED")
            self.assertEqual(session.input_state, "failed")
            event = session.next_event(timeout=1)
            self.assertEqual((event.kind, event.error.code), ("failed", "VAD_FAILED"))
            session.reset()
            session.feed_audio(np.full(515, 700, np.int16))
            session.end_input()
            final = session.next_event(timeout=5)
            self.assertEqual((final.turn.generation, final.boundary.start_sample), (1, 0))

    def test_close_during_detection_does_not_wait_or_register_late_segment(self):
        entered, release = Event(), Event()
        outcomes = []
        with STTRuntime(LocalModelConfig(adapter="whisper", model_path=self.path)) as runtime:
            session = runtime.create_auto_session(vad=self.vad)

            def blocked(names, feeds):
                entered.set()
                if not release.wait(5):
                    raise RuntimeError("没有释放外部检测")
                return np.array([0.9]), feeds["h"], feeds["c"]

            def feed():
                try:
                    session.feed_audio(np.full(512, 500, np.int16))
                except STTError as exc:
                    outcomes.append(exc.code)

            with patch("onnxruntime.InferenceSession.run", side_effect=blocked):
                worker = Thread(target=feed)
                worker.start()
                try:
                    self.assertTrue(entered.wait(5))
                    session.close()
                    self.assertEqual(runtime.capacity.open_sessions, 0)
                finally:
                    release.set()
                    worker.join(5)
            self.assertEqual(outcomes, ["SESSION_CLOSED"])
            self.assertEqual(session.pending_segments, 0)

    def test_slow_final_keeps_next_segment_and_rejected_block_can_be_resent(self):
        entered, release = Event(), Event()

        def blocked(audio, **kwargs):
            self.final_audio.append(audio.copy())
            entered.set()
            if not release.wait(5):
                raise RuntimeError("没有释放外部识别")
            return iter([]), SimpleNamespace(language="en", language_probability=1)

        config = SessionConfig(max_audio_samples=16000)
        segmentation = SegmentationConfig(max_buffer_samples=40000)
        with patch("faster_whisper.BatchedInferencePipeline.transcribe", side_effect=blocked):
            with STTRuntime(
                LocalModelConfig(adapter="whisper", model_path=self.path),
                limits=RuntimeLimits(max_tasks=1),
            ) as runtime:
                session = runtime.create_auto_session(
                    vad=self.vad, config=config, segmentation=segmentation
                )
                session.feed_audio(np.full(16384, 500, np.int16))
                try:
                    self.assertTrue(entered.wait(5))
                    session.feed_audio(np.full(16000, 700, np.int16))
                    block = np.full(35000, 900, np.int16)
                    with self.assertRaises(STTError) as caught:
                        session.feed_audio(block)
                    self.assertEqual(caught.exception.code, "AUDIO_CAPACITY_EXCEEDED")
                finally:
                    release.set()
                for _ in range(4):
                    self.assertIsNotNone(session.next_event(timeout=5))
                session.feed_audio(block)
                session.end_input()
                while session.pending_segments:
                    self.assertIsNotNone(session.next_event(timeout=5))
                actual = np.concatenate(self.final_audio)
                expected = np.concatenate((np.full(16384, 500), np.full(16000, 700), block))
                np.testing.assert_array_equal(actual, expected / 32768)

    def test_backdated_stream_finishes_each_left_segment_without_retranscription(self):
        streams = []

        class Native(RuntimeRecognizer):
            def create_stream(self):
                stream = RuntimeStream()
                streams.append(stream)
                return stream

        config = test_realtime.RealtimeTests.nemotron_fixture(self, Native())
        audio = np.concatenate(
            (
                np.full(10240, 500, np.int16),
                np.zeros(12288, np.int16),
                np.full(515, 700, np.int16),
            )
        )
        with STTRuntime(
            config,
            realtime=RealtimeConfig(
                mode="streaming", initial_delay_seconds=0, interval_seconds=0.01
            ),
        ) as runtime:
            session = runtime.create_auto_session(
                vad=self.vad,
                config=SessionConfig(realtime=True, options=TranscriptionOptions(language="zh-CN")),
                segmentation=SegmentationConfig(backdate_duration=0.2),
            )
            session.feed_audio(audio[:10240])
            first = session.next_event(timeout=5)
            self.assertEqual(first.source, "streaming")
            self.assertLessEqual(first.stream_end_sample, 10240 - 3200)
            session.feed_audio(audio[10240:])
            session.end_input()
            finals = []
            while session.pending_segments:
                event = session.next_event(timeout=5)
                self.assertIsNotNone(event)
                if event.kind == "final":
                    finals.append(event)
            self.assertEqual(len(finals), 2)
            self.assertEqual(finals[0].boundary.end_sample, finals[1].boundary.start_sample)
            for final in finals:
                boundary = final.boundary
                expected = audio[boundary.start_sample : boundary.end_sample] / 32768
                matching = [s for s in streams if len(s.samples) == len(expected)]
                self.assertEqual(len(matching), 1, "每段直接复用流式结果，不创建第二个流")
                for stream in matching:
                    np.testing.assert_array_equal(stream.samples, expected)
                    self.assertTrue(stream.ended)

    def test_source_resampling_stays_continuous_across_automatic_units(self):

        audio = np.repeat(np.array([500, 700, 900], np.float32) / 32768, 9000)
        source = np.column_stack((audio, audio))
        format = AudioFormat(48000, 2, "float32")
        converter = AudioConverter(format)
        expected = b"".join(f.pcm for f in converter.feed(source) + converter.finish())
        outputs = []
        with STTRuntime(LocalModelConfig(adapter="whisper", model_path=self.path)) as runtime:
            for block_size in (317, 2000):
                self.final_audio.clear()
                session = runtime.create_auto_session(
                    vad=self.vad,
                    config=SessionConfig(format=format, max_audio_samples=4000),
                    segmentation=SegmentationConfig(
                        pre_roll_duration=0, min_recording_duration=0, max_buffer_samples=20000
                    ),
                )
                for start in range(0, len(source), block_size):
                    session.feed_audio(source[start : start + block_size])
                session.end_input()
                boundaries = []
                while session.pending_segments:
                    event = session.next_event(timeout=5)
                    self.assertIsNotNone(event)
                    if event.kind == "final":
                        boundaries.append((event.boundary.start_sample, event.boundary.end_sample))
                outputs.append((boundaries, np.concatenate(self.final_audio)))
                session.close()
        self.assertEqual(outputs[0][0], [(0, 4000), (4000, 8000), (8000, 9000)])
        self.assertEqual(outputs[0][0], outputs[1][0])
        for _, actual in outputs:
            np.testing.assert_array_equal(actual, np.frombuffer(expected, "<i2") / 32768)

    def test_explicit_finish_seals_short_segment_and_keeps_listening(self):
        with STTRuntime(LocalModelConfig(adapter="whisper", model_path=self.path)) as runtime:
            session = runtime.create_auto_session(vad=self.vad)
            session.feed_audio(np.full(512, 500, np.int16))
            task = session.end_turn()
            self.assertEqual((task.audio_samples, task.boundary.reason), (512, "manual"))
            self.assertEqual(session.input_state, "listening")
            session.feed_audio(np.full(515, 700, np.int16))
            session.end_input()
            session.end_input()
            finals = []
            while session.pending_segments:
                event = session.next_event(timeout=5)
                self.assertIsNotNone(event)
                if event.kind == "final":
                    finals.append(event)
            self.assertEqual([event.audio_samples for event in finals], [512, 515])

    def test_silent_input_finishes_without_creating_a_task(self):
        with STTRuntime(LocalModelConfig(adapter="whisper", model_path=self.path)) as runtime:
            session = runtime.create_auto_session(vad=self.vad)
            session.feed_audio(bytes(32000))
            session.end_input()
            self.assertEqual(session.input_state, "finished")
            self.assertEqual(session.pending_segments, 0)
            self.assertIsNone(session.next_event())
            self.assertEqual(runtime.capacity.tasks, 0)
