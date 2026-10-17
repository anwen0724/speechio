"""从正式自动会话验证唤醒；替身只位于外部检测与识别引擎。"""

import unittest
from threading import Event, Thread

import numpy as np

import speechio as sdk
from speechio.runtime.asr import ASRRuntime
from speechio.segmentation.vad import VoiceActivity
from tests.runtime.test_engine_injection import RecordingEngine


class SpeechDetector:
    info = {"backend": "controlled-vad"}

    def detect(self, pcm, start_sample, *, recording=False):
        speech = bool(np.any(np.frombuffer(pcm, dtype="<i2")))
        return VoiceActivity(speech, speech, speech, speech, 0.1 if speech else 0)

    def reset(self):
        pass


class WakeDetector:
    sample_rate = 16000
    frame_samples = 512

    def __init__(self):
        self.closed = 0

    def detect(self, pcm):
        return "hello" if np.frombuffer(pcm, dtype="<i2")[0] == 500 else None

    def reset(self):
        pass

    def close(self):
        self.closed += 1


class WakeTests(unittest.TestCase):
    def test_close_waits_for_native_detection_before_releasing(self):
        class BlockingWake(WakeDetector):
            def __init__(self):
                super().__init__()
                self.entered, self.release = Event(), Event()

            def detect(self, pcm):
                self.entered.set()
                if not self.release.wait(5):
                    raise TimeoutError("检测未释放")
                return super().detect(pcm)

        detector, errors = BlockingWake(), []
        runtime = ASRRuntime(engine=RecordingEngine())
        self.addCleanup(runtime.close)
        runtime.prepare()
        session = runtime.create_auto_session(
            detector_factory=SpeechDetector,
            wake=sdk.WakeConfig(),
            wake_detector_factory=lambda: detector,
        )

        def feed():
            try:
                session.feed_audio(np.full(512, 500, np.int16))
            except sdk.ASRError as exc:
                errors.append(exc.code)

        worker = Thread(target=feed)
        worker.start()
        try:
            self.assertTrue(detector.entered.wait(5))
            self.assertEqual(session.close().pending_tasks, 1)
            self.assertFalse(session.wake_status["released"])
            self.assertEqual(detector.closed, 0)
            self.assertEqual(runtime.close(timeout=0).pending_tasks, 1)
        finally:
            detector.release.set()
            worker.join(5)
        self.assertEqual(errors, ["SESSION_CLOSED"])
        self.assertEqual(runtime.close().status, "ok")
        self.assertEqual(detector.closed, 1)
        self.assertTrue(session.wake_status["released"])

    def test_detector_failure_requires_reset_and_uses_new_instance(self):
        class FailedWake(WakeDetector):
            def detect(self, pcm):
                raise RuntimeError("检测错误")

        detectors = [FailedWake(), WakeDetector()]
        supplied = iter(detectors)
        with ASRRuntime(engine=RecordingEngine()) as runtime:
            session = runtime.create_auto_session(
                detector_factory=SpeechDetector,
                wake=sdk.WakeConfig(trim_duration=0),
                wake_detector_factory=lambda: next(supplied),
            )
            with self.assertRaises(sdk.ASRError) as caught:
                session.feed_audio(np.full(512, 500, np.int16))
            self.assertEqual(caught.exception.code, "WAKE_DETECTION_FAILED")
            self.assertEqual(session.wake_status["state"], "failed")
            session.reset()
            session.feed_audio(np.full(512, 500, np.int16))
            self.assertEqual(session.active_turn.generation, 1)
        self.assertEqual([detector.closed for detector in detectors], [1, 1])

    def test_speech_timeout_and_finish_followup_window_use_audio_clock(self):
        class BodyDetector(SpeechDetector):
            def detect(self, pcm, start_sample, *, recording=False):
                speech = np.frombuffer(pcm, dtype="<i2")[0] == 200
                return VoiceActivity(speech, speech, speech, speech, 0)

        with ASRRuntime(engine=RecordingEngine()) as runtime:
            session = runtime.create_auto_session(
                detector_factory=BodyDetector,
                segmentation=sdk.SegmentationConfig(pre_roll_duration=0, min_recording_duration=0),
                wake=sdk.WakeConfig(trim_duration=0, speech_timeout=0.032, followup_window=0.064),
                wake_detector_factory=WakeDetector,
            )
            for value in (500, 0, 200):
                session.feed_audio(np.full(512, value, np.int16))
            self.assertIsNone(session.active_turn)
            self.assertEqual(session.wake_status["timeouts"], 1)
            session.feed_audio(np.full(512, 500, np.int16))
            session.feed_audio(np.full(512, 200, np.int16))
            session.end_turn()
            self.assertEqual(session.wake_status["state"], "followup")
            self.assertEqual(session.wake_status["deadline_sample"], 3584)
            session.feed_audio(np.full(512, 200, np.int16))
            self.assertIsNotNone(session.active_turn)
            session.end_turn()
            session.feed_audio(np.zeros(1024, np.int16))
            session.feed_audio(np.full(512, 200, np.int16))
            self.assertEqual(session.wake_status["state"], "awaiting_wake")
            self.assertIsNone(session.active_turn)

    def test_wake_gates_speech_and_trims_only_fixed_prefix(self):
        engine, detector = RecordingEngine(), WakeDetector()
        with ASRRuntime(engine=engine) as runtime:
            session = runtime.create_auto_session(
                detector_factory=SpeechDetector,
                segmentation=sdk.SegmentationConfig(
                    pre_roll_duration=0.064, min_recording_duration=0
                ),
                wake=sdk.WakeConfig(trim_duration=0.008),
                wake_detector_factory=lambda: detector,
            )
            before, wake, speech = (np.full(512, value, np.int16) for value in (100, 500, 800))
            session.feed_audio(before)
            self.assertEqual(session.wake_status["state"], "awaiting_wake")
            self.assertIsNone(session.active_turn)
            session.feed_audio(wake)
            session.feed_audio(speech)
            self.assertEqual(session.wake_status["state"], "recording")
            session.end_input()
            final = session.next_event(timeout=5)
            self.assertEqual(final.kind, "final")
            self.assertEqual(final.stream_start_sample, 128)
            np.testing.assert_array_equal(
                engine.inputs[0][0],
                np.concatenate((before, wake, speech))[128:].astype(np.float32) / 32768,
            )
            self.assertEqual(session.next_event(timeout=5).kind, "completed")
        self.assertEqual(detector.closed, 1)
