"""通过公开自动会话接入检测提供方，分句及任务仍执行正式核心。"""

import unittest

import numpy as np

from realtime_asr import ASRRuntime, SegmentationConfig
from tests.runtime.test_engine_injection import RecordingEngine


class ControlledDetector:
    """外部检测边界；已知音频块给出已知活动判断。"""

    def __init__(self, identity):
        self.info = {"backend": "custom-vad", "context": identity}

    def detect(self, pcm, start_sample, *, recording=False):
        from realtime_asr import VoiceActivity

        speech = bool(np.any(np.frombuffer(pcm, dtype="<i2")))
        return VoiceActivity(speech, speech, speech, speech, 0.1 if speech else 0)

    def reset(self):
        pass


class DetectorInjectionTests(unittest.TestCase):
    def test_factory_owns_one_context_per_input_generation_and_keeps_automatic_flow(self):
        contexts = []

        def factory():
            detector = ControlledDetector(len(contexts))
            contexts.append(detector)
            return detector

        runtime = ASRRuntime(engine=RecordingEngine())
        self.addCleanup(runtime.close)
        runtime.prepare()
        session = runtime.create_auto_session(
            detector_factory=factory,
            segmentation=SegmentationConfig(
                silence_confirmation=0,
                post_speech_silence=0.032,
                min_recording_duration=0.032,
                pre_roll_duration=0.032,
            ),
        )
        for generation in (0, 1):
            self.assertEqual(session.vad_info, {"backend": "custom-vad", "context": generation})
            session.feed_audio(np.full(1024, 2000, np.int16))
            session.feed_audio(np.zeros(1024, np.int16))
            session.finish_input()
            kinds = []
            while session.pending_segments:
                event = session.get_event(timeout=5)
                self.assertIsNotNone(event)
                kinds.append(event.kind)
                self.assertEqual(event.unit.generation, generation)
            self.assertEqual(kinds, ["final", "completed"])
            if generation == 0:
                session.reset()
        self.assertEqual(len(contexts), 2)
        self.assertEqual(runtime.close().status, "ok")
