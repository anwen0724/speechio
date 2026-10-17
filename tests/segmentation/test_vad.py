"""检测器通过外部 WebRTC／ORT 边界验证错误和连续音频。"""

import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from speechio import ASRError
from speechio.segmentation.vad import VadConfig, VoiceActivityDetector


class VadTests(unittest.TestCase):
    def test_webrtc_warmup_uses_silence_without_changing_noise_floor(self):
        seen = []
        model = Path(".venv/Lib/site-packages/faster_whisper/assets/silero_vad_v6.onnx")
        with patch(
            "webrtcvad.Vad.is_speech", side_effect=lambda pcm, rate: seen.append(pcm) or False
        ):
            detector = VoiceActivityDetector(VadConfig(model_path=model))
        self.assertTrue(detector.info["warmed_up"])
        self.assertTrue(all(not any(frame) for frame in seen), "WebRTC 预热不能包含模拟语音")

    def test_detector_failure_is_visible_instead_of_silence(self):
        model = Path(".venv/Lib/site-packages/faster_whisper/assets/silero_vad_v6.onnx")
        detector = VoiceActivityDetector(VadConfig(model_path=model, warmup=False))
        with patch("webrtcvad.Vad.is_speech", side_effect=RuntimeError("device detector error")):
            with self.assertRaises(ASRError) as caught:
                detector.detect(bytes(1024), 0)
        self.assertEqual(caught.exception.code, "VAD_FAILED")

    def test_webrtc_short_frames_preserve_all_input_and_detector_tail_only_is_padded(self):
        frames = []

        def detect(frame, rate):
            frames.append(frame)
            return False

        model = Path(".venv/Lib/site-packages/faster_whisper/assets/silero_vad_v6.onnx")
        detector = VoiceActivityDetector(VadConfig(model_path=model, warmup=False))
        original = np.arange(1027, dtype="<i2").tobytes()
        with patch("webrtcvad.Vad.is_speech", side_effect=detect):
            detector.detect(original[:1024], 0)
            detector.detect(original[1024:2048], 512)
            detector.detect(original[2048:], 1024)
        output = b"".join(frames)
        self.assertEqual(output[: len(original)], original)
        self.assertEqual(output[len(original) :], bytes(2 * (1120 - 1027)))

    def test_silero_failure_is_visible_on_the_selected_end_path(self):
        model = Path(".venv/Lib/site-packages/faster_whisper/assets/silero_vad_v6.onnx")
        detector = VoiceActivityDetector(
            VadConfig(model_path=model, warmup=False, silero_end_detection=True)
        )
        with patch("webrtcvad.Vad.is_speech", return_value=False):
            with patch("onnxruntime.InferenceSession.run", side_effect=RuntimeError("bad graph")):
                with self.assertRaises(ASRError) as caught:
                    detector.detect(bytes(1024), 0, recording=True)
        self.assertEqual(caught.exception.code, "VAD_FAILED")

    def test_intra_op_threads_must_be_positive(self):
        with self.assertRaises(ASRError) as caught:
            VadConfig(model_path=Path("model.onnx"), intra_op_threads=0)
        self.assertEqual(caught.exception.code, "INVALID_CONFIGURATION")


if __name__ == "__main__":
    unittest.main()
