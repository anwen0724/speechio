"""用外部模型替身验证真实合成入口和音频结果。"""

import tempfile
import unittest
import wave
from pathlib import Path
from threading import Event, Thread

from speechio import AudioFormat, TTSError
from speechio.adapters.contracts import TTSAdapter, TTSCapabilities
from speechio.audio.results import AudioResult
from speechio.tts.options import SynthesisOptions
from speechio.tts.synthesizer import Synthesizer


class RecordingEngine(TTSAdapter):
    capabilities = TTSCapabilities(default_sample_rate=24000, synthesis=True)

    def __init__(self):
        self.received = []
        self.closes = 0

    def synthesize(self, text, *, options):
        self.received.append((text, options))
        return AudioResult(b"\x01\x00" * 240, AudioFormat(sample_rate=24000))

    def close(self):
        self.closes += 1


class SynthesisTests(unittest.TestCase):
    def test_close_waits_for_running_native_call_before_releasing_engine(self):
        entered = Event()
        finish = Event()
        released = Event()

        class BlockingEngine(RecordingEngine):
            def synthesize(self, text, *, options):
                entered.set()
                finish.wait()
                return super().synthesize(text, options=options)

            def close(self):
                released.set()

        synth = Synthesizer.from_adapter(adapter=BlockingEngine())
        outputs = []
        worker = Thread(
            target=lambda: outputs.append(
                synth.synthesize("Hello.", options=SynthesisOptions(voice="test"))
            )
        )
        worker.start()
        self.assertTrue(entered.wait(1))
        try:
            self.assertEqual(synth.close(timeout=0).status, "pending")
            self.assertFalse(released.is_set())
        finally:
            finish.set()
            worker.join(1)
        self.assertEqual(synth.close(timeout=1).status, "ok")
        self.assertEqual(len(outputs), 1)

    def test_context_retains_body_error_and_reports_cleanup_failure(self):
        class BrokenClose(RecordingEngine):
            def close(self):
                raise RuntimeError("cleanup failed")

        with self.assertRaisesRegex(ValueError, "body failure") as caught:
            with Synthesizer.from_adapter(adapter=BrokenClose()):
                raise ValueError("body failure")
        self.assertTrue(caught.exception.__notes__)

    def test_failed_preparation_releases_owned_engine_without_explicit_close(self):
        released = Event()

        class BrokenPreparation(RecordingEngine):
            def prepare(self):
                raise RuntimeError("private credential")

            def close(self):
                released.set()

        synth = Synthesizer.from_adapter(adapter=BrokenPreparation())
        with self.assertRaises(TTSError):
            synth.synthesize("Hello.", options=SynthesisOptions(voice="test"))
        self.assertTrue(released.wait(1))

    def test_unsupported_options_are_rejected_before_model_execution(self):
        engine = RecordingEngine()
        with Synthesizer.from_adapter(adapter=engine) as synth:
            with self.assertRaises(TTSError) as caught:
                synth.synthesize("Hello.", options=SynthesisOptions(voice="test", speed=2))
            self.assertEqual(caught.exception.code, "CAPABILITY_UNSUPPORTED")
            self.assertEqual(engine.received, [])

    def test_context_closes_adapter_and_save_does_not_overwrite(self):
        engine = RecordingEngine()
        with Synthesizer.from_adapter(engine) as synthesizer:
            audio = synthesizer.synthesize("Hello.", options=SynthesisOptions(voice="test"))
        self.assertEqual(engine.closes, 1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audio.wav"
            path.write_bytes(b"original")
            with self.assertRaises(TTSError):
                audio.save(path)
            self.assertEqual(path.read_bytes(), b"original")

    def test_complete_audio_can_be_reused_and_saved_as_wav(self):
        engine = RecordingEngine()
        synthesizer = Synthesizer.from_adapter(adapter=engine)
        options = SynthesisOptions(voice="test")
        result = synthesizer.synthesize("Hello.", options=options)
        second = synthesizer.synthesize("你好。", options=options)
        self.assertEqual([text for text, _ in engine.received], ["Hello.", "你好。"])
        self.assertEqual(engine.received[0][1].voice, options.voice)
        self.assertEqual(engine.received[0][1].sample_rate, 24000)
        self.assertEqual(second.data, result.data)
        self.assertEqual(result.duration_seconds, 0.01)
        with tempfile.TemporaryDirectory() as directory:
            path = result.save(Path(directory) / "result.wav")
            with wave.open(str(path), "rb") as audio:
                self.assertEqual(
                    (audio.getnchannels(), audio.getsampwidth(), audio.getframerate()),
                    (1, 2, 24000),
                )
                self.assertEqual(audio.readframes(240), result.data)
        self.assertEqual(synthesizer.close().status, "ok")
        synthesizer.close()
        self.assertEqual(engine.closes, 1)
