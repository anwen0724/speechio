"""两个入门入口的输入分派与实时结果交付，不启动真实模型或设备。"""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import speechio
from speechio import (
    ASRError,
    AudioFormat,
    BaseTranscriptionEngine,
    CloseResult,
    EngineCapabilities,
    StreamTranscriber,
    Transcript,
    transcribe_audio,
    transcribe_stream,
)
from speechio.bundled import default_vad_model
from speechio.runtime.asr import ASRRuntime
from speechio.segmentation.vad import VoiceActivity


class RecordingEngine(BaseTranscriptionEngine):
    """替代模型边界，记录实际收到的样本并支持推理故障。"""

    capabilities = EngineCapabilities(engine="test")

    def __init__(self, *, fail=False):
        self.fail = fail
        self.samples = 0
        self.closed = False
        self.preparations = 0

    def prepare(self):
        self.preparations += 1

    def transcribe(self, audio, language=None, *, role="final", word_timestamps=False):
        self.samples += len(audio)
        if self.fail:
            raise ASRError("INFERENCE_FAILED", "transcribe", "controlled model failure")
        return Transcript(text="recognized")

    def close(self):
        self.closed = True


class SpeechDetector:
    """替代 VAD 模型边界，将固定输入判定为语音。"""

    info = {"backend": "test"}

    def detect(self, pcm, start_sample, *, recording=False):
        return VoiceActivity(True, True, True, True, 1.0)

    def reset(self):
        pass


class StreamRuntime(ASRRuntime):
    """保留真实运行时和会话，只注入无需加载权重的检测器。"""

    def create_auto_session(self, **options):
        options.pop("vad")
        return super().create_auto_session(detector_factory=SpeechDetector, **options)


class RecordingTranscriber:
    """记录入门函数选中了哪个转写方法。"""

    calls = []

    def __init__(self, **options):
        self.options = options

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def transcribe_file(self, path, **options):
        self.calls.append(("file", path, options))
        return Transcript(text="file")

    def transcribe(self, audio, *, format, **options):
        self.calls.append(("audio", audio, format, options))
        return Transcript(text="audio")

    def transcribe_chunks(self, chunks, *, format, **options):
        self.calls.append(("chunks", list(chunks), format, options))
        return Transcript(text="chunks")


class FakeMicrophone:
    """按帧交付固定音频，读完返回 None。"""

    def __init__(self, config):
        self.config = config
        self.frames = [b"\x00\x00"]
        self.closed = False

    def prepare(self):
        return AudioFormat()

    def start(self):
        pass

    def read_audio(self, *, timeout=0):
        return self.frames.pop(0) if self.frames else None

    def close(self, *, timeout=2):
        self.closed = True
        return CloseResult()


class FinishingMicrophone(FakeMicrophone):
    """确定性模拟停止时仍有一次设备读取：先返回 pending，再交付尾块。"""

    def __init__(self, config):
        super().__init__(config)
        self.frames = []
        self.stopping = False

    def close(self, *, timeout=2):
        if not self.stopping:
            self.stopping = True
            return CloseResult(status="pending")
        if not self.closed:
            self.frames.append(b"\xd0\x07" * 1600)
        return super().close(timeout=timeout)


class FakeSession:
    """按预设顺序交付事件；结束输入后不再产生新事件。"""

    def __init__(self, events):
        self.events = list(events)
        self.fed = []
        self.pending_segments = 0
        self.closed = False

    def feed_audio(self, chunk):
        self.fed.append(chunk)

    def next_event(self, *, timeout=0):
        return self.events.pop(0) if self.events else None

    def end_input(self):
        pass

    def close(self):
        self.closed = True


class FakeRuntime:
    def __init__(self, session):
        self.session = session
        self.prepared = False

    def prepare(self):
        self.prepared = True

    def create_auto_session(self, **options):
        self.options = options
        return self.session

    def close(self):
        return CloseResult()


def event(kind, text="", error=None):
    result = Transcript(text=text) if text else None
    return SimpleNamespace(kind=kind, result=result, merge=None, stabilization=None, error=error)


class StreamLifecycleTests(unittest.TestCase):
    """通过门面验证停止与清理，运行时和会话均使用真实实现。"""

    def make_stream(self, microphone, *, fail=False):
        engine = RecordingEngine(fail=fail)
        runtime = StreamRuntime(engine=engine)
        self.addCleanup(runtime.close)
        self.enterContext(patch("speechio.api.ASRRuntime", return_value=runtime))
        self.enterContext(patch("speechio.api.MicrophoneInput", return_value=microphone))
        stream = StreamTranscriber(engine=engine, realtime=False)
        stream.start()
        self.addCleanup(stream.close)
        return stream, engine, runtime

    def test_stream_accepts_an_injected_engine(self):
        engine = RecordingEngine()
        microphone = FakeMicrophone(None)
        microphone.frames = [b"\xd0\x07" * 1600]
        with (
            patch("speechio.api.ASRRuntime", StreamRuntime),
            patch("speechio.api.MicrophoneInput", return_value=microphone),
            StreamTranscriber(engine=engine, realtime=False) as stream,
        ):
            stream.end_input()
            transcripts = list(stream)
        self.assertEqual([item.text for item in transcripts], ["recognized"])
        self.assertEqual(engine.samples, 1600)
        self.assertTrue(engine.closed)

    def test_end_input_delivers_all_queued_audio(self):
        microphone = FakeMicrophone(None)
        microphone.frames = [b"\xd0\x07" * 1600 for _ in range(3)]
        stream, engine, _ = self.make_stream(microphone)

        stream.end_input()
        transcripts = list(stream)

        self.assertEqual(engine.samples, 4800)
        self.assertEqual([item.text for item in transcripts], ["recognized"])

    def test_end_input_waits_for_last_capture_block(self):
        stream, engine, _ = self.make_stream(FinishingMicrophone(None))

        stream.end_input()
        transcripts = list(stream)

        self.assertEqual(engine.samples, 1600)
        self.assertEqual([item.text for item in transcripts], ["recognized"])

    def test_close_releases_resources_after_inference_failure(self):
        microphone = FakeMicrophone(None)
        microphone.frames = [b"\xd0\x07" * 16000]
        stream, engine, runtime = self.make_stream(microphone, fail=True)
        stream.end_input()

        with self.assertRaises(ASRError) as caught:
            stream.close()

        self.assertEqual(caught.exception.code, "INFERENCE_FAILED")
        self.assertTrue(microphone.closed)
        self.assertTrue(engine.closed)
        self.assertEqual(runtime.state, "closed")
        self.assertEqual(stream.state, "idle")
        stream.close()


class EntryTests(unittest.TestCase):
    def test_audio_entry_validates_input_before_preparing_and_always_closes(self):
        for audio, code in (
            (b"", "INVALID_AUDIO"),
            ("missing-audit-audio.wav", "AUDIO_READ_FAILED"),
        ):
            with self.subTest(audio=audio):
                engine = RecordingEngine()
                with self.assertRaises(ASRError) as caught:
                    transcribe_audio(audio, engine=engine)
                self.assertEqual(caught.exception.code, code)
                self.assertEqual(engine.preparations, 0)
                self.assertTrue(engine.closed)
        engine = RecordingEngine()
        self.assertEqual(transcribe_audio(b"\xd0\x07" * 1600, engine=engine).text, "recognized")
        self.assertEqual(engine.preparations, 1)
        self.assertTrue(engine.closed)

    def test_audio_entry_dispatches_by_input_type(self):
        RecordingTranscriber.calls = []
        with patch("speechio.api.Transcriber", RecordingTranscriber):
            self.assertEqual(transcribe_audio("a.wav").text, "file")
            self.assertEqual(transcribe_audio(b"\x00\x00").text, "audio")
            self.assertEqual(transcribe_audio([b"\x00\x00"]).text, "chunks")
        self.assertEqual(
            [call[0] for call in RecordingTranscriber.calls], ["file", "audio", "chunks"]
        )

    def test_entry_names_are_public_and_runtime_is_not(self):
        for name in ("transcribe_audio", "transcribe_stream", "Transcriber", "StreamTranscriber"):
            self.assertIn(name, speechio.__all__)
        self.assertNotIn("ASRRuntime", speechio.__all__)
        self.assertFalse(hasattr(speechio, "ASRRuntime"))

    def test_stream_uses_bundled_vad_model_by_default(self):
        session = FakeSession([event("final", "hello")])
        runtime = FakeRuntime(session)
        with (
            patch("speechio.api.ASRRuntime", lambda *args, **kwargs: runtime),
            patch("speechio.api.MicrophoneInput", FakeMicrophone),
        ):
            for _ in transcribe_stream(engine="whisper", model_path="models/tiny"):
                break
        self.assertEqual(runtime.options["vad"].model_path, default_vad_model())

    def test_stream_uses_explicit_vad_model(self):
        with tempfile.TemporaryDirectory() as directory:
            custom = Path(directory) / "custom.onnx"
            custom.write_bytes(b"model")
            session = FakeSession([event("final", "hello")])
            runtime = FakeRuntime(session)
            with (
                patch("speechio.api.ASRRuntime", lambda *args, **kwargs: runtime),
                patch("speechio.api.MicrophoneInput", FakeMicrophone),
            ):
                for _ in transcribe_stream(
                    engine="whisper", model_path="models/tiny", vad_model=custom
                ):
                    break
        self.assertEqual(runtime.options["vad"].model_path, custom)

    def test_stream_rejects_missing_custom_vad_model(self):
        with self.assertRaises(ASRError) as caught:
            StreamTranscriber(engine="whisper", model_path="models/tiny", vad_model="missing.onnx")
        self.assertEqual(caught.exception.code, "INVALID_CONFIGURATION")

    def test_stream_delivers_final_and_partial(self):
        session = FakeSession(
            [
                event("realtime", "hel"),
                event("final", "hello"),
            ]
        )
        runtime = FakeRuntime(session)
        partials = []
        with (
            patch("speechio.api.ASRRuntime", lambda *args, **kwargs: runtime),
            patch("speechio.api.MicrophoneInput", FakeMicrophone),
        ):
            for transcript in transcribe_stream(
                engine="whisper", model_path="models/tiny", on_partial=partials.append
            ):
                self.assertEqual(transcript.text, "hello")
                break
        self.assertEqual(partials, ["hel"])
        self.assertTrue(session.closed)

    def test_stream_failure_becomes_error(self):
        error = SimpleNamespace(code="VAD_FAILED", operation="detect", message="bad frame")
        session = FakeSession([event("failed", error=error)])
        runtime = FakeRuntime(session)
        with (
            patch("speechio.api.ASRRuntime", lambda *args, **kwargs: runtime),
            patch("speechio.api.MicrophoneInput", FakeMicrophone),
        ):
            with self.assertRaises(ASRError) as caught:
                for _ in transcribe_stream(engine="whisper", model_path="models/tiny"):
                    pass
        self.assertEqual(caught.exception.code, "VAD_FAILED")
        self.assertTrue(session.closed)


if __name__ == "__main__":
    unittest.main()
