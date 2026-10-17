"""两个入门入口的输入分派与实时结果交付，不启动真实模型或设备。"""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import realtime_asr
from realtime_asr import (
    ASRError,
    AudioFormat,
    CloseResult,
    StreamTranscriber,
    Transcript,
    transcribe_audio,
    transcribe_stream,
)
from realtime_asr.bundled import default_vad_model


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
        pass


def event(kind, text="", error=None):
    result = Transcript(text=text) if text else None
    return SimpleNamespace(kind=kind, result=result, merge=None, stabilization=None, error=error)


class EntryTests(unittest.TestCase):
    def test_audio_entry_dispatches_by_input_type(self):
        RecordingTranscriber.calls = []
        with patch("realtime_asr.api.Transcriber", RecordingTranscriber):
            self.assertEqual(transcribe_audio("a.wav").text, "file")
            self.assertEqual(transcribe_audio(b"\x00\x00").text, "audio")
            self.assertEqual(transcribe_audio([b"\x00\x00"]).text, "chunks")
        self.assertEqual(
            [call[0] for call in RecordingTranscriber.calls], ["file", "audio", "chunks"]
        )

    def test_entry_names_are_public_and_runtime_is_not(self):
        for name in ("transcribe_audio", "transcribe_stream", "Transcriber", "StreamTranscriber"):
            self.assertIn(name, realtime_asr.__all__)
        self.assertNotIn("ASRRuntime", realtime_asr.__all__)
        self.assertFalse(hasattr(realtime_asr, "ASRRuntime"))

    def test_stream_uses_bundled_vad_model_by_default(self):
        session = FakeSession([event("final", "hello")])
        runtime = FakeRuntime(session)
        with (
            patch("realtime_asr.api.ASRRuntime", lambda *args, **kwargs: runtime),
            patch("realtime_asr.api.MicrophoneInput", FakeMicrophone),
        ):
            for _ in transcribe_stream(model="tiny"):
                break
        self.assertEqual(runtime.options["vad"].model_path, default_vad_model())

    def test_stream_uses_explicit_vad_model(self):
        with tempfile.TemporaryDirectory() as directory:
            custom = Path(directory) / "custom.onnx"
            custom.write_bytes(b"model")
            session = FakeSession([event("final", "hello")])
            runtime = FakeRuntime(session)
            with (
                patch("realtime_asr.api.ASRRuntime", lambda *args, **kwargs: runtime),
                patch("realtime_asr.api.MicrophoneInput", FakeMicrophone),
            ):
                for _ in transcribe_stream(model="tiny", vad_model=custom):
                    break
        self.assertEqual(runtime.options["vad"].model_path, custom)

    def test_stream_rejects_missing_custom_vad_model(self):
        with self.assertRaises(ASRError) as caught:
            StreamTranscriber(model="tiny", vad_model="missing.onnx")
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
            patch("realtime_asr.api.ASRRuntime", lambda *args, **kwargs: runtime),
            patch("realtime_asr.api.MicrophoneInput", FakeMicrophone),
        ):
            for transcript in transcribe_stream(model="tiny", on_partial=partials.append):
                self.assertEqual(transcript.text, "hello")
                break
        self.assertEqual(partials, ["hel"])
        self.assertTrue(session.closed)

    def test_stream_failure_becomes_error(self):
        error = SimpleNamespace(code="VAD_FAILED", operation="detect", message="bad frame")
        session = FakeSession([event("failed", error=error)])
        runtime = FakeRuntime(session)
        with (
            patch("realtime_asr.api.ASRRuntime", lambda *args, **kwargs: runtime),
            patch("realtime_asr.api.MicrophoneInput", FakeMicrophone),
        ):
            with self.assertRaises(ASRError) as caught:
                for _ in transcribe_stream(model="tiny"):
                    pass
        self.assertEqual(caught.exception.code, "VAD_FAILED")
        self.assertTrue(session.closed)


if __name__ == "__main__":
    unittest.main()
