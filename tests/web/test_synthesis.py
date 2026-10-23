"""从 Web 边界验证合成；模型替身不连接云端。"""

import tempfile
import unittest
from pathlib import Path
from threading import Event
from unittest.mock import patch

from fastapi.testclient import TestClient

from apps.web.server.app import create_app
from apps.web.server.settings import Settings, SynthesisPreset
from speechio import AudioChunk, AudioFormat, AudioResult, RemoteModelConfig, SynthesisOptions
from speechio.adapters.contracts import TTSAdapter, TTSCapabilities
from tests.tts.test_stream import ControlledBackend


class VoiceBackend(ControlledBackend):
    def send_text(self, text):
        super().send_text(text)
        self.queue.put(AudioChunk(b"\x01\x00" * 120, self.format))

    def finish_input(self):
        super().finish_input()
        self.queue.put(AudioChunk(b"\x02\x00" * 120, self.format))
        self.queue.put(None)


class VoiceEngine(TTSAdapter):
    capabilities = TTSCapabilities(
        default_sample_rate=24000, synthesis=True, audio_streaming=True, text_streaming=True
    )

    def __init__(self):
        self.backend = VoiceBackend()
        self.closed = False
        self.options = None

    def synthesize(self, text, *, options):
        self.options = options
        return AudioResult(b"\x01\x00" * 240, AudioFormat(sample_rate=24000))

    def stream(self, text, *, options):
        self.options = options
        self.backend.send_text(text)
        self.backend.finish_input()
        return self.backend

    def open_stream(self, *, options):
        self.options = options
        return self.backend

    def close(self):
        self.closed = True


def synthesis_settings(directory, *, debug=True):
    return Settings(
        models={},
        debug=debug,
        upload_dir=Path(directory),
        synthesis_models={
            "voice": SynthesisPreset(
                RemoteModelConfig(
                    provider="bailian", model="qwen-audio-3.1-tts-flash", api_key="private-test-key"
                ),
                SynthesisOptions(voice="Cherry"),
            )
        },
    )


class SynthesisWebTests(unittest.TestCase):
    def test_cancel_full_task_keeps_shared_slot_until_model_returns(self):
        entered, finish = Event(), Event()

        class SlowEngine(VoiceEngine):
            def synthesize(self, text, *, options):
                entered.set()
                finish.wait(5)
                return super().synthesize(text, options=options)

        with tempfile.TemporaryDirectory() as directory:
            settings = synthesis_settings(directory)
            engine = SlowEngine()
            with patch("speechio.models.factory.build_synthesis_engine", return_value=engine):
                with TestClient(create_app(settings)) as client:
                    with client.websocket_connect("/api/synthesis") as owner:
                        owner.send_json(
                            {
                                "type": "start",
                                "text": "Hello",
                                "config": {"model": "voice", "mode": "full"},
                            }
                        )
                        self.assertEqual(owner.receive_json()["type"], "ready")
                        self.assertTrue(entered.wait(1))
                        owner.send_json({"type": "cancel"})
                        self.assertEqual(client.post("/api/debug/release").status_code, 409)
                        self.assertEqual(
                            client.post("/api/uploads", content=b"test").status_code, 409
                        )
                        finish.set()
                        self.assertEqual(owner.receive_json()["type"], "cancelled")
                    self.assertTrue(engine.closed)
                    self.assertEqual(client.post("/api/debug/release").status_code, 200)
                    self.assertEqual(list(Path(directory).rglob("*.wav")), [])

    def test_disconnect_cleans_incremental_stream_without_waiting_for_text(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = synthesis_settings(directory)
            engine = VoiceEngine()
            with patch("speechio.models.factory.build_synthesis_engine", return_value=engine):
                with TestClient(create_app(settings)) as client:
                    with client.websocket_connect("/api/synthesis?debug=1") as socket:
                        socket.send_json(
                            {"type": "start", "config": {"model": "voice", "mode": "incremental"}}
                        )
                        self.assertEqual(socket.receive_json()["type"], "ready")
                    self.assertTrue(engine.backend.closed.wait(1))
                self.assertTrue(engine.closed)

    def test_output_limit_removes_partial_file_and_debug_switch_is_enforced(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = synthesis_settings(directory, debug=False)
            engine = VoiceEngine()
            with patch("speechio.models.factory.build_synthesis_engine", return_value=engine):
                with TestClient(create_app(settings)) as client:
                    self.assertEqual(client.post("/api/debug/synthesis/release").status_code, 404)
                    for mode, code in (
                        ("incremental", "INVALID_REQUEST"),
                        ("full", "OUTPUT_LIMIT"),
                    ):
                        with patch("apps.web.server.synthesis.SynthesisFiles.limit", 100):
                            with client.websocket_connect("/api/synthesis") as socket:
                                socket.send_json(
                                    {
                                        "type": "start",
                                        "text": "Hello",
                                        "config": {"model": "voice", "mode": mode},
                                    }
                                )
                                event = socket.receive_json()
                                if event["type"] == "ready":
                                    event = socket.receive_json()
                                self.assertEqual(event["code"], code)
                    self.assertEqual(list(Path(directory).rglob("*.wav")), [])

    def test_debug_prepare_reports_bad_configuration_and_releases_resources(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = synthesis_settings(directory)
            engine = VoiceEngine()
            with patch("speechio.models.factory.build_synthesis_engine", return_value=engine):
                with TestClient(create_app(settings), raise_server_exceptions=False) as client:
                    self.assertEqual(
                        client.post(
                            "/api/debug/synthesis/prepare", json={"model": "absent"}
                        ).status_code,
                        400,
                    )
                    result = client.post("/api/debug/synthesis/prepare", json={"model": "voice"})
                    self.assertEqual(result.status_code, 200)
                    self.assertNotIn("private-test-key", result.text)
                    self.assertEqual(client.post("/api/debug/synthesis/release").status_code, 200)
                    self.assertTrue(engine.closed)

    def test_full_stream_and_incremental_produce_downloadable_wav_and_cleanup(self):
        for mode in ("full", "stream", "incremental"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                engine = VoiceEngine()
                settings = synthesis_settings(directory)
                with patch("speechio.models.factory.build_synthesis_engine", return_value=engine):
                    with TestClient(create_app(settings)) as client:
                        with client.websocket_connect("/api/synthesis?debug=1") as socket:
                            socket.send_json(
                                {
                                    "type": "start",
                                    "text": "Hello",
                                    "config": {
                                        "model": "voice",
                                        "mode": mode,
                                        "voice": "custom-voice",
                                    },
                                }
                            )
                            ready = socket.receive_json()
                            self.assertEqual(ready["type"], "ready", ready)
                            self.assertNotIn("private-test-key", str(ready))
                            if mode == "incremental":
                                socket.send_json({"type": "text", "text": "Hello "})
                                socket.send_json({"type": "text", "text": "world"})
                                socket.send_json({"type": "finish_input"})
                            audio = bytearray()
                            while True:
                                message = socket.receive()
                                if message.get("bytes") is not None:
                                    audio.extend(message["bytes"])
                                    continue
                                import json

                                event = json.loads(message["text"])
                                self.assertNotEqual(event["type"], "error", event)
                                if event["type"] == "completed":
                                    break
                            result = client.get(event["audio_url"])
                            self.assertEqual(result.status_code, 200)
                            self.assertEqual(result.content[:4], b"RIFF")
                            self.assertEqual(len(result.content) - 44, event["metrics"]["bytes"])
                            if mode != "full":
                                self.assertEqual(result.content[44:], audio)
                            self.assertEqual(engine.options.voice, "custom-voice")
                            with patch("apps.web.server.synthesis.SynthesisFiles.lifetime", -1):
                                self.assertEqual(client.get(event["audio_url"]).status_code, 404)
                        self.assertTrue(engine.closed)
                        if mode == "incremental":
                            self.assertEqual(engine.backend.texts, ["Hello ", "world"])
                self.assertEqual(list(Path(directory).rglob("*.wav")), [])


class SynthesisSettingsTests(unittest.TestCase):
    def test_synthesis_models_have_separate_public_defaults_and_capabilities(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.toml"
            path.write_text(
                '[models.voice]\ntask="synthesis"\nsource="remote"\n'
                'provider="bailian"\nmodel="qwen-audio-3.1-tts-flash"\n'
                'apikey="private-test-key"\nvoices=["Cherry"]\n'
                '[models.voice.synthesis]\nvoice="Cherry"\nspeed=1.2\n',
                encoding="utf-8",
            )
            settings = Settings.load(path)
            public = settings.public()
            self.assertEqual(public["models"], [])
            model = public["synthesis_models"][0]
            self.assertEqual(model["defaults"]["speed"], 1.2)
            self.assertTrue(model["capabilities"]["text_streaming"])
            self.assertNotIn("private-test-key", str(public))
