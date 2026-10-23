"""从网络入口验证应用行为，只在模型边界使用替身。"""

import io
import tempfile
import unittest
from pathlib import Path
from threading import Event

import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

from apps.web.server.app import create_app
from apps.web.server.settings import Settings
from speechio import LocalModelConfig
from tests.audio.encoded_samples import encoded_audio
from tests.runtime.test_engine_injection import RecordingEngine


def make_settings(**options):
    """路由测试显式提供模型，不读取开发者的本机配置。"""
    return Settings(
        models={"whisper-tiny": LocalModelConfig(adapter="whisper", model_path="test-model")},
        **options,
    )


class WebAppTests(unittest.TestCase):
    def test_m4a_upload_reaches_model_in_file_and_debug_chunks_modes(self):
        content = encoded_audio()
        for mode in ("file", "chunks"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                engine = RecordingEngine()
                settings = make_settings(debug=True, upload_dir=Path(directory))
                with TestClient(create_app(settings, engine_factory=lambda _: engine)) as client:
                    token = client.post("/api/uploads", content=content).json()["token"]
                    with client.websocket_connect("/api/session?debug=1") as socket:
                        socket.send_json(
                            {
                                "type": "start",
                                "file_token": token,
                                "config": {
                                    "model": "whisper-tiny",
                                    "input_mode": mode,
                                    "segmentation": "manual",
                                    "realtime": False,
                                    "pace": "fast",
                                },
                            }
                        )
                        while True:
                            message = socket.receive_json()
                            self.assertNotEqual(message["type"], "error", message)
                            if message["type"] == "finished":
                                break
                        socket.send_json({"type": "close"})
                    self.assertGreaterEqual(len(engine.inputs[0][0]), 8000)
                    self.assertEqual(list(Path(directory).iterdir()), [])

    def test_active_task_rejects_another_page_without_interrupting_owner(self):
        with TestClient(
            create_app(make_settings(), engine_factory=lambda _: RecordingEngine())
        ) as client:
            with client.websocket_connect("/api/session") as owner:
                start = {
                    "type": "start",
                    "config": {
                        "model": "whisper-tiny",
                        "segmentation": "manual",
                        "realtime": False,
                    },
                }
                owner.send_json(start)
                self.assertEqual(owner.receive_json()["type"], "ready")
                with client.websocket_connect("/api/session") as other:
                    other.send_json(start)
                    self.assertEqual(other.receive_json()["code"], "WEB_BUSY")
                owner.send_bytes(np.ones(100, dtype=np.int16).tobytes())
                owner.send_json({"type": "end_input"})
                while owner.receive_json()["type"] != "finished":
                    pass
                owner.send_json({"type": "close"})

    def test_manual_wake_and_cross_origin_requests_are_rejected(self):
        with TestClient(create_app(make_settings())) as client:
            self.assertEqual(
                client.post(
                    "/api/uploads",
                    content=b"x",
                    headers={
                        "Origin": "https://unrelated.example",
                    },
                ).status_code,
                403,
            )
            with client.websocket_connect("/api/session") as socket:
                socket.send_json(
                    {
                        "type": "start",
                        "config": {
                            "model": "whisper-tiny",
                            "segmentation": "manual",
                            "wake": "wake-preset",
                        },
                    }
                )
                self.assertEqual(socket.receive_json()["type"], "error")

    def test_debug_chunks_deliver_complete_file_and_support_correction(self):
        engine = RecordingEngine()
        buffer = io.BytesIO()
        sf.write(buffer, np.arange(1625, dtype=np.int16), 16000, format="WAV")
        with tempfile.TemporaryDirectory() as directory:
            settings = make_settings(debug=True, upload_dir=Path(directory))
            with TestClient(create_app(settings, engine_factory=lambda _: engine)) as client:
                token = client.post("/api/uploads", content=buffer.getvalue()).json()["token"]
                with client.websocket_connect("/api/session?debug=1") as socket:
                    socket.send_json(
                        {
                            "type": "start",
                            "file_token": token,
                            "config": {
                                "model": "whisper-tiny",
                                "input_mode": "chunks",
                                "segmentation": "manual",
                                "realtime": False,
                                "candidates": True,
                                "correction": True,
                                "pace": "fast",
                            },
                        }
                    )
                    while True:
                        message = socket.receive_json()
                        self.assertNotEqual(message["type"], "error", message)
                        if message["type"] == "finished":
                            break
                    self.assertEqual(len(engine.inputs[0][0]), 1625)
                    socket.send_json({"type": "correct"})
                    while True:
                        message = socket.receive_json()
                        self.assertNotEqual(message["type"], "error", message)
                        if message.get("kind") == "correction":
                            self.assertEqual(message["result"]["text"], "收到完整音频")
                            break
                    socket.send_json({"type": "close"})
                self.assertEqual(list(Path(directory).iterdir()), [])

    def test_uploaded_file_transcribes_and_is_removed(self):
        engine = RecordingEngine()
        buffer = io.BytesIO()
        sf.write(buffer, np.arange(800, dtype=np.int16), 16000, format="WAV")
        with tempfile.TemporaryDirectory() as directory:
            settings = make_settings(upload_dir=Path(directory))
            with TestClient(create_app(settings, engine_factory=lambda _: engine)) as client:
                token = client.post("/api/uploads", content=buffer.getvalue()).json()["token"]
                with client.websocket_connect("/api/session") as socket:
                    socket.send_json(
                        {
                            "type": "start",
                            "file_token": token,
                            "config": {
                                "model": "whisper-tiny",
                                "input_mode": "file",
                                "realtime": False,
                            },
                        }
                    )
                    messages = []
                    while True:
                        message = socket.receive_json()
                        messages.append(message)
                        if message["type"] in ("finished", "error"):
                            break
                    self.assertEqual(messages[-1]["type"], "finished", messages)
                    self.assertEqual(messages[-2]["result"]["text"], "收到完整音频")
                    socket.send_json({"type": "close"})
                self.assertEqual(list(Path(directory).iterdir()), [])

    def test_upload_limit_rejects_and_removes_partial_file(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = make_settings(upload_dir=Path(directory), upload_limit_bytes=8)
            with TestClient(create_app(settings)) as client:
                response = client.post("/api/uploads", content=b"012345678")
                self.assertEqual(response.status_code, 413)
                self.assertEqual(list(Path(directory).iterdir()), [])

    def test_upload_extension_cannot_supply_a_server_path(self):
        with tempfile.TemporaryDirectory() as directory:
            with TestClient(create_app(make_settings(upload_dir=Path(directory)))) as client:
                invalid = client.post(
                    "/api/uploads",
                    content=b"audio",
                    headers={"X-Audio-Extension": "../../recording.m4a"},
                )
                self.assertEqual(invalid.status_code, 400)
                self.assertEqual(list(Path(directory).iterdir()), [])
                valid = client.post(
                    "/api/uploads",
                    content=b"audio",
                    headers={"X-Audio-Extension": ".m4a"},
                )
                self.assertEqual(valid.status_code, 200)
                client.delete(f"/api/uploads/{valid.json()['token']}")
                self.assertEqual(list(Path(directory).iterdir()), [])

    def test_disconnect_closes_owned_engine_and_allows_next_task(self):
        closed = Event()

        class Engine(RecordingEngine):
            def close(self):
                super().close()
                closed.set()

        with TestClient(create_app(make_settings(), engine_factory=lambda _: Engine())) as client:
            with client.websocket_connect("/api/session") as socket:
                socket.send_json(
                    {
                        "type": "start",
                        "config": {
                            "model": "whisper-tiny",
                            "segmentation": "manual",
                            "realtime": False,
                        },
                    }
                )
                self.assertEqual(socket.receive_json()["type"], "ready")
            self.assertTrue(closed.wait(2), "断线后必须释放会话拥有的模型资源")
            with client.websocket_connect("/api/session") as socket:
                socket.send_json(
                    {
                        "type": "start",
                        "config": {
                            "model": "whisper-tiny",
                            "segmentation": "manual",
                            "realtime": False,
                        },
                    }
                )
                self.assertEqual(socket.receive_json()["type"], "ready")
                socket.send_json({"type": "close"})

    def test_stop_delivers_tail_audio_before_finished(self):
        engine = RecordingEngine()
        with TestClient(create_app(make_settings(), engine_factory=lambda _: engine)) as client:
            with client.websocket_connect("/api/session") as socket:
                socket.send_json(
                    {
                        "type": "start",
                        "config": {
                            "model": "whisper-tiny",
                            "segmentation": "manual",
                            "realtime": False,
                        },
                    }
                )
                self.assertEqual(socket.receive_json()["type"], "ready")
                source = np.arange(525, dtype=np.int16)
                socket.send_bytes(source[:512].tobytes())
                socket.send_bytes(source[512:].tobytes())
                socket.send_json({"type": "end_input"})
                events = []
                while True:
                    message = socket.receive_json()
                    events.append(message)
                    if message["type"] == "finished":
                        break
                final = next(item for item in events if item.get("kind") == "final")
                self.assertEqual(final["result"]["text"], "收到完整音频")
                np.testing.assert_array_equal(engine.inputs[0][0], source / 32768)
                socket.send_json({"type": "close"})
        self.assertEqual(engine.closes, 1)

    def test_configuration_exposes_presets_but_disables_debug(self):
        with TestClient(create_app(make_settings())) as client:
            response = client.get("/api/config")
            self.assertEqual(response.status_code, 200)
            config = response.json()
            self.assertFalse(config["debug"])
            self.assertEqual(config["models"][0]["id"], "whisper-tiny")
            self.assertNotIn("model_path", config["models"][0])
            self.assertEqual(client.get("/debug").status_code, 404)
            self.assertEqual(client.post("/api/debug/prepare", json={}).status_code, 404)


if __name__ == "__main__":
    unittest.main()
