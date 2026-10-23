"""从 Web 入口验证远程配置和音频交付，不使用真实百炼账户。"""

import io
import tempfile
import unittest
import wave
from pathlib import Path

from fastapi.testclient import TestClient

from apps.web.server.app import create_app
from apps.web.server.settings import Settings
from speechio import LocalModelConfig, RemoteModelConfig
from tests.audio.encoded_samples import encoded_audio
from tests.runtime.test_engine_injection import RecordingEngine, RecordingStreamingEngine
from tests.runtime.test_remote import FLASH, STREAMING, http_service
from tests.runtime.test_remote_file import MODEL as FILE_MODEL
from tests.runtime.test_remote_file import file_service


class WebRemoteTests(unittest.TestCase):
    def test_file_model_is_public_but_rejected_for_continuous_input(self):
        model = RemoteModelConfig(provider="bailian", model=FILE_MODEL, api_key="test-secret")
        with TestClient(create_app(Settings(models={"long": model}, debug=True))) as client:
            public = client.get("/api/config")
            self.assertEqual(public.json()["upload_limit_bytes"], 1024**3)
            capabilities = public.json()["models"][0]
            self.assertTrue(capabilities["file_transcription"])
            self.assertFalse(capabilities["transcription"])
            self.assertEqual(capabilities["task_timeout_seconds"], 3600)
            self.assertNotIn("test-secret", public.text)
            for mode in ("live", "chunks"):
                with self.subTest(mode=mode):
                    response = client.post(
                        "/api/debug/prepare",
                        json={
                            "model": "long",
                            "input_mode": mode,
                            "realtime": False,
                        },
                    )
                    self.assertEqual(response.status_code, 400, response.text)

    def test_long_audio_upload_returns_final_result_and_cleans_file(self):
        self.check_long_audio_upload()

    def test_long_audio_failure_reports_error_and_cleans_file(self):
        self.check_long_audio_upload(failed=True)

    def check_long_audio_upload(self, *, failed=False):
        with (
            tempfile.TemporaryDirectory() as directory,
            file_service(states=("FAILED",) if failed else ("SUCCEEDED",)) as (url, _),
        ):
            model = RemoteModelConfig(
                provider="bailian", model=FILE_MODEL, api_key="test-secret", base_url=url
            )
            settings = Settings(models={"long": model}, upload_dir=Path(directory))
            with TestClient(create_app(settings)) as client:
                response = client.post(
                    "/api/uploads",
                    content=encoded_audio(),
                    headers={"X-Audio-Extension": ".m4a"},
                )
                self.assertEqual(response.status_code, 200, response.text)
                with client.websocket_connect("/api/session") as socket:
                    socket.send_json(
                        {
                            "type": "start",
                            "file_token": response.json()["token"],
                            "config": {
                                "model": "long",
                                "input_mode": "file",
                                "task_timeout_seconds": 120,
                            },
                        }
                    )
                    ready = socket.receive_json()
                    self.assertEqual(ready["type"], "ready", ready)
                    self.assertEqual(ready["effective_config"]["main"]["task_timeout_seconds"], 120)
                    final = None
                    while True:
                        event = socket.receive_json()
                        if failed and event["type"] == "error":
                            self.assertTrue(event["fatal"])
                            self.assertNotIn("test-secret", str(event))
                            break
                        self.assertNotEqual(event["type"], "error", event)
                        if event.get("kind") == "final":
                            final = event["result"]["text"]
                        if event["type"] == "finished":
                            self.assertFalse(failed, "失败任务必须报告错误")
                            break
                    if not failed:
                        socket.send_json({"type": "close"})
                self.assertEqual(final, None if failed else "你好 hello")
                self.assertEqual(model.task_timeout_seconds, 3600)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_independent_remote_timeouts_reach_engines_and_public_configuration(self):
        for mode in ("auto", "final"):
            with self.subTest(completion_mode=mode):
                self.check_remote_completion(mode)

    def check_remote_completion(self, mode):
        main = RemoteModelConfig(provider="bailian", model=FLASH, api_key="test-secret")
        preview = RemoteModelConfig(provider="bailian", model=STREAMING, api_key="test-secret")
        received = []
        engines = {}

        def factory(config):
            received.append(config)
            engine = RecordingStreamingEngine() if config.model == STREAMING else RecordingEngine()
            engines[config.model] = engine
            return engine

        with TestClient(
            create_app(
                Settings(models={"main": main, "preview": preview}, debug=True),
                engine_factory=factory,
            )
        ) as client:
            with client.websocket_connect("/api/session?debug=1") as socket:
                socket.send_json(
                    {
                        "type": "start",
                        "config": {
                            "model": "main",
                            "timeout_seconds": 12,
                            "realtime_model": "preview",
                            "realtime_timeout_seconds": 18,
                            "realtime_mode": "streaming",
                            "completion_mode": mode,
                            "segmentation": "manual",
                        },
                    }
                )
                ready = socket.receive_json()
                self.assertEqual(ready["type"], "ready", ready)
                self.assertEqual(ready["effective_config"]["main"]["timeout_seconds"], 12)
                self.assertEqual(ready["effective_config"]["realtime"]["timeout_seconds"], 18)
                self.assertNotIn("test-secret", str(ready))
                socket.send_bytes(bytes(1050))
                socket.send_json({"type": "end_input"})
                events = []
                while True:
                    event = socket.receive_json()
                    self.assertNotEqual(event["type"], "error", event)
                    events.append(event)
                    if event["type"] == "finished":
                        break
                self.assertTrue(any(e.get("stream_finished") for e in events))
                self.assertTrue(any(e.get("kind") == "final" for e in events))
                final = next(e for e in events if e.get("kind") == "final")
                self.assertEqual(
                    final["result"]["text"],
                    "完整实时结果" if mode == "auto" else "收到完整音频",
                )
                self.assertEqual(
                    ready["effective_config"]["completion_source"],
                    "streaming" if mode == "auto" else "final",
                )
                self.assertEqual(len(engines[FLASH].inputs), int(mode == "final"))
                self.assertEqual(engines[FLASH].preparations, int(mode == "final"))
                self.assertEqual(len(engines[STREAMING].streams), 1)
                socket.send_json({"type": "close"})
        self.assertEqual({c.model: c.timeout_seconds for c in received}, {FLASH: 12, STREAMING: 18})
        self.assertEqual((main.timeout_seconds, preview.timeout_seconds), (30, 30))

    def test_uploaded_file_reaches_core_remote_adapter(self):
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as output:
            output.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
            output.writeframes(bytes(3200))
        with (
            tempfile.TemporaryDirectory() as directory,
            http_service({"output": {"text": "你好 hello"}}) as (url, requests),
        ):
            settings = Settings(
                models={
                    "cloud": RemoteModelConfig(
                        provider="bailian",
                        model=FLASH,
                        api_key="test-secret",
                        base_url=url,
                    )
                },
                upload_dir=Path(directory),
            )
            with TestClient(create_app(settings)) as client:
                token = client.post("/api/uploads", content=buffer.getvalue()).json()["token"]
                with client.websocket_connect("/api/session") as socket:
                    socket.send_json(
                        {
                            "type": "start",
                            "file_token": token,
                            "config": {
                                "model": "cloud",
                                "input_mode": "file",
                                "realtime": False,
                            },
                        }
                    )
                    final = None
                    while True:
                        event = socket.receive_json()
                        self.assertNotEqual(event["type"], "error", event)
                        if event.get("kind") == "final":
                            final = event["result"]["text"]
                        if event["type"] == "finished":
                            self.assertIsNot(event.get("success"), False)
                            break
                    socket.send_json({"type": "close"})
                self.assertEqual(final, "你好 hello")
                self.assertEqual(len(requests), 1)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_inapplicable_parameters_and_missing_key_fail_before_model_use(self):
        settings = Settings(
            models={
                "cloud": RemoteModelConfig(provider="bailian", model=FLASH, api_key="test-secret"),
                "unconfigured": RemoteModelConfig(provider="bailian", model=FLASH),
                "local": LocalModelConfig(engine="whisper", model_path="unused"),
            },
            debug=True,
        )

        def unexpected(config):
            self.fail("不适用的配置不能准备模型")

        with TestClient(create_app(settings, engine_factory=unexpected)) as client:
            self.assertEqual(client.get("/api/config").status_code, 200)
            for fields in (
                {"model": "cloud", "device": "cpu"},
                {"model": "local", "timeout_seconds": 10},
                {"model": "local", "task_timeout_seconds": 10},
                {"model": "cloud", "task_timeout_seconds": 10},
                {"model": "cloud", "realtime_timeout_seconds": 10},
                {"model": "unconfigured"},
            ):
                with self.subTest(fields=fields):
                    response = client.post(
                        "/api/debug/prepare",
                        json={
                            "input_mode": "file",
                            **fields,
                        },
                    )
                    self.assertEqual(response.status_code, 400, response.text)
                    self.assertNotIn("test-secret", response.text)
