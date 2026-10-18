"""从 HTTP 入口验证本次模型参数，模型执行边界使用记录引擎。"""

import unittest

from fastapi.testclient import TestClient

from apps.web.server.app import create_app
from apps.web.server.settings import Settings
from speechio import ModelConfig
from tests.runtime.test_engine_injection import RecordingEngine


class ModelOptionsTests(unittest.TestCase):
    def test_prepare_overrides_defaults_without_changing_saved_configuration(self):
        original = ModelConfig(engine="whisper", model_path="test-model", compute_type="int8")
        settings = Settings(models={"main": original}, debug=True)
        received = []

        def create_engine(config):
            received.append(config)
            return RecordingEngine()

        with TestClient(create_app(settings, engine_factory=create_engine)) as client:
            response = client.post(
                "/api/debug/prepare",
                json={
                    "model": "main",
                    "input_mode": "file",
                    "device": "gpu",
                    "device_index": 1,
                    "compute_type": "float16",
                },
            )
            self.assertEqual(response.status_code, 200, response.text)
            effective = response.json()["effective_config"]["main"]
            self.assertEqual((effective["device"], effective["compute_type"]), ("gpu", "float16"))
            self.assertNotIn("model_path", effective)
        self.assertEqual((received[0].device, received[0].device_index), ("gpu", 1))
        self.assertEqual(received[0].compute_type, "float16")
        self.assertEqual((original.device, original.compute_type), ("cpu", "int8"))

    def test_realtime_model_uses_its_own_overrides_and_returns_them_on_start(self):
        settings = Settings(
            models={
                "main": ModelConfig(engine="whisper", model_path="main", compute_type="int8"),
                "preview": ModelConfig(engine="whisper", model_path="preview", compute_type="int8"),
            },
            debug=True,
        )
        received = []

        def create_engine(config):
            received.append(config)
            return RecordingEngine()

        with TestClient(create_app(settings, engine_factory=create_engine)) as client:
            with client.websocket_connect("/api/session?debug=1") as socket:
                socket.send_json(
                    {
                        "type": "start",
                        "config": {
                            "model": "main",
                            "device": "gpu",
                            "compute_type": "float16",
                            "realtime_model": "preview",
                            "realtime_device": "cpu",
                            "realtime_compute_type": None,
                            "segmentation": "manual",
                        },
                    }
                )
                message = socket.receive_json()
                self.assertEqual(message["type"], "ready", message)
                effective = message["effective_config"]
                self.assertEqual(effective["main"]["device"], "gpu")
                self.assertEqual(effective["realtime"]["device"], "cpu")
                self.assertIsNone(effective["realtime"]["compute_type"])
                socket.send_json({"type": "close"})
        self.assertEqual(
            {str(item.model_path): (item.device, item.compute_type) for item in received},
            {"main": ("gpu", "float16"), "preview": ("cpu", None)},
        )

    def test_precision_inheritance_default_and_changes_control_model_reuse(self):
        settings = Settings(
            models={
                "main": ModelConfig(engine="whisper", model_path="main", compute_type="int8"),
            },
            debug=True,
        )
        received, engines = [], []

        def create_engine(config):
            received.append(config)
            engine = RecordingEngine()
            engines.append(engine)
            return engine

        with TestClient(create_app(settings, engine_factory=create_engine)) as client:
            base = {"model": "main", "input_mode": "file"}
            for overrides in ({}, {}, {"compute_type": None}, {"device": "gpu"}):
                response = client.post("/api/debug/prepare", json={**base, **overrides})
                self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(
                [(item.device, item.compute_type) for item in received],
                [("cpu", "int8"), ("cpu", None), ("gpu", "int8")],
            )
            self.assertEqual([engine.closes for engine in engines], [1, 1, 0])

    def test_incompatible_overrides_are_rejected_before_loading(self):
        settings = Settings(
            models={
                "native": ModelConfig(engine="nemotron", model_path="native"),
                "main": ModelConfig(engine="whisper", model_path="main"),
            },
            debug=True,
        )

        def unexpected_load(config):
            self.fail("无效参数不能加载模型")

        with TestClient(create_app(settings, engine_factory=unexpected_load)) as client:
            for fields in (
                {"model": "native", "device": "gpu"},
                {"model": "native", "compute_type": "float16"},
                {"model": "main", "realtime_device": "gpu"},
            ):
                with self.subTest(fields=fields):
                    response = client.post("/api/debug/prepare", json=fields)
                    self.assertEqual(response.status_code, 400, response.text)
