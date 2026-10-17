"""服务宿主注入外部引擎，HTTP 与实际核心会话仍执行正式流程。"""

import unittest
from time import monotonic

import numpy as np
from fastapi.testclient import TestClient

from apps.server import ServerConfig, create_app
from realtime_asr import EngineCapabilities, RecognitionResult, WordTimestamp
from tests.runtime.test_engine_injection import RecordingEngine


class TimestampEngine(RecordingEngine):
    @property
    def capabilities(self):
        return EngineCapabilities(engine="company-api", device="remote", word_timestamps=True)

    def transcribe(self, audio, language=None, *, role="final", word_timestamps=False):
        super().transcribe(audio, language, role=role, word_timestamps=word_timestamps)
        return RecognitionResult(
            text="公司识别结果",
            words=(WordTimestamp("公司识别结果", 0, len(audio)),) if word_timestamps else None,
            timestamp_status="reported" if word_timestamps else "not_requested",
        )


class ServiceEngineInjectionTests(unittest.TestCase):
    def test_host_injection_drives_http_and_capabilities_without_local_model_config(self):
        engine = TimestampEngine()
        app = create_app(ServerConfig(engine=engine, realtime=False))
        with TestClient(app) as client:
            deadline = monotonic() + 5
            while client.get("/api/v1/ready").status_code != 200:
                self.assertLess(monotonic(), deadline, "引擎准备应在测试时限内完成")
            capabilities = client.get("/api/v1/capabilities").json()
            self.assertEqual(capabilities["engine"], "company-api")
            self.assertEqual(capabilities["device"], "remote")
            self.assertTrue(capabilities["word_timestamps"])
            self.assertFalse(capabilities["realtime"])
            source = np.arange(525, dtype=np.int16)
            response = client.post(
                "/api/v1/transcribe?word_timestamps=true",
                content=source.tobytes(),
                headers={"Content-Type": "application/octet-stream"},
            )
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["result"]["text"], "公司识别结果")
            self.assertEqual(response.json()["result"]["words"][0]["end_sample"], 525)
            np.testing.assert_array_equal(engine.inputs[0][0], source.astype(np.float32) / 32768)
            self.assertTrue(engine.inputs[0][3])
        self.assertEqual(engine.closes, 1)
