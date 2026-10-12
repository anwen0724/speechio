"""服务边界只允许申请宿主配置的唤醒，并拒绝手动模式。"""

import unittest

from fastapi.testclient import TestClient

import realtime_asr as sdk
from realtime_asr.server import ServerConfig, create_app
from tests.runtime.test_engine_injection import RecordingEngine
from tests.server.test_service import wait_ready
from tests.session.test_wakeword import WakeDetector


class WakeServiceTests(unittest.TestCase):
    def test_host_capability_and_manual_rejection(self):
        config = ServerConfig(
            engine=RecordingEngine(),
            realtime=False,
            wake_config=sdk.WakeConfig(),
            wake_detector_factory=WakeDetector,
        )
        with TestClient(create_app(config)) as client:
            wait_ready(client)
            capability = client.get("/api/v1/capabilities").json()["wake"]
            self.assertTrue(capability["enabled"])
            self.assertNotIn("access_key", capability)
            with client.websocket_connect("/api/v1/ws") as socket:
                socket.send_json(
                    {
                        "protocol_version": 1,
                        "type": "open",
                        "request_id": "open",
                        "mode": "manual",
                        "realtime": False,
                        "wake": True,
                    }
                )
                self.assertEqual(socket.receive_json()["code"], "MODE_UNAVAILABLE")
