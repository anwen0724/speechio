"""服务通过公开协议选择双实时，核心合并规则由文本和运行时测试保护。"""

import unittest

from fastapi.testclient import TestClient

from apps.protocol import encode_audio_packet
from apps.server import ServerConfig, create_app
from realtime_asr import RealtimeConfig
from tests.runtime.test_dual_realtime import RoleEngine
from tests.server.test_service import wait_ready


class DualRealtimeServerTests(unittest.TestCase):
    def test_host_roles_and_websocket_choice_return_both_sources_and_merge(self):
        accurate, fast = RoleEngine("speech system"), RoleEngine("speech system today")
        config = ServerConfig(
            engine=accurate,
            realtime_config=RealtimeConfig(mode="native", initial_delay=0),
            ultrafast_config=RealtimeConfig(mode="native", engine=fast, initial_delay=0),
        )
        with TestClient(create_app(config)) as client:
            wait_ready(client)
            caps = client.get("/api/v1/capabilities").json()
            self.assertTrue(caps["dual_realtime"])
            with client.websocket_connect("/api/v1/ws") as socket:
                socket.send_json(
                    {
                        "protocol_version": 1,
                        "type": "open",
                        "request_id": "open",
                        "dual_realtime": True,
                    }
                )
                ready = socket.receive_json()
                self.assertEqual(ready["type"], "ready")
                self.assertTrue(ready["text_options"]["dual_realtime"])
                control = {
                    "protocol_version": 1,
                    "session_id": ready["session_id"],
                    "generation": 0,
                }
                socket.send_json({**control, "type": "start", "request_id": "start"})
                ack = socket.receive_json()
                unit = ack["unit"]
                metadata = {
                    **control,
                    "unit_id": unit["unit_id"],
                    "audio_sequence": 0,
                    "format": {"sample_rate": 16000, "channels": 1, "sample_type": "int16"},
                    "frame_count": 525,
                }
                socket.send_bytes(encode_audio_packet(metadata, bytes(1050)))
                socket.send_json(
                    {**control, "type": "finish", "request_id": "finish", "unit": unit}
                )
                roles, merged = set(), []
                while True:
                    message = socket.receive_json()
                    self.assertNotEqual(message["type"], "error", message)
                    if message["type"] != "event":
                        continue
                    event = message["event"]
                    if event["kind"] == "realtime":
                        roles.add(event["realtime_role"])
                        merged.append(event["merge"]["text"])
                    if event["kind"] == "completed":
                        break
                self.assertEqual(roles, {"accurate", "ultrafast"})
                self.assertIn("speech system today", merged)
        self.assertEqual((accurate.closes, fast.closes), (1, 1))
