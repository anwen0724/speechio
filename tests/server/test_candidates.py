"""公开网络协议验证候选控制交接，不重复核心的文本匹配矩阵。"""

import unittest

from fastapi.testclient import TestClient

import realtime_asr as sdk
from apps.protocol import encode_audio_packet
from apps.server import ServerConfig, create_app
from tests.runtime.test_engine_injection import RecordingEngine
from tests.server.test_service import wait_ready


class CandidateServerTests(unittest.TestCase):
    def test_candidate_only_host_reports_mode_and_rejects_final_http(self):
        engine = RecordingEngine()
        config = ServerConfig(
            candidate_config=sdk.CandidateConfig(engine=engine), final_enabled=False, realtime=False
        )
        with TestClient(create_app(config)) as client:
            wait_ready(client)
            capabilities = client.get("/api/v1/capabilities").json()
            self.assertFalse(capabilities["final_enabled"])
            self.assertEqual(capabilities["candidate"]["completion_modes"], ["candidate"])
            response = client.post(
                "/api/v1/transcribe",
                content=bytes(1024),
                headers={"Content-Type": "application/octet-stream"},
            )
            self.assertEqual(response.status_code, 422)
            self.assertEqual(response.json()["code"], "MODE_UNAVAILABLE")
            self.assertEqual(engine.inputs, [])
        self.assertEqual((engine.preparations, engine.closes), (1, 1))

    def test_candidate_resume_completion_and_correction_share_one_session(self):
        engine = RecordingEngine()
        config = ServerConfig(engine=engine, realtime=False, candidate_config=sdk.CandidateConfig())
        with TestClient(create_app(config)) as client:
            wait_ready(client)
            self.assertTrue(client.get("/api/v1/capabilities").json()["candidate"]["enabled"])
            with client.websocket_connect("/api/v1/ws") as socket:
                socket.send_json(
                    {
                        "protocol_version": 1,
                        "type": "open",
                        "request_id": "open",
                        "realtime": False,
                        "completion_mode": "candidate",
                        "retain_for_correction": True,
                    }
                )
                ready = socket.receive_json()
                self.assertEqual(ready["type"], "ready")
                control = {
                    "protocol_version": 1,
                    "session_id": ready["session_id"],
                    "generation": 0,
                }

                def command(kind, request_id, **fields):
                    socket.send_json({**control, "type": kind, "request_id": request_id, **fields})
                    reply = socket.receive_json()
                    self.assertEqual(reply.get("request_id"), request_id, reply)
                    return reply

                unit = command("start", "start")["unit"]

                def feed(sequence, frames):
                    metadata = {
                        **control,
                        "unit_id": unit["unit_id"],
                        "audio_sequence": sequence,
                        "format": {"sample_rate": 16000, "channels": 1, "sample_type": "int16"},
                        "frame_count": frames,
                    }
                    socket.send_bytes(encode_audio_packet(metadata, bytes(frames * 2)))

                feed(0, 525)
                request = command("candidate", "candidate")["candidate_request"]
                candidate = socket.receive_json()["event"]
                self.assertEqual(candidate["candidate"]["request"], request)
                resume = command("resume", "resume", expected_audio_samples=525)["resume"]
                repeated = command("resume", "resume", expected_audio_samples=525)["resume"]
                self.assertEqual(resume, repeated)
                feed(1, 13)
                command("candidate", "new-candidate")
                current = socket.receive_json()["event"]["candidate"]
                self.assertEqual(
                    (current["request"]["end_sample"], current["request"]["resume_epoch"]), (538, 1)
                )
                final_task = command("finish", "finish")["task"]
                final, done = socket.receive_json()["event"], socket.receive_json()["event"]
                self.assertEqual((final["kind"], done["kind"]), ("final", "completed"))
                self.assertTrue(final["optimization"]["candidate_reused"])
                command("correct", "correct", unit=unit)
                corrected = socket.receive_json()["event"]
                self.assertEqual(corrected["kind"], "correction")
                self.assertEqual(corrected["candidate"]["replaces_task_id"], final_task["task_id"])
                command("start", "next")
                unavailable = command("correct", "old-correct", unit=unit)
                self.assertEqual(unavailable["code"], "CORRECTION_UNAVAILABLE")
        self.assertEqual((engine.preparations, engine.closes), (1, 1))
