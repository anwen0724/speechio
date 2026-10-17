"""服务的状态、HTTP 与连接契约；只控制外部模型边界。"""

import unittest
from dataclasses import replace
from threading import Event, Thread
from time import monotonic
from types import SimpleNamespace
from unittest.mock import patch

import soxr
from fastapi.testclient import TestClient

from realtime_asr import RecognizerConfig
from realtime_asr.server import ServerConfig, create_app
from tests.runtime import test_recognizer


def wait_ready(client, *, headers=None):
    """轮询公开就绪条件，截止时间只用于防止失败测试挂起。"""
    deadline = monotonic() + 5
    while monotonic() < deadline:
        result = client.get("/api/v1/ready", headers=headers)
        if result.status_code == 200:
            return
    raise AssertionError("服务未就绪")


class ServiceTests(unittest.TestCase):
    def setUp(self):
        test_recognizer.RecognizerTests.setUp(self)
        self.config = ServerConfig(recognizer=RecognizerConfig(model=self.path), realtime=False)

    def test_loading_model_does_not_block_live_or_claim_readiness(self):
        entered, release = Event(), Event()

        def load(*args, **kwargs):
            entered.set()
            if not release.wait(5):
                raise OSError("测试没有释放外部加载")
            return SimpleNamespace(feature_extractor=SimpleNamespace(chunk_length=30))

        with patch("faster_whisper.WhisperModel", side_effect=load):
            with TestClient(create_app(self.config)) as client:
                try:
                    self.assertEqual(client.get("/api/v1/live").status_code, 200)
                    self.assertTrue(entered.wait(5))
                    self.assertEqual(client.get("/api/v1/ready").status_code, 503)
                    release.set()
                    wait_ready(client)
                finally:
                    release.set()

    def test_metrics_waiting_on_sdk_conversion_does_not_block_other_connection(self):
        from tests.server.test_websocket import command, open_socket

        entered, release, metrics_started, pong_received = (Event() for _ in range(4))
        original = soxr.ResampleStream.resample_chunk
        responses, errors = [], []

        def resample(stream, audio, last=False):
            if not last:
                entered.set()
                if not release.wait(15):
                    raise RuntimeError("转换边界未释放")
            return original(stream, audio, last=last)

        inner = create_app(self.config)

        async def application(scope, receive, send):
            if scope.get("path") == "/api/v1/metrics":
                metrics_started.set()
            await inner(scope, receive, send)

        with (
            patch("soxr.ResampleStream.resample_chunk", resample),
            patch(
                "faster_whisper.BatchedInferencePipeline.transcribe",
                return_value=(iter([]), SimpleNamespace(language=None, language_probability=None)),
            ),
        ):
            with TestClient(application) as app:
                wait_ready(app)
                with app.websocket_connect("/api/v1/ws") as socket:
                    ready = open_socket(socket)

                    def upload():
                        responses.append(
                            app.post(
                                "/api/v1/transcribe",
                                content=bytes(1545 * 2),
                                params={"sample_rate": 48000},
                                headers={"Content-Type": "application/octet-stream"},
                            )
                        )

                    def metrics():
                        responses.append(app.get("/api/v1/metrics"))

                    def ping():
                        try:
                            command(socket, ready, "ping", "during-conversion")
                            if socket.receive_json()["type"] == "pong":
                                pong_received.set()
                        except Exception as error:
                            errors.append(error)

                    threads = [Thread(target=upload), Thread(target=metrics), Thread(target=ping)]
                    try:
                        threads[0].start()
                        self.assertTrue(entered.wait(5))
                        threads[1].start()
                        self.assertTrue(metrics_started.wait(5))
                        threads[2].start()
                        responsive = pong_received.wait(1)
                    finally:
                        release.set()
                        for thread in threads:
                            if thread.ident is not None:
                                thread.join(5)
                    self.assertTrue(responsive, "指标查询阻塞了其他连接")
                    self.assertEqual(errors, [])
                    self.assertTrue(all(response.status_code == 200 for response in responses))

    def test_upload_disconnect_reclaims_session_without_unhandled_server_error(self):
        inner = create_app(self.config)

        async def application(scope, receive, send):
            received = False

            async def transport():
                nonlocal received
                if scope.get("path") != "/api/v1/transcribe":
                    return await receive()
                if not received:
                    received = True
                    return {"type": "http.request", "body": b"\x00", "more_body": True}
                return {"type": "http.disconnect"}

            await inner(scope, transport, send)

        with TestClient(application) as app:
            wait_ready(app)
            response = app.post(
                "/api/v1/transcribe",
                content=b"\x00",
                headers={"Content-Type": "application/octet-stream"},
            )
            self.assertEqual(response.status_code, 499)
            self.assertEqual(response.json()["code"], "CLIENT_DISCONNECTED")
            self.assertEqual(app.get("/api/v1/metrics").json()["sessions_open"], 0)

    def test_http_raw_audio_delivers_final_and_reclaims_its_session(self):
        def transcribe(audio, **kwargs):
            return iter([SimpleNamespace(text="正文", start=0, end=0.032)]), (
                SimpleNamespace(language="zh", language_probability=1)
            )

        with patch("faster_whisper.BatchedInferencePipeline.transcribe", side_effect=transcribe):
            with TestClient(create_app(self.config)) as client:
                wait_ready(client)
                response = client.post(
                    "/api/v1/transcribe",
                    content=b"\x01\x00" * 515,
                    headers={"content-type": "application/octet-stream"},
                    params={"language": "zh-CN"},
                )
                self.assertEqual(response.status_code, 200)
                record = response.json()
                self.assertEqual(record["event"]["kind"], "final")
                self.assertEqual(record["result"]["text"], "正文")
                self.assertEqual(record["event"]["audio_samples"], 515)
                self.assertEqual(client.get("/api/v1/metrics").json()["sessions_open"], 0)

    def test_truncated_request_is_rejected_without_poisoning_the_shared_runtime(self):
        with TestClient(create_app(self.config)) as client:
            wait_ready(client)
            headers = {"content-type": "application/octet-stream"}
            rejected = client.post("/api/v1/transcribe", content=b"\x01", headers=headers)
            self.assertEqual(rejected.status_code, 422)
            self.assertEqual(rejected.json()["code"], "AUDIO_INPUT_TRUNCATED")
            self.assertEqual(client.get("/api/v1/metrics").json()["sessions_open"], 0)
            result = (iter([]), SimpleNamespace(language="en", language_probability=1))
            with patch("faster_whisper.BatchedInferencePipeline.transcribe", return_value=result):
                accepted = client.post(
                    "/api/v1/transcribe", content=b"\x00\x00" * 512, headers=headers
                )
            self.assertEqual(accepted.status_code, 200)
            self.assertEqual(accepted.json()["result"]["text"], "")

    def test_unauthorized_requests_do_not_allocate_sessions_or_echo_the_token(self):
        config = replace(self.config, token="test-secret")
        with TestClient(create_app(config)) as client:
            self.assertEqual(client.get("/api/v1/live").status_code, 200)
            response = client.post(
                "/api/v1/transcribe",
                content=b"\x01",
                headers={"content-type": "application/octet-stream"},
            )
            self.assertEqual(response.status_code, 401)
            self.assertNotIn("test-secret", response.text)
            metrics = client.get(
                "/api/v1/metrics", headers={"Authorization": "Bearer test-secret"}
            ).json()
            self.assertEqual(metrics["sessions_open"], 0)

    def test_slow_inference_does_not_block_readiness_and_capacity_is_reclaimed(self):
        entered, release = Event(), Event()

        def transcribe(audio, **kwargs):
            entered.set()
            if not release.wait(5):
                raise OSError("测试未释放推理")
            return iter([]), SimpleNamespace(language="en", language_probability=1)

        config = replace(self.config, max_sessions=1)
        responses = []
        with patch("faster_whisper.BatchedInferencePipeline.transcribe", side_effect=transcribe):
            with TestClient(create_app(config)) as client:
                wait_ready(client)

                def post():
                    responses.append(
                        client.post(
                            "/api/v1/transcribe",
                            content=b"\x00\x00" * 512,
                            headers={"content-type": "application/octet-stream"},
                        )
                    )

                request = Thread(target=post)
                try:
                    request.start()
                    self.assertTrue(entered.wait(5))
                    self.assertEqual(client.get("/api/v1/ready").status_code, 200)
                    rejected = client.post(
                        "/api/v1/transcribe",
                        content=b"\x00\x00" * 512,
                        headers={"content-type": "application/octet-stream"},
                    )
                    self.assertEqual(rejected.status_code, 429)
                finally:
                    release.set()
                    request.join(5)
                self.assertEqual(responses[0].status_code, 200)
                self.assertEqual(client.get("/api/v1/metrics").json()["sessions_open"], 0)

    def test_runtime_config_rejects_startup_fields_atomically(self):
        with TestClient(create_app(self.config)) as client:
            wait_ready(client)
            initial = client.get("/api/v1/config")
            self.assertEqual(initial.status_code, 200)
            rejected = client.patch(
                "/api/v1/config", json={"idle_timeout": 1, "backend": "nemotron"}
            )
            self.assertEqual(rejected.status_code, 422)
            self.assertEqual(
                client.patch("/api/v1/config", json={"max_audio_seconds": None}).status_code, 422
            )
            self.assertEqual(client.get("/api/v1/config").json()["values"]["idle_timeout"], 300)
            updated = client.patch(
                "/api/v1/config", json={"idle_timeout": 1, "max_audio_seconds": 10}
            )
            self.assertEqual(updated.status_code, 200)
            self.assertEqual(updated.json()["values"]["idle_timeout"], 1)
            self.assertEqual(updated.json()["values"]["max_audio_seconds"], 10)
