"""服务边界的拒绝、超时和清理；只在外部模型边界控制等待。"""

import asyncio
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Event, Thread
from time import monotonic
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

from realtime_asr import ASRError, RecognizerConfig
from realtime_asr.server import ServerConfig, create_app
from tests.runtime import test_recognizer
from tests.server.test_service import wait_ready
from tests.server.test_websocket import open_socket


class LifecycleTests(unittest.TestCase):
    def setUp(self):
        test_recognizer.RecognizerTests.setUp(self)
        self.config = ServerConfig(
            recognizer=RecognizerConfig(model_path=self.path), realtime=False
        )

    def test_prepare_failure_is_visible_without_claiming_ready(self):
        with patch("faster_whisper.WhisperModel", side_effect=RuntimeError("load failed")):
            with TestClient(create_app(self.config)) as app:
                deadline = monotonic() + 5
                while monotonic() < deadline:
                    response = app.get("/api/v1/ready")
                    if response.json()["phase"] == "failed":
                        break
                self.assertEqual(response.status_code, 503)
                self.assertEqual(response.json()["phase"], "failed")
                self.assertIsNotNone(response.json()["error"])
                self.assertEqual(app.get("/api/v1/live").status_code, 200)

    def test_prepare_finishing_after_asgi_shutdown_still_releases_owned_model(self):
        entered, release, shutdown_complete, executor_stopping, model_released = (
            Event() for _ in range(5)
        )
        failures = []

        class Model:
            feature_extractor = SimpleNamespace(chunk_length=30)

            def __del__(self):
                model_released.set()

        def load(*args, **kwargs):
            entered.set()
            if not release.wait(5):
                raise RuntimeError("加载边界未释放")
            return Model()

        class Executor(ThreadPoolExecutor):
            def shutdown(self, wait=True, **kwargs):
                # asyncio.run 已取消循环任务后才关闭执行器，明确建立迟到时序。
                executor_stopping.set()
                super().shutdown(wait=wait, **kwargs)

        async def lifecycle():
            asyncio.get_running_loop().set_default_executor(Executor(max_workers=4))
            app = create_app(replace(self.config, close_timeout=0.05))
            inbox = asyncio.Queue()
            await inbox.put({"type": "lifespan.startup"})

            async def send(message):
                if message["type"] == "lifespan.startup.complete":
                    if not await asyncio.to_thread(entered.wait, 5):
                        raise AssertionError("加载没有开始")
                    await inbox.put({"type": "lifespan.shutdown"})
                if message["type"] == "lifespan.shutdown.complete":
                    shutdown_complete.set()

            await app({"type": "lifespan", "asgi": {"version": "3.0"}}, inbox.get, send)

        def run():
            try:
                asyncio.run(lifecycle())
            except BaseException as error:
                failures.append(error)

        with patch("faster_whisper.WhisperModel", side_effect=load):
            thread = Thread(target=run)
            thread.start()
            try:
                self.assertTrue(shutdown_complete.wait(5), "ASGI 关闭未遵守时限")
                self.assertTrue(executor_stopping.wait(5))
            finally:
                release.set()
                thread.join(5)
            self.assertFalse(thread.is_alive())
            self.assertEqual(failures, [])
            self.assertTrue(model_released.wait(1), "加载迟到后自有模型未释放")

    def test_ws_token_and_origin_are_checked_before_allocating_session(self):
        config = replace(self.config, token="secret", allowed_origins=("https://accepted.example",))
        with TestClient(create_app(config)) as app:
            headers = {"Authorization": "Bearer secret"}
            wait_ready(app, headers=headers)
            for origin, token, expected in (
                ("https://rejected.example", "secret", "ORIGIN_REJECTED"),
                ("https://accepted.example", "wrong", "UNAUTHORIZED"),
            ):
                with self.subTest(expected=expected):
                    with app.websocket_connect("/api/v1/ws", headers={"Origin": origin}) as socket:
                        error = open_socket(socket, token=token)
                        self.assertEqual(error["code"], expected)
                        self.assertNotIn("secret", str(error))
            with app.websocket_connect(
                "/api/v1/ws", headers={"Origin": "https://accepted.example"}
            ) as socket:
                self.assertEqual(open_socket(socket, token="secret")["type"], "ready")
            self.assertEqual(app.get("/api/v1/metrics", headers=headers).json()["sessions_open"], 0)

    def test_idle_timeout_closes_only_the_idle_session_and_releases_capacity(self):
        with TestClient(create_app(replace(self.config, idle_timeout=0.05))) as app:
            wait_ready(app)
            with app.websocket_connect("/api/v1/ws") as socket:
                open_socket(socket)
                self.assertEqual(socket.receive_json()["code"], "IDLE_TIMEOUT")
            self.assertEqual(app.get("/api/v1/ready").status_code, 200)
            self.assertEqual(app.get("/api/v1/metrics").json()["sessions_open"], 0)

    def test_http_drain_timeout_invalidates_request_without_freeing_busy_model(self):
        entered, release = Event(), Event()

        def inference(audio, **kwargs):
            entered.set()
            if not release.wait(5):
                raise RuntimeError("模型边界未释放")
            return iter([]), SimpleNamespace(language=None, language_probability=None)

        with patch("faster_whisper.BatchedInferencePipeline.transcribe", side_effect=inference):
            with TestClient(create_app(replace(self.config, drain_timeout=0.05))) as app:
                wait_ready(app)
                try:
                    response = app.post(
                        "/api/v1/transcribe",
                        content=b"\x01\x00" * 515,
                        headers={"Content-Type": "application/octet-stream"},
                    )
                    self.assertTrue(entered.is_set())
                    self.assertEqual(response.status_code, 504)
                    self.assertEqual(response.json()["code"], "DRAIN_TIMEOUT")
                    self.assertEqual(app.get("/api/v1/metrics").json()["sessions_open"], 0)
                    self.assertEqual(app.get("/api/v1/ready").status_code, 200)
                finally:
                    release.set()

    def test_remote_bind_requires_auth_and_explicit_tls_route(self):
        for fields in (
            {"host": "0.0.0.0"},
            {"host": "0.0.0.0", "token": "secret"},
            {"host": "0.0.0.0", "token": "secret", "proxy_tls": True, "trusted_proxy_ips": "*"},
        ):
            with self.subTest(fields=fields), self.assertRaises(ASRError):
                replace(self.config, **fields)
        config = replace(
            self.config,
            host="0.0.0.0",
            token="secret",
            proxy_tls=True,
            trusted_proxy_ips="127.0.0.1",
        )
        self.assertTrue(config.proxy_tls)
