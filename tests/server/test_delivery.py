"""在 ASGI 传输边界阻塞发送，验证有界输出与连接释放。"""

import asyncio
import json
import unittest
from threading import Event
from time import monotonic

from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from realtime_asr import RecognizerConfig
from realtime_asr.server import ServerConfig, create_app
from tests.runtime import test_recognizer
from tests.server.test_service import wait_ready
from tests.server.test_websocket import command, open_socket


class DeliveryTests(unittest.TestCase):
    def setUp(self):
        test_recognizer.RecognizerTests.setUp(self)

    def test_send_timeout_releases_only_its_connection(self):
        entered, release = Event(), Event()
        inner = create_app(
            ServerConfig(
                recognizer=RecognizerConfig(model_path=self.path), realtime=False, send_timeout=0.05
            )
        )

        async def application(scope, receive, send):
            async def transport(message):
                if (
                    message["type"] == "websocket.send"
                    and json.loads(message["text"]).get("request_id") == "stall"
                ):
                    entered.set()
                    await asyncio.to_thread(release.wait, 5)
                await send(message)

            await inner(scope, receive, transport)

        with TestClient(application) as app:
            wait_ready(app)
            try:
                with app.websocket_connect("/api/v1/ws") as first:
                    one = open_socket(first)
                    with app.websocket_connect("/api/v1/ws") as second:
                        two = open_socket(second)
                        command(first, one, "ping", "stall")
                        self.assertTrue(entered.wait(5))
                        with self.assertRaises(WebSocketDisconnect) as closed:
                            first.receive_json()
                        self.assertEqual(closed.exception.code, 1013)
                        command(second, two, "ping", "other")
                        self.assertEqual(second.receive_json()["type"], "pong")
                metrics = app.get("/api/v1/metrics").json()
                self.assertEqual(metrics["counts"]["send_timeouts"], 1)
                self.assertEqual(metrics["sessions_open"], 0)
            finally:
                release.set()

    def test_full_critical_output_reports_slow_consumer_and_releases_session(self):
        entered, release = Event(), Event()
        inner = create_app(
            ServerConfig(
                recognizer=RecognizerConfig(model_path=self.path),
                realtime=False,
                max_pending_messages=4,
                send_timeout=5,
                idle_timeout=5,
            )
        )

        async def application(scope, receive, send):
            async def transport(message):
                if message["type"] == "websocket.send":
                    record = json.loads(message["text"])
                    if record.get("request_id") == "ping-0":
                        entered.set()
                        if not await asyncio.to_thread(release.wait, 5):
                            raise OSError("测试未释放传输")
                await send(message)

            await inner(scope, receive, transport)

        with TestClient(application) as client:
            wait_ready(client)
            try:
                with client.websocket_connect("/api/v1/ws") as socket:
                    ready = open_socket(socket)
                    command(socket, ready, "ping", "ping-0")
                    self.assertTrue(entered.wait(5))
                    for index in range(1, 8):
                        command(socket, ready, "ping", f"ping-{index}")
                    deadline = monotonic() + 5
                    while client.get("/api/v1/metrics").json()["counts"].get("errors", 0) == 0:
                        self.assertLess(monotonic(), deadline)
                    self.assertEqual(client.get("/api/v1/ready").status_code, 200)
                    release.set()
                    messages = [socket.receive_json()]
                    while messages[-1]["type"] != "error":
                        messages.append(socket.receive_json())
                    self.assertEqual(messages[-1]["code"], "SLOW_CONSUMER")
            finally:
                release.set()
            deadline = monotonic() + 5
            while client.get("/api/v1/metrics").json()["sessions_open"]:
                self.assertLess(monotonic(), deadline)

    def test_rejection_messages_at_capacity_keep_session_and_wire_ownership(self):
        entered, release = Event(), Event()
        inner = create_app(
            ServerConfig(
                recognizer=RecognizerConfig(model_path=self.path),
                realtime=False,
                max_pending_messages=4,
                send_timeout=5,
            )
        )

        async def application(scope, receive, send):
            async def transport(message):
                if (
                    message["type"] == "websocket.send"
                    and json.loads(message["text"]).get("request_id") == "stall"
                ):
                    entered.set()
                    await asyncio.to_thread(release.wait, 5)
                await send(message)

            await inner(scope, receive, transport)

        with TestClient(application) as app:
            wait_ready(app)
            try:
                with app.websocket_connect("/api/v1/ws") as socket:
                    ready = open_socket(socket)
                    command(socket, ready, "ping", "stall")
                    self.assertTrue(entered.wait(5))
                    for _ in range(8):
                        socket.send_text("{")
                    deadline = monotonic() + 5
                    while app.get("/api/v1/metrics").json()["counts"].get("errors", 0) == 0:
                        self.assertLess(monotonic(), deadline)
                    release.set()
                    while True:
                        message = socket.receive_json()
                        if message.get("code") == "SLOW_CONSUMER":
                            break
                    self.assertEqual(message.get("session_id"), ready["session_id"])
                    self.assertEqual(message.get("generation"), ready["generation"])
                    self.assertIsInstance(message.get("wire_sequence"), int)
            finally:
                release.set()
