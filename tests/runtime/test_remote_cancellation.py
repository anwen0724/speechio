"""发送阻塞只在外部传输边界注入，运行时取消与关闭执行真实流程。"""

import json
import socket
import unittest
from dataclasses import replace
from threading import Event, Timer
from unittest.mock import patch

from speechio import (
    AudioFormat,
    RealtimeConfig,
    RemoteModelConfig,
    SessionConfig,
    STTError,
    Transcriber,
)
from speechio.stt.runtime.asr import STTRuntime
from tests.runtime.test_engine_injection import RecordingEngine
from tests.runtime.test_remote import websocket_service


class BlockedTransport:
    """模拟连接存活但不能完成发送；shutdown 会解除网络等待。"""

    def __init__(self):
        self.entered = Event()
        self.release = Event()
        self.closed = Event()
        self.socket = self
        self.started = False
        self.task_id = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.closed.set()

    def send(self, message):
        if isinstance(message, bytes):
            self.entered.set()
            self.release.wait(1)
            raise OSError("transport stopped")
        self.task_id = json.loads(message)["header"]["task_id"]

    def recv(self, timeout=None):
        if self.started:
            raise TimeoutError
        self.started = True
        return json.dumps({"header": {"task_id": self.task_id, "event": "task-started"}})

    def shutdown(self, how):
        self.release.set()

    def close(self):
        self.release.set()


def remote_config(timeout):
    return RemoteModelConfig(
        provider="bailian",
        model="qwen-audio-3.1-asr-flash-streaming",
        api_key="test-only",
        timeout_seconds=timeout,
    )


class RemoteCancellationTests(unittest.TestCase):
    def test_timeout_releases_real_socket_when_peer_stops_reading(self):
        from speechio.adapters.remote.bailian_stream import _DirectConnection

        release = Event()
        safety_expired = Event()
        peers = []

        def handler(ws):
            peers.append(ws.socket)
            ws.socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 2048)
            task_id = json.loads(ws.recv())["header"]["task_id"]
            ws.send(json.dumps({"header": {"task_id": task_id, "event": "task-started"}}))
            release.wait(5)

        class SmallSendBuffer(_DirectConnection):
            def __enter__(self):
                ws = super().__enter__()
                ws.socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 2048)
                return ws

        def safety_stop():
            safety_expired.set()
            for peer in peers:
                try:
                    peer.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass

        # 真实本机 WebSocket，缩小缓冲区并停止消费，让 sendall 确实阻塞。
        with websocket_service(
            handler, max_queue=1, ping_interval=None, close_timeout=0.2, compression=None
        ) as root:
            timer = Timer(4, safety_stop)
            timer.start()
            try:
                config = replace(remote_config(0.2), base_url=root)
                with patch(
                    "speechio.adapters.remote.bailian_stream._DirectConnection", SmallSendBuffer
                ):
                    with Transcriber(config) as transcriber:
                        with self.assertRaises(STTError) as caught:
                            transcriber.transcribe(b"\x01\x00" * 1000000, format=AudioFormat())
                self.assertEqual(caught.exception.code, "REMOTE_TIMEOUT")
                self.assertFalse(safety_expired.is_set())
            finally:
                timer.cancel()
                timer.join()
                release.set()

    def test_blocked_send_reports_configured_timeout(self):
        transport = BlockedTransport()
        with patch(
            "speechio.adapters.remote.bailian_stream._DirectConnection", return_value=transport
        ):
            with Transcriber(remote_config(0.02)) as transcriber:
                with self.assertRaises(STTError) as caught:
                    transcriber.transcribe(b"\x01\x00" * 1600, format=AudioFormat())
        self.assertEqual(caught.exception.code, "REMOTE_TIMEOUT")
        self.assertTrue(transport.closed.is_set())

    def test_cancel_interrupts_send_without_waiting_for_network_timeout(self):
        transport = BlockedTransport()
        with patch(
            "speechio.adapters.remote.bailian_stream._DirectConnection", return_value=transport
        ):
            runtime = STTRuntime.from_adapter(
                RecordingEngine(),
                realtime=RealtimeConfig(
                    mode="streaming",
                    config=remote_config(30),
                    initial_delay_seconds=0,
                    interval_seconds=0.01,
                ),
            )
            try:
                runtime.prepare()
                session = runtime.create_session(SessionConfig(realtime=True))
                session.start()
                session.feed_audio(b"\x01\x00" * 1600)
                self.assertTrue(transport.entered.wait(1))
                session.cancel_turn()
                self.assertEqual(runtime.close(timeout=0.3).status, "ok")
                self.assertTrue(transport.closed.is_set())
                self.assertEqual(runtime.capacity.realtime_units, 0)
            finally:
                transport.release.set()
                runtime.close(timeout=2)
