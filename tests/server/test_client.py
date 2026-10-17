"""通过真实 ASGI 应用验证客户端，不替代核心会话。"""

import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

from realtime_asr import AudioFormat, RecognizerConfig
from realtime_asr.client import RemoteClient
from realtime_asr.server import ServerConfig, create_app
from tests.runtime import test_recognizer
from tests.server.test_service import wait_ready
from tests.session import test_automatic


class SocketTransport:
    """仅适配外部 websocket-client 的传输边界。"""

    def __init__(self, socket):
        self.socket = socket

    def send(self, message):
        self.socket.send_text(message)

    def send_binary(self, message):
        self.socket.send_bytes(message)

    def recv(self):
        return json.dumps(self.socket.receive_json())

    def close(self):
        self.socket.close()

    def abort(self):
        self.socket.close()

    def settimeout(self, timeout):
        pass


class ClientTests(unittest.TestCase):
    def setUp(self):
        test_recognizer.RecognizerTests.setUp(self)
        inference = patch(
            "faster_whisper.BatchedInferencePipeline.transcribe",
            return_value=(iter([]), SimpleNamespace(language=None, language_probability=None)),
        )
        inference.start()
        self.addCleanup(inference.stop)

    def test_remote_file_decodes_source_frames_before_http_transmission(self):
        path = self.path / "source.wav"
        sf.write(path, np.ones(515, dtype=np.int16), 16000, subtype="PCM_16")
        config = ServerConfig(recognizer=RecognizerConfig(model=self.path), realtime=False)
        with TestClient(create_app(config)) as app:
            wait_ready(app)

            def transport(request):
                response = app.post(
                    request.url.path,
                    params=request.url.params,
                    content=request.content,
                    headers=request.headers,
                )
                return httpx.Response(response.status_code, json=response.json())

            factory = httpx.Client
            with patch(
                "httpx.Client",
                side_effect=lambda **options: factory(
                    **options, transport=httpx.MockTransport(transport)
                ),
            ):
                with RemoteClient("http://testserver") as client:
                    result = client.transcribe_file(path)
            self.assertEqual(result["event"]["audio_samples"], 515)

    def test_remote_reset_feed_tail_and_graceful_close_preserve_messages(self):
        config = ServerConfig(recognizer=RecognizerConfig(model=self.path), realtime=False)
        with TestClient(create_app(config)) as app:
            wait_ready(app)
            with app.websocket_connect("/api/v1/ws") as socket:
                with patch("websocket.create_connection", return_value=SocketTransport(socket)):
                    with RemoteClient("http://testserver") as client:
                        remote = client.connect(format=AudioFormat(), realtime=False)
                        remote.start()
                        self.assertEqual(remote.reset()["generation"], 1)
                        unit = remote.start()["unit"]
                        # 源帧可跨 feed 分块，结束时仍递交短尾。
                        remote.feed_audio(b"\x01")
                        remote.feed_audio(b"\x00" + b"\x01\x00" * 514)
                        remote.finish()
                        remote.close()
                        messages = []
                        while message := remote.get_message(timeout=0):
                            messages.append(message)
                        final = next(
                            m["event"]
                            for m in messages
                            if m.get("event", {}).get("kind") == "final"
                        )
                        self.assertEqual(final["unit"], unit)
                        self.assertEqual(final["audio_samples"], 515)
                        self.assertEqual(messages[-1]["type"], "closed")


class AutomaticClientTests(unittest.TestCase):
    def setUp(self):
        test_automatic.AutomaticTests.setUp(self)

    def test_cancel_auto_segment_preserves_continuous_source_frame_remainder(self):
        config = ServerConfig(
            recognizer=RecognizerConfig(model=self.path),
            vad_model=self.vad.model_path,
            realtime=False,
        )
        with TestClient(create_app(config)) as app:
            wait_ready(app)
            with app.websocket_connect("/api/v1/ws") as socket:
                with patch("websocket.create_connection", return_value=SocketTransport(socket)):
                    with RemoteClient("http://testserver") as client:
                        with client.connect(mode="auto", realtime=False) as remote:
                            remote.feed_audio(b"\xf4\x01" * 1000)
                            remote.feed_audio(b"\xb0")
                            self.assertTrue(remote.cancel()["cancelled"])
                            remote.feed_audio(b"\x04" + b"\x58\x02" * 515)
                            remote.end_input()
                            remote.close()
                            messages = []
                            while message := remote.get_message(timeout=0):
                                messages.append(message)
                            final = next(
                                m["event"]
                                for m in messages
                                if m.get("event", {}).get("kind") == "final"
                            )
                            self.assertEqual(final["boundary"]["end_sample"], 1516)
