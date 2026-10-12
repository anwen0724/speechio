"""网络只映射 P7 选项，不重复核心稳定与切段规则矩阵。"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

from realtime_asr import RecognizerConfig
from realtime_asr.server import ServerConfig, create_app
from tests.runtime import test_recognizer
from tests.server.test_service import wait_ready


class TextOptionsTests(unittest.TestCase):
    def setUp(self):
        test_recognizer.RecognizerTests.setUp(self)

    def test_http_word_request_and_ws_stable_selection_are_reported(self):
        word = SimpleNamespace(word="hello", start=0.1, end=0.2, probability=1)

        def model(audio, **kwargs):
            return iter(
                [SimpleNamespace(text="hello", start=0, end=0.2, words=[word])]
            ), SimpleNamespace(language="en", language_probability=1)

        with patch("faster_whisper.BatchedInferencePipeline.transcribe", side_effect=model):
            with TestClient(
                create_app(ServerConfig(recognizer=RecognizerConfig(model_path=self.path)))
            ) as client:
                wait_ready(client)
                response = client.post(
                    "/api/v1/transcribe",
                    params={"word_timestamps": "true"},
                    content=bytes(32000),
                    headers={"Content-Type": "application/octet-stream"},
                )
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json()["result"]["timestamp_status"], "reported")
                self.assertEqual(response.json()["result"]["words"][0]["end_sample"], 3200)
                with client.websocket_connect("/api/v1/ws") as socket:
                    socket.send_json(
                        {
                            "protocol_version": 1,
                            "type": "open",
                            "request_id": "open",
                            "stable_text": True,
                        }
                    )
                    ready = socket.receive_json()
                    self.assertEqual(ready["type"], "ready")
                    self.assertTrue(ready["text_options"]["stable_text"])
