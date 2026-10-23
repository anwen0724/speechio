"""仅替代云端 HTTP 和 WebSocket 服务，验证公开克隆与合成流程。"""

import base64
import io
import json
import tempfile
import unittest
import wave
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import httpx

from speechio import (
    AudioFormat,
    AudioResult,
    RemoteModelConfig,
    SynthesisOptions,
    Synthesizer,
    TTSError,
    Voice,
    clone_voice,
    synthesize,
)
from tests.runtime.test_remote import websocket_service

MODEL = "qwen-audio-3.1-tts-flash"


@contextmanager
def enrollment_service(handler):
    """使用真实 HTTP 客户端，只替代服务响应；不读取用户密钥。"""
    client_type = httpx.Client
    with patch(
        "httpx.Client",
        side_effect=lambda **kwargs: client_type(transport=httpx.MockTransport(handler), **kwargs),
    ):
        yield


class BailianVoiceTests(unittest.TestCase):
    def test_required_voice_and_mismatched_saved_voice_fail_before_network(self):
        config = RemoteModelConfig(provider="bailian", model=MODEL, api_key="test-key")
        with Synthesizer(config) as tts, patch("websockets.sync.client.reconnect") as connection:
            for voice, code in (
                (None, "INVALID_INPUT"),
                (Voice(adapter="qwen_tts", model="local", data=b"{}"), "VOICE_INCOMPATIBLE"),
            ):
                with self.subTest(voice=voice), self.assertRaises(TTSError) as caught:
                    tts.synthesize("你好", options=SynthesisOptions(voice=voice))
                self.assertEqual(caught.exception.code, code)
            connection.assert_not_called()

    def test_reference_audio_synthesis_creates_one_voice_and_one_audio_task(self):
        actions = []
        voices = []

        def enrollment(request):
            payload = json.loads(request.content)["input"]
            actions.append(payload["action"])
            output = (
                {"voice_id": "one-voice"}
                if payload["action"] == "create_voice"
                else {"status": "OK"}
            )
            return httpx.Response(200, json={"output": output})

        def synthesis(ws):
            run = json.loads(ws.recv())
            voices.append(run["payload"]["parameters"]["voice"])
            task_id = run["header"]["task_id"]
            ws.send(json.dumps({"header": {"task_id": task_id, "event": "task-started"}}))
            ws.recv()
            ws.recv()
            ws.send(b"\x03\x00" * 24)
            ws.send(json.dumps({"header": {"task_id": task_id, "event": "task-finished"}}))

        with websocket_service(synthesis) as url, enrollment_service(enrollment):
            audio = synthesize(
                "组合调用",
                config=RemoteModelConfig(
                    provider="bailian",
                    model=MODEL,
                    api_key="test-key",
                    base_url=url,
                ),
                reference_audio=AudioResult(b"\x01\x00" * 120000, AudioFormat(sample_rate=24000)),
            )
        self.assertEqual(audio.data, b"\x03\x00" * 24)
        self.assertEqual(actions, ["create_voice", "query_voice"])
        self.assertEqual(voices, ["one-voice"])

    def test_failed_or_timed_out_enrollment_never_starts_synthesis(self):
        for outcome, code in (("failed", "CLONING_FAILED"), ("timeout", "REMOTE_TIMEOUT")):
            with self.subTest(outcome=outcome):
                actions = []
                now = [0.0]

                def enrollment(request):
                    action = json.loads(request.content)["input"]["action"]
                    actions.append(action)
                    if action == "create_voice":
                        output = {"voice_id": "retained-voice"}
                    else:
                        output = {"status": "UNDEPLOYED" if outcome == "failed" else "DEPLOYING"}
                        now[0] = 2.0
                    return httpx.Response(200, json={"output": output})

                with (
                    enrollment_service(enrollment),
                    patch(
                        "speechio.adapters.remote.bailian_tts.monotonic", side_effect=lambda: now[0]
                    ),
                    patch("speechio.adapters.remote.bailian_tts.sleep"),
                    patch("websockets.sync.client.reconnect") as connection,
                ):
                    with self.assertRaises(TTSError) as caught:
                        synthesize(
                            "不能合成",
                            config=RemoteModelConfig(
                                provider="bailian",
                                model=MODEL,
                                api_key="test-key",
                                timeout_seconds=1,
                            ),
                            reference_audio=AudioResult(
                                b"\x01\x00" * 120000, AudioFormat(sample_rate=24000)
                            ),
                        )
                    self.assertEqual(caught.exception.code, code)
                    self.assertIn("retained-voice", " ".join(caught.exception.__notes__))
                    connection.assert_not_called()
                self.assertEqual(actions, ["create_voice", "query_voice"])

    def test_clone_is_saved_and_reused_by_new_instance_without_enrollment(self):
        requests = []
        voices = []
        reference = AudioResult(b"\x01\x00" * 120000, AudioFormat(sample_rate=24000))

        def enrollment(request):
            body = json.loads(request.content)
            requests.append(body)
            self.assertEqual(request.url.path, "/api/v1/services/audio/tts/customization")
            self.assertEqual(request.headers["Authorization"], "Bearer test-key")
            self.assertEqual(body["model"], "voice-enrollment")
            if body["input"]["action"] == "create_voice":
                self.assertEqual(body["input"]["target_model"], MODEL)
                encoded = body["input"]["audio"].split(",", 1)[1]
                with wave.open(io.BytesIO(base64.b64decode(encoded)), "rb") as audio:
                    self.assertEqual(audio.getframerate(), 24000)
                    self.assertEqual(audio.readframes(audio.getnframes()), reference.data)
                return httpx.Response(200, json={"output": {"voice_id": "test-cloned-voice"}})
            self.assertEqual(
                body["input"], {"action": "query_voice", "voice_id": "test-cloned-voice"}
            )
            return httpx.Response(200, json={"output": {"status": "OK", "target_model": MODEL}})

        def synthesis(ws):
            run = json.loads(ws.recv())
            voices.append(run["payload"]["parameters"]["voice"])
            task_id = run["header"]["task_id"]
            ws.send(json.dumps({"header": {"task_id": task_id, "event": "task-started"}}))
            ws.recv()
            ws.recv()
            ws.send(b"\x02\x00" * 24)
            ws.send(json.dumps({"header": {"task_id": task_id, "event": "task-finished"}}))

        with websocket_service(synthesis) as url, tempfile.TemporaryDirectory() as folder:
            config = RemoteModelConfig(
                provider="bailian", model=MODEL, api_key="test-key", base_url=url
            )
            with enrollment_service(enrollment):
                voice = clone_voice(reference, config=config)
            self.assertEqual(voices, [])
            path = voice.save(Path(folder) / "speaker.voice")
            self.assertNotIn(b"test-key", path.read_bytes())
            restored = Voice.load(path)
            with Synthesizer(config) as tts:
                audio = tts.synthesize("你好", options=SynthesisOptions(voice=restored))
        self.assertEqual(audio.data, b"\x02\x00" * 24)
        self.assertEqual(voices, ["test-cloned-voice"])
        self.assertEqual(
            [item["input"]["action"] for item in requests], ["create_voice", "query_voice"]
        )
