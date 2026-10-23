"""从 Web 操作验证音色持久化；仅替换外部模型适配器。"""

import io
import json
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from apps.web.server.app import create_app
from speechio import Voice
from tests.web.test_synthesis import VoiceEngine, synthesis_settings


def reference_wav():
    output = io.BytesIO()
    with wave.open(output, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(24000)
        audio.writeframes(b"\x01\x00" * 2400)
    return output.getvalue()


class CloneAdapter(VoiceEngine):
    from dataclasses import replace

    capabilities = replace(VoiceEngine.capabilities, voice_cloning=True)
    clone_count = 0

    def clone_voice(self, reference_audio, *, reference_text=None):
        type(self).clone_count += 1
        return Voice(adapter="bailian", model="qwen-audio-3.1-tts-flash", data=b"test-voice")

    def synthesize(self, text, *, options):
        if not isinstance(options.voice, Voice):
            raise ValueError("没有收到可复用音色")
        return super().synthesize(text, options=options)


def upload(client):
    result = client.post(
        "/api/uploads", content=reference_wav(), headers={"X-Audio-Extension": ".wav"}
    )
    assert result.status_code == 200, result.text
    return result.json()["token"]


def synthesize(client, config):
    with client.websocket_connect("/api/synthesis") as socket:
        socket.send_json(
            {
                "type": "start",
                "text": "你好",
                "config": {"model": "voice", "mode": "full", **config},
            }
        )
        while True:
            message = socket.receive()
            if message.get("text"):
                event = json.loads(message["text"])
                if event["type"] in ("completed", "error"):
                    return event


class VoiceWebTests(unittest.TestCase):
    def test_import_download_and_remove_voice_without_model_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = synthesis_settings(directory)
            settings.voices_dir = Path(directory) / "voices"
            source = Path(directory) / "source.voice"
            Voice(adapter="bailian", model="qwen-audio-3.1-tts-flash", data=b"portable-voice").save(
                source
            )
            with TestClient(create_app(settings)) as client:
                token = client.post(
                    "/api/uploads",
                    content=source.read_bytes(),
                    headers={"X-Audio-Extension": ".voice"},
                ).json()["token"]
                result = client.post(
                    "/api/voices/import",
                    json={"model": "voice", "name": "导入音色", "file_token": token},
                )
                self.assertEqual(result.status_code, 200, result.text)
                voice_id = result.json()["voice"]["id"]
                self.assertEqual(
                    client.get(f"/api/voices/{voice_id}/download").content, source.read_bytes()
                )
                self.assertEqual(client.delete(f"/api/voices/{voice_id}").status_code, 200)
                self.assertEqual(client.get("/api/voices?model=voice").json()["voices"], [])

    def test_reference_synthesis_does_not_save_voice_and_rejects_streaming(self):
        CloneAdapter.clone_count = 0
        with tempfile.TemporaryDirectory() as directory:
            settings = synthesis_settings(directory)
            settings.voices_dir = Path(directory) / "voices"
            with patch(
                "speechio.models.factory.build_synthesis_engine",
                side_effect=lambda _: CloneAdapter(),
            ):
                with TestClient(create_app(settings)) as client:
                    token = upload(client)
                    result = synthesize(client, {"reference_audio_token": token, "mode": "stream"})
                    self.assertEqual(result["type"], "error")
                    self.assertEqual(CloneAdapter.clone_count, 0)
                    result = synthesize(client, {"reference_audio_token": token})
                    self.assertEqual(result["type"], "completed", result)
                    self.assertEqual(CloneAdapter.clone_count, 1)
                    self.assertEqual(client.get("/api/voices?model=voice").json()["voices"], [])
                    self.assertFalse((Path(directory) / f"{token}.wav").exists())

    def test_failed_clone_cleans_reference_and_releases_shared_task(self):
        class FailedAdapter(CloneAdapter):
            def clone_voice(self, reference_audio, *, reference_text=None):
                raise ValueError("模拟模型失败")

        with tempfile.TemporaryDirectory() as directory:
            settings = synthesis_settings(directory)
            settings.voices_dir = Path(directory) / "voices"
            with patch(
                "speechio.models.factory.build_synthesis_engine",
                side_effect=lambda _: FailedAdapter(),
            ):
                with TestClient(create_app(settings)) as client:
                    token = upload(client)
                    response = client.post(
                        "/api/voices/clone",
                        json={"model": "voice", "name": "失败", "reference_audio_token": token},
                    )
                    self.assertEqual(response.status_code, 400)
                    self.assertEqual(client.get("/api/voices?model=voice").json()["voices"], [])
                    self.assertFalse((Path(directory) / f"{token}.wav").exists())
                    self.assertEqual(client.post("/api/debug/release").status_code, 200)

    def test_clone_survives_restart_and_is_reused_without_cloning_again(self):
        CloneAdapter.clone_count = 0
        with tempfile.TemporaryDirectory() as directory:
            settings = synthesis_settings(directory)
            settings.voices_dir = Path(directory) / "voices"
            with patch(
                "speechio.models.factory.build_synthesis_engine",
                side_effect=lambda _: CloneAdapter(),
            ):
                with TestClient(create_app(settings)) as client:
                    result = client.post(
                        "/api/voices/clone",
                        json={
                            "model": "voice",
                            "name": "测试音色",
                            "reference_audio_token": upload(client),
                        },
                    )
                    self.assertEqual(result.status_code, 200, result.text)
                    saved = result.json()["voice"]
                with TestClient(create_app(settings)) as client:
                    self.assertEqual(
                        client.get("/api/voices?model=voice").json()["voices"][0]["id"], saved["id"]
                    )
                    result = synthesize(client, {"saved_voice_id": saved["id"]})
                    self.assertEqual(result["type"], "completed", result)
                    self.assertEqual(client.get(result["audio_url"]).content[:4], b"RIFF")
                    self.assertEqual(CloneAdapter.clone_count, 1)
                    self.assertNotIn("test-voice", str(saved))


if __name__ == "__main__":
    unittest.main()
