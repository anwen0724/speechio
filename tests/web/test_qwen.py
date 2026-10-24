"""验证本地 Qwen 的设备选择与按模型管理音色，不执行真实模型推理。"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from apps.web.server.app import create_app
from apps.web.server.settings import Settings
from speechio import Voice
from speechio.adapters.local.qwen_tts import QwenTTSAdapter
from tests.web.test_synthesis import VoiceEngine
from tests.web.test_voices import synthesize, upload


def qwen_settings(directory):
    """通过应用配置入口声明两个本地模型，默认使用 GPU。"""
    root = Path(directory)
    config = root / "config.toml"
    config.write_text(
        "\n".join(
            f'[models.{key}]\ntask="synthesis"\nadapter="qwen_tts"\n'
            f"model_path='{root / size}'\ndevice=\"gpu\"\n"
            for key, size in (("voice", "0.6b"), ("large", "1.7b"))
        ),
        encoding="utf-8",
    )
    settings = Settings.load(config, debug=True)
    settings.upload_dir = root / "uploads"
    settings.voices_dir = root / "voices"
    return settings


class LocalCloneEngine(VoiceEngine):
    """外部推理替身，把实际收到的设备写入声音结果供公开接口验证。"""

    capabilities = QwenTTSAdapter.capabilities

    def __init__(self, config):
        super().__init__()
        self.config = config

    def clone_voice(self, reference_audio, *, reference_text=None):
        return Voice(
            adapter="qwen_tts",
            model=Path(self.config.model_path).name,
            data=self.config.device.encode(),
        )


class QwenWebTests(unittest.TestCase):
    def test_clone_uses_request_device_or_config_default(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = qwen_settings(directory)
            with patch("speechio.models.factory.build_synthesis_engine", LocalCloneEngine):
                with TestClient(create_app(settings)) as client:
                    for override, expected in (({"device": "cpu"}, "cpu"), ({}, "gpu")):
                        response = client.post(
                            "/api/voices/clone",
                            json={
                                "model": "voice",
                                "name": "示例音色",
                                "reference_audio_token": upload(client),
                                **override,
                            },
                        )
                        self.assertEqual(response.status_code, 200, response.text)
                        result = response.json()
                        self.assertEqual(result["effective_config"]["device"], expected)
                        voice = Voice.load(result["voice"]["file_path"])
                        self.assertEqual(voice.data, expected.encode())
                    model = client.get("/api/config").json()["synthesis_models"][0]
                    self.assertEqual(model["device"], "gpu")

    def test_saved_voice_can_switch_device_but_remains_with_its_model(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = qwen_settings(directory)
            with patch("speechio.models.factory.build_synthesis_engine", LocalCloneEngine):
                with TestClient(create_app(settings)) as client:
                    response = client.post(
                        "/api/voices/clone",
                        json={
                            "model": "voice",
                            "name": "小模型音色",
                            "reference_audio_token": upload(client),
                        },
                    )
                    self.assertEqual(response.status_code, 200, response.text)
                    voice_id = response.json()["voice"]["id"]
                    self.assertEqual(len(client.get("/api/voices?model=voice").json()["voices"]), 1)
                    self.assertEqual(client.get("/api/voices?model=large").json()["voices"], [])
                    result = synthesize(client, {"saved_voice_id": voice_id, "device": "cpu"})
                    self.assertEqual(result["type"], "completed", result)
