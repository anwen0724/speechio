"""验证 Web 的本地音色发现与单次设备选择，不执行模型推理。"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from apps.web.server.app import create_app
from apps.web.server.settings import Settings
from tests.web.test_synthesis import VoiceEngine


def local_settings(directory):
    """从真实 TOML 入口构造本机模型配置，音色文件不需要用于推理。"""
    root = Path(directory)
    (root / "voices").mkdir()
    for name in ("zf_001.pt", "af_maple.pt", "notes.txt"):
        (root / "voices" / name).write_bytes(b"fixture")
    config = root / "config.toml"
    config.write_text(
        f"""[models.kokoro]
task = "synthesis"
adapter = "kokoro"
model_path = '{root}'
device = "cpu"
compute_type = "float32"
[models.kokoro.synthesis]
voice = "zf_001"
""",
        encoding="utf-8",
    )
    settings = Settings.load(config, debug=True)
    settings.upload_dir = root / "uploads"
    return settings


class KokoroWebTests(unittest.TestCase):
    def test_device_override_reaches_synthesis_without_changing_defaults(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = local_settings(directory)

            # 仅替换实际模型运行库边界，其余请求解析、配置合并和任务流程保留。
            def engine(config):
                self.assertIn(config.device, ("cpu", "gpu"))
                return VoiceEngine()

            with patch("speechio.models.factory.build_synthesis_engine", side_effect=engine):
                with TestClient(create_app(settings)) as client:
                    for override, expected in (({"device": "gpu"}, "gpu"), ({}, "cpu")):
                        with client.websocket_connect("/api/synthesis") as socket:
                            socket.send_json(
                                {
                                    "type": "start",
                                    "text": "测试设备选择",
                                    "config": {"model": "kokoro", "mode": "full", **override},
                                }
                            )
                            ready = socket.receive_json()
                            self.assertEqual(ready["type"], "ready", ready)
                            self.assertEqual(ready["effective_config"]["model"]["device"], expected)
                            self.assertEqual(socket.receive_json()["type"], "completed")
                    model = client.get("/api/config").json()["synthesis_models"][0]
                    self.assertEqual(model["device"], "cpu")

    def test_config_lists_actual_voice_files_and_local_devices(self):
        with tempfile.TemporaryDirectory() as directory:
            with TestClient(create_app(local_settings(directory))) as client:
                model = client.get("/api/config").json()["synthesis_models"][0]
                self.assertEqual(model["voices"], ["af_maple", "zf_001"])
                self.assertEqual(model["devices"], ["cpu", "gpu"])
                self.assertEqual(model["device"], "cpu")
                self.assertFalse(model["capabilities"]["voice_cloning"])
                (Path(directory) / "voices" / "zm_010.pt").write_bytes(b"fixture")
                updated = client.get("/api/config").json()["synthesis_models"][0]
                self.assertEqual(updated["voices"], ["af_maple", "zf_001", "zm_010"])
