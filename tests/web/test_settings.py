"""验证 Web 配置选择与启动错误，不加载模型或绑定服务端口。"""

import io
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from apps.web.__main__ import main
from apps.web.server.app import create_app
from apps.web.server.settings import Settings


class WebSettingsTests(unittest.TestCase):
    def test_default_configuration_supplies_page_models(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.toml"
            path.write_text(
                '[models.native]\nengine = "nemotron"\nmodel_path = "models/native"\n',
                encoding="utf-8",
            )
            with (
                patch("apps.web.server.settings.DEFAULT_CONFIG_PATH", path, create=True),
                TestClient(create_app()) as client,
            ):
                config = client.get("/api/config").json()
            self.assertEqual([model["id"] for model in config["models"]], ["native"])
            self.assertEqual(config["models"][0]["engine"], "nemotron")
            self.assertNotIn("model_path", config["models"][0])

    def test_invalid_configuration_stops_startup_with_file_context(self):
        cases = {
            "missing": None,
            "syntax": "[models",
            "empty": "[models]\n",
            "fields": '[models.test]\nengine = "whisper"\n',
        }
        with tempfile.TemporaryDirectory() as directory:
            for name, content in cases.items():
                with self.subTest(name=name):
                    path = Path(directory) / f"{name}.toml"
                    if content is not None:
                        path.write_text(content, encoding="utf-8")
                    output = io.StringIO()
                    with patch("sys.argv", ["speechio-web", "--config", str(path)]):
                        with redirect_stderr(output), self.assertRaises(SystemExit) as caught:
                            main()
                    self.assertEqual(caught.exception.code, 2)
                    self.assertIn(str(path), output.getvalue())
                    if name == "missing":
                        self.assertIn("config.example.toml", output.getvalue())

    def test_explicit_configuration_overrides_default(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "custom.toml"
            path.write_text(
                '[models.custom]\nengine = "whisper"\nmodel_path = "models/custom"\n',
                encoding="utf-8",
            )
            with patch("apps.web.server.settings.DEFAULT_CONFIG_PATH", path.with_name("absent")):
                config = Settings.load(path, debug=True)
            self.assertEqual(list(config.models), ["custom"])
            self.assertTrue(config.debug)
            self.assertEqual(config.models["custom"].model_path, "models/custom")
