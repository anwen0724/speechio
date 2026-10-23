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
    def test_key_priority_and_blank_values(self):
        """文件直填优先；空值继续查进程环境和应用固定位置的 .env。"""
        cases = (
            ("toml-secret", "process-secret", "toml-secret"),
            ("", "process-secret", "process-secret"),
            ("   ", "   ", "dotenv-secret"),
            (None, None, "dotenv-secret"),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            env_path = root / ".env"
            env_path.write_text('SPEECHIO_TEST_KEY="dotenv-secret"\n', encoding="utf-8")
            config_path = root / "other" / "config.toml"
            config_path.parent.mkdir()
            for direct, process, expected in cases:
                with self.subTest(direct=direct, process=process):
                    config_path.write_text(
                        '[models.cloud]\nsource="remote"\nprovider="bailian"\n'
                        'model="qwen-audio-3.1-asr-flash"\napi_key_env="SPEECHIO_TEST_KEY"\n'
                        + (f'apikey="{direct}"\n' if direct is not None else ""),
                        encoding="utf-8",
                    )
                    environment = {} if process is None else {"SPEECHIO_TEST_KEY": process}
                    with (
                        patch.dict("os.environ", environment, clear=True),
                        patch("apps.web.server.settings.DEFAULT_ENV_PATH", env_path, create=True),
                    ):
                        settings = Settings.load(config_path)
                    self.assertEqual(settings.models["cloud"].api_key, expected)
                    with TestClient(create_app(settings)) as client:
                        self.assertNotIn(expected, client.get("/api/config").text)
            env_path.unlink()
            with (
                patch.dict("os.environ", {}, clear=True),
                patch("apps.web.server.settings.DEFAULT_ENV_PATH", env_path, create=True),
            ):
                self.assertIsNone(Settings.load(config_path).models["cloud"].api_key)

    def test_remote_configuration_reads_key_without_exposing_it_to_browser(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.toml"
            path.write_text(
                '[models.cloud]\nsource="remote"\nprovider="bailian"\n'
                'model="qwen-audio-3.1-asr-flash"\napikey="test-only-secret"\n',
                encoding="utf-8",
            )
            settings = Settings.load(path)
            self.assertEqual(settings.models["cloud"].api_key, "test-only-secret")
            with TestClient(create_app(settings)) as client:
                response = client.get("/api/config")
                model = response.json()["models"][0]
                self.assertEqual(model["source"], "remote")
                self.assertEqual(model["provider"], "bailian")
                self.assertEqual(model["timeout_seconds"], 30)
                self.assertTrue(model["api_key_configured"])
                self.assertEqual(model["devices"], [])
                self.assertNotIn("test-only-secret", response.text)

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
            self.assertEqual(config["upload_limit_bytes"], 1024**3)
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
