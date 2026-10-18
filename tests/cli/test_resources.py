"""正式资源命令不启动模型，测试从已安装同构的 CLI 入口进入。"""

import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from apps.cli.runner import main
from speechio.runtime.models.manifest import create_manifest


class ResourceCommandTests(unittest.TestCase):
    def test_explicit_download_preserves_conflicting_existing_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "model.bin").write_bytes(b"user existing file")

            def download(*, repo_id, revision, local_dir):
                Path(local_dir).mkdir(parents=True, exist_ok=True)
                for name in ("model.bin", "config.json", "tokenizer.json"):
                    (Path(local_dir) / name).write_bytes(b"remote fixed content")

            output = io.StringIO()
            with (
                patch("huggingface_hub.HfApi") as api,
                patch("huggingface_hub.snapshot_download", side_effect=download),
            ):
                api.return_value.model_info.return_value = SimpleNamespace(sha="1" * 40)
                code = main(
                    [
                        "models",
                        "download",
                        "--engine",
                        "whisper",
                        "--model-id",
                        "tiny",
                        "--destination",
                        str(root),
                    ],
                    stdout=output,
                )
            self.assertEqual(code, 1)
            self.assertEqual(json.loads(output.getvalue())["code"], "MODEL_CONFLICT")
            self.assertEqual((root / "model.bin").read_bytes(), b"user existing file")

    def test_fixed_detection_import_rejects_conflicting_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "silero.onnx"
            source.write_bytes(b"not the fixed model")
            output = io.StringIO()
            code = main(
                [
                    "models",
                    "prepare",
                    "--kind",
                    "vad",
                    "--path",
                    str(root / "model"),
                    "--local-source",
                    str(source),
                ],
                stdout=output,
            )
            self.assertEqual(code, 1)
            self.assertEqual(json.loads(output.getvalue())["code"], "MODEL_INTEGRITY")
            self.assertFalse((root / "model" / "model-manifest.json").exists())

    def test_existing_asr_directory_and_detection_resources_use_separate_checks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("model.bin", "config.json", "tokenizer.json"):
                (root / name).write_bytes(b"known cached model")
            create_manifest(
                root, backend="whisper", model_id="tiny", source="local fixture", revision="abc"
            )
            output = io.StringIO()
            self.assertEqual(
                main(
                    ["models", "prepare", "--kind", "whisper", "--path", str(root)], stdout=output
                ),
                0,
            )
            self.assertEqual(json.loads(output.getvalue())["model_path"], str(root.resolve()))
            output = io.StringIO()
            self.assertEqual(
                main(["models", "prepare", "--kind", "vad", "--path", str(root)], stdout=output),
                1,
            )
            self.assertEqual(json.loads(output.getvalue())["code"], "MODEL_INTEGRITY")
