"""模型校验应拒绝篡改，不通过重新生成清单掩盖变化。"""

import json
import tempfile
import unittest
from pathlib import Path

from realtime_asr.core.errors import ASRError
from realtime_asr.core.runtime.models.manifest import create_manifest, verify_manifest


class ModelTests(unittest.TestCase):
    def test_manifest_cannot_omit_required_files(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            for name in ["model.bin", "config.json", "tokenizer.json"]:
                (path / name).write_bytes(b"original")
            manifest = create_manifest(
                path, backend="whisper", model_id="tiny", source="test", revision="abc"
            )
            manifest["files"] = []
            (path / "model-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaises(ASRError) as caught:
                verify_manifest(path)
            self.assertEqual(caught.exception.code, "MODEL_INCOMPLETE")

    def test_reuse_checks_original_manifest_and_reports_changed_file(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            for name in ["model.bin", "config.json", "tokenizer.json"]:
                (path / name).write_bytes(b"original")
            first = create_manifest(
                path, backend="whisper", model_id="tiny", source="test", revision="abc"
            )
            self.assertEqual(verify_manifest(path)["revision"], "abc")
            (path / "model.bin").write_bytes(b"tampered")
            with self.assertRaises(ASRError) as caught:
                verify_manifest(path)
            self.assertEqual(caught.exception.code, "MODEL_INTEGRITY")
            with self.assertRaises(ASRError):
                create_manifest(
                    path, backend="whisper", model_id="tiny", source="test", revision="abc"
                )
            saved = json.loads((path / "model-manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(saved, first)

    def test_missing_required_model_file_is_not_prepared(self):
        with tempfile.TemporaryDirectory() as folder:
            Path(folder, "model.bin").write_bytes(b"partial")
            with self.assertRaises(ASRError) as caught:
                create_manifest(
                    folder, backend="whisper", model_id="tiny", source="test", revision="abc"
                )
            self.assertEqual(caught.exception.code, "MODEL_INCOMPLETE")
