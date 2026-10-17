"""固定权重安装器的选择、离线与完整性边界；不加载模型。"""

import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from realtime_asr.install_sherpa_models import (
    MODEL_MANIFESTS,
    ModelInstallError,
    install_models,
    manifests_for_selection,
    verify_archive,
)
from realtime_asr.model_manifests import ModelManifest
from realtime_asr.runtime.models.pinned import NEMOTRON_MODELS


class InstallSherpaModelsTests(unittest.TestCase):
    def test_selection_resolves_to_pinned_identity(self):
        selected = manifests_for_selection("all")
        self.assertEqual([name for name, _ in selected], ["560ms", "1120ms", "80ms"])
        for name, manifest in selected:
            self.assertEqual(manifest.model_id, NEMOTRON_MODELS[name]["id"])
            self.assertEqual(manifest.archive_sha256, NEMOTRON_MODELS[name]["archive"][1])

    def test_unknown_selection_reports_choices(self):
        with self.assertRaises(ModelInstallError) as caught:
            manifests_for_selection("large")
        self.assertIn("560ms", str(caught.exception))
        self.assertIn("all", str(caught.exception))

    def test_default_install_covers_all_variants(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch("realtime_asr.install_sherpa_models.prepare_nemotron") as prepare:
                prepare.side_effect = lambda destination, **kwargs: destination.mkdir(
                    parents=True, exist_ok=True
                )
                installed = install_models(folder)
        self.assertEqual(set(installed), {"560ms", "1120ms", "80ms"})
        self.assertEqual(prepare.call_count, 3)

    def test_explicit_selection_installs_only_that_variant(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch("realtime_asr.install_sherpa_models.prepare_nemotron") as prepare:
                prepare.side_effect = lambda destination, **kwargs: destination.mkdir(
                    parents=True, exist_ok=True
                )
                installed = install_models(folder, "1120ms")
        self.assertEqual(set(installed), {"1120ms"})
        self.assertEqual(prepare.call_count, 1)

    def test_offline_install_requires_existing_model_and_creates_nothing(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ModelInstallError) as caught:
                install_models(folder, "560ms", offline=True)
            self.assertIn("离线", str(caught.exception))
            destination = Path(folder) / MODEL_MANIFESTS["560ms"].model_id
            self.assertFalse(destination.exists())

    def test_symlinked_root_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "real"
            target.mkdir()
            link = Path(folder) / "link"
            try:
                link.symlink_to(target, target_is_directory=True)
            except OSError:
                self.skipTest("当前环境不能创建符号链接")
            with self.assertRaises(ModelInstallError):
                install_models(link, "560ms", offline=True)

    def test_archive_verification_uses_size_and_digest(self):
        content = b"model-archive"
        manifest = ModelManifest(
            model_id="test",
            archive_url="",
            archive_filename="test.tar.bz2",
            archive_size_bytes=len(content),
            archive_sha256=hashlib.sha256(content).hexdigest(),
            expected_files=(),
            license_name="",
            license_url="",
        )
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / manifest.archive_filename
            path.write_bytes(content)
            verify_archive(path, manifest)
            path.write_bytes(content + b"x")
            with self.assertRaises(ModelInstallError):
                verify_archive(path, manifest)
            path.write_bytes(b"changed-content")
            with self.assertRaises(ModelInstallError):
                verify_archive(path, manifest)
