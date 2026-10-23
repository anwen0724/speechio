"""模型下载通过外部网络替身验证实际文件交付，不加载识别器。"""

import io
import tarfile
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from speechio import STTError
from speechio.models import download_model
from speechio.models.pinned import nemotron_model


class DownloadModelTests(unittest.TestCase):
    def test_download_records_identity_without_hashes(self):
        def download(*, repo_id, revision, local_dir, **kwargs):
            Path(local_dir).mkdir(parents=True, exist_ok=True)
            for name in ("model.bin", "config.json", "tokenizer.json"):
                (Path(local_dir) / name).write_bytes(b"model data")

        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "large"
            with (
                patch("huggingface_hub.HfApi") as api,
                patch("huggingface_hub.snapshot_download", side_effect=download),
            ):
                api.return_value.model_info.return_value = SimpleNamespace(sha="1" * 40)
                result = download_model(engine="whisper", model_id="large-v3", destination=target)
            self.assertEqual(result, target.resolve())
            self.assertEqual((result / "model.bin").read_bytes(), b"model data")
            record = (result / "model-manifest.json").read_text(encoding="utf-8")
            self.assertNotIn("sha256", record)
            with patch("huggingface_hub.HfApi", side_effect=AssertionError("不应重复下载")):
                self.assertEqual(
                    download_model(engine="whisper", model_id="large-v3", destination=target),
                    result,
                )

    def test_conflicting_destination_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            existing = target / "model.bin"
            existing.write_bytes(b"my model")
            with self.assertRaises(STTError) as caught:
                download_model(engine="whisper", model_id="large-v3-turbo", destination=target)
            self.assertEqual(caught.exception.code, "MODEL_CONFLICT")
            self.assertEqual(existing.read_bytes(), b"my model")

    def test_failed_download_does_not_publish_partial_model(self):
        def interrupted(**kwargs):
            (Path(kwargs["local_dir"]) / "model.bin").write_bytes(b"partial")
            raise OSError("网络中断")

        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "model"
            with (
                patch("huggingface_hub.HfApi") as api,
                patch("huggingface_hub.snapshot_download", side_effect=interrupted),
            ):
                api.return_value.model_info.return_value = SimpleNamespace(sha="1" * 40)
                with self.assertRaises(STTError) as caught:
                    download_model(engine="whisper", model_id="large-v3-turbo", destination=target)
            self.assertEqual(caught.exception.code, "MODEL_DOWNLOAD")
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_nemotron_archive_is_extracted_without_fixed_hash_requirement(self):
        identity = nemotron_model("560ms")["id"]
        payload = io.BytesIO()
        with tarfile.open(fileobj=payload, mode="w:bz2") as archive:
            for name in (
                "encoder.int8.onnx",
                "decoder.int8.onnx",
                "joiner.int8.onnx",
                "tokens.txt",
            ):
                info = tarfile.TarInfo(f"{identity}/{name}")
                info.size = 5
                archive.addfile(info, io.BytesIO(b"model"))
        payload.seek(0)
        payload.headers = {"Content-Length": str(len(payload.getvalue()))}
        with tempfile.TemporaryDirectory() as directory:
            with patch("speechio.models.prepare.urlopen", return_value=payload):
                result = download_model(
                    engine="nemotron", model_id="560ms", destination=Path(directory) / "model"
                )
            self.assertEqual((result / "encoder.int8.onnx").read_bytes(), b"model")
            self.assertNotIn("sha256", (result / "model-manifest.json").read_text(encoding="utf-8"))

    def test_qwen_download_delivers_local_model_and_is_reusable(self):
        from tests.tts.test_local import qwen_folder

        def download(*, local_dir, **kwargs):
            qwen_folder(Path(local_dir))

        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "qwen"
            with (
                patch("huggingface_hub.HfApi") as api,
                patch("huggingface_hub.snapshot_download", side_effect=download),
            ):
                api.return_value.model_info.return_value = SimpleNamespace(sha="2" * 40)
                result = download_model(
                    engine="qwen_tts", model_id="Qwen/Qwen3-TTS-12Hz-0.6B-Base", destination=target
                )
            self.assertTrue((result / "speech_tokenizer" / "model.safetensors").is_file())
            with patch("huggingface_hub.HfApi", side_effect=AssertionError("no network")):
                self.assertEqual(
                    download_model(
                        engine="qwen_tts",
                        model_id="Qwen/Qwen3-TTS-12Hz-0.6B-Base",
                        destination=target,
                    ),
                    result,
                )
