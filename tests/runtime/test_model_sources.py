"""本地目录加载的公开契约；只替换第三方模型运行库边界。"""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from speechio import LocalModelConfig, STTError, Transcriber
from speechio.models.files import MODEL_FILES


class ModelSourceTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.path = Path(self.folder.name)
        for name in MODEL_FILES["whisper"]:
            (self.path / name).write_bytes(b"test-model")
        self.model = SimpleNamespace(feature_extractor=SimpleNamespace(chunk_length=30))

    def test_whisper_directory_without_manifest_loads_offline(self):
        with patch("faster_whisper.WhisperModel", return_value=self.model) as loader:
            with Transcriber(
                LocalModelConfig(adapter="whisper", model_path=self.path)
            ) as recognizer:
                self.assertEqual(recognizer.state, "ready")
        self.assertEqual(Path(loader.call_args.args[0]), self.path.resolve())
        self.assertTrue(loader.call_args.kwargs["local_files_only"])
        self.assertFalse((self.path / "model-manifest.json").exists())

    def test_existing_manifest_is_not_a_loading_gate_or_rewritten(self):
        record = self.path / "model-manifest.json"
        record.write_text('{"files": [{"sha256": "old digest"}]}', encoding="utf-8")
        before = record.read_bytes()
        with patch("faster_whisper.WhisperModel", return_value=self.model):
            with Transcriber(
                LocalModelConfig(adapter="whisper", model_path=self.path)
            ) as recognizer:
                self.assertEqual(recognizer.state, "ready")
        self.assertEqual(record.read_bytes(), before)

    def test_missing_directory_and_required_file_fail_before_native_loading(self):
        (self.path / "model.bin").unlink()
        for path in (self.path, self.path / "missing"):
            with self.subTest(path=path):
                with patch(
                    "faster_whisper.WhisperModel", side_effect=AssertionError("不应尝试加载")
                ):
                    with self.assertRaises(STTError) as caught:
                        with Transcriber(LocalModelConfig(adapter="whisper", model_path=path)):
                            pass
                self.assertEqual(caught.exception.code, "MODEL_INCOMPLETE")

    def test_gpu_and_compute_type_reach_whisper_runtime(self):
        with patch("faster_whisper.WhisperModel", return_value=self.model) as loader:
            with Transcriber(
                LocalModelConfig(
                    adapter="whisper",
                    model_path=self.path,
                    device="gpu",
                    device_index=1,
                    compute_type="float16",
                )
            ):
                pass
        self.assertEqual(loader.call_args.kwargs["device"], "cuda")
        self.assertEqual(loader.call_args.kwargs["device_index"], 1)
        self.assertEqual(loader.call_args.kwargs["compute_type"], "float16")

    def test_nemotron_loads_without_manifest_and_rejects_unsupported_gpu(self):
        for name in MODEL_FILES["nemotron"]:
            (self.path / name).write_bytes(b"test-model")
        with patch("sherpa_onnx.OnlineRecognizer.from_transducer") as loader:
            with Transcriber(
                LocalModelConfig(adapter="nemotron", model_path=self.path)
            ) as recognizer:
                self.assertEqual(recognizer.state, "ready")
            self.assertEqual(loader.call_args.kwargs["provider"], "cpu")
            with self.assertRaises(STTError) as caught:
                with Transcriber(
                    LocalModelConfig(adapter="nemotron", model_path=self.path, device="gpu")
                ):
                    pass
            self.assertEqual(caught.exception.code, "DEVICE_UNAVAILABLE")

    def test_engine_options_cannot_override_device_selection(self):
        with self.assertRaises(STTError) as caught:
            Transcriber(
                LocalModelConfig(
                    adapter="nemotron", model_path=self.path, adapter_options={"provider": "cuda"}
                )
            )
        self.assertEqual(caught.exception.code, "INVALID_CONFIGURATION")

    def test_whisper_engine_options_override_beam_size(self):
        info = SimpleNamespace(language=None, language_probability=None)
        with (
            patch("faster_whisper.WhisperModel", return_value=self.model),
            patch(
                "faster_whisper.BatchedInferencePipeline.transcribe", return_value=(iter([]), info)
            ) as run,
        ):
            with Transcriber(
                LocalModelConfig(
                    adapter="whisper", model_path=self.path, adapter_options={"beam_size": 2}
                )
            ) as recognizer:
                from speechio import AudioFormat

                recognizer.transcribe(b"\x01\x00" * 100, format=AudioFormat())
        self.assertEqual(run.call_args.kwargs["beam_size"], 2)

    def test_invalid_engine_and_options_are_rejected(self):
        for options in (
            {"adapter": "other"},
            {"adapter_options": ["beam_size"]},
            {"adapter_options": {"beam_size": 0}},
        ):
            with self.subTest(options=options), self.assertRaises(STTError):
                with Transcriber(
                    LocalModelConfig(**{"adapter": "whisper", "model_path": self.path, **options})
                ):
                    pass
