"""SDK 模型来源的公开行为：名字交给引擎，目录由本项目校验。"""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from realtime_asr import ASRError, Recognizer, RecognizerConfig
from realtime_asr.engines.whisper import WhisperEngine
from realtime_asr.runtime.models.manifest import create_manifest


class ModelSourceTests(unittest.TestCase):
    def test_default_recognizer_uses_whisper_tiny(self):
        recognizer = Recognizer()
        self.assertEqual(recognizer.config.model, "tiny")
        self.assertEqual(recognizer.config.engine, "whisper")
        self.assertEqual(recognizer.capabilities.engine, "whisper")
        recognizer.close()

    def test_whisper_model_name_is_passed_to_upstream_with_download_options(self):
        model = SimpleNamespace(feature_extractor=SimpleNamespace(chunk_length=30))
        with tempfile.TemporaryDirectory() as folder:
            with patch("faster_whisper.WhisperModel", return_value=model) as loader:
                engine = WhisperEngine(
                    RecognizerConfig(
                        model="small.en",
                        compute_type="int8",
                        download_root=folder,
                        local_files_only=True,
                    )
                )
                engine.close()
        call = loader.call_args
        self.assertEqual(call.args[0], "small.en")
        self.assertEqual(call.kwargs["download_root"], folder)
        self.assertTrue(call.kwargs["local_files_only"])

    def test_whisper_directory_without_manifest_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ASRError) as caught:
                WhisperEngine(RecognizerConfig(model=folder))
        self.assertEqual(caught.exception.code, "MODEL_INCOMPLETE")

    def test_prepared_whisper_directory_is_read_locally_without_download(self):
        model = SimpleNamespace(feature_extractor=SimpleNamespace(chunk_length=30))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            for name in ("model.bin", "config.json", "tokenizer.json"):
                (path / name).write_bytes(b"test-model")
            create_manifest(path, backend="whisper", model_id="tiny", source="test", revision="abc")
            with patch("faster_whisper.WhisperModel", return_value=model) as loader:
                engine = WhisperEngine(RecognizerConfig(model=path, compute_type="int8"))
                engine.close()
        call = loader.call_args
        self.assertEqual(Path(call.args[0]), path.resolve())
        self.assertTrue(call.kwargs["local_files_only"])

    def test_engine_alias_is_normalized_before_loading(self):
        model = SimpleNamespace(feature_extractor=SimpleNamespace(chunk_length=30))
        with patch("faster_whisper.WhisperModel", return_value=model):
            recognizer = Recognizer(
                RecognizerConfig(model="tiny", engine="faster_whisper", compute_type="int8")
            )
            self.assertEqual(recognizer.config.engine, "whisper")
            recognizer.prepare()
            recognizer.close()

    def test_unknown_engine_reports_available_names(self):
        with self.assertRaises(ASRError) as caught:
            Recognizer(RecognizerConfig(model="tiny", engine="other"))
        self.assertIn("whisper", caught.exception.message)
        self.assertIn("nemotron", caught.exception.message)

    def test_nemotron_requires_a_prepared_directory(self):
        recognizer = Recognizer(RecognizerConfig(model="nemotron", engine="nemotron"))
        self.addCleanup(recognizer.close)
        with self.assertRaises(ASRError) as caught:
            recognizer.prepare()
        self.assertEqual(caught.exception.code, "MODEL_INCOMPLETE")
        self.assertIn("install_models", caught.exception.message)

    def test_engine_options_must_be_a_mapping(self):
        with self.assertRaises(ASRError) as caught:
            Recognizer(RecognizerConfig(model="tiny", engine_options=["beam_size"]))
        self.assertEqual(caught.exception.code, "INVALID_CONFIGURATION")

    def test_whisper_engine_options_override_beam_size(self):
        model = SimpleNamespace(feature_extractor=SimpleNamespace(chunk_length=30))
        info = SimpleNamespace(language=None, language_probability=None)
        with patch("faster_whisper.WhisperModel", return_value=model):
            engine = WhisperEngine(
                RecognizerConfig(model="tiny", compute_type="int8", engine_options={"beam_size": 2})
            )
        with patch(
            "faster_whisper.BatchedInferencePipeline.transcribe",
            return_value=(iter([]), info),
        ) as run:
            engine.transcribe([0.1])
        self.assertEqual(run.call_args.kwargs["beam_size"], 2)

    def test_invalid_engine_option_is_rejected_before_loading(self):
        with self.assertRaises(ASRError) as caught:
            WhisperEngine(
                RecognizerConfig(model="tiny", compute_type="int8", engine_options={"beam_size": 0})
            )
        self.assertEqual(caught.exception.code, "INVALID_CONFIGURATION")
        recognizer = Recognizer(
            RecognizerConfig(model="nemotron", engine="nemotron", engine_options={"num_threads": 0})
        )
        self.addCleanup(recognizer.close)
        with self.assertRaises(ASRError):
            recognizer.prepare()
