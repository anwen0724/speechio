"""SDK 使用真实转换与适配器，只替换外部模型加载和推理边界。"""

import hashlib
import json
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import soundfile as sf

from realtime_asr import ASRError, AudioFormat, Recognizer, RecognizerConfig
from realtime_asr.runtime.models.manifest import create_manifest


class RecognizerTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.path = Path(self.folder.name)
        for name in ("model.bin", "config.json", "tokenizer.json"):
            (self.path / name).write_bytes(b"test-model")
        create_manifest(
            self.path, backend="whisper", model_id="tiny", source="test", revision="abc"
        )
        model = SimpleNamespace(feature_extractor=SimpleNamespace(chunk_length=30))
        self.loader = patch("faster_whisper.WhisperModel", return_value=model)
        self.loader.start()
        self.addCleanup(self.loader.stop)

    def test_complete_sdk_audio_and_language_reach_model_without_tail_padding(self):
        expected = np.array([1, -1, 32767], np.int16).astype(np.float32) / 32768

        def native_transcribe(audio, **kwargs):
            np.testing.assert_array_equal(audio, expected)
            self.assertEqual(kwargs["language"], "zh")
            return iter([SimpleNamespace(text=" 结果", start=0, end=0.001)]), SimpleNamespace(
                language="zh", language_probability=0.9
            )

        with patch(
            "faster_whisper.BatchedInferencePipeline.transcribe", side_effect=native_transcribe
        ):
            recognizer = Recognizer(RecognizerConfig(model=self.path))
            result = recognizer.transcribe(
                np.array([1, -1, 32767], np.int16), format=AudioFormat(), language="zh-CN"
            )
        self.assertEqual(result.text, "结果")
        self.assertEqual(result.requested_language, "zh-CN")
        self.assertEqual(result.reported_language, "zh")
        self.assertEqual(result.language_source, "hint")
        self.assertEqual(recognizer.close().status, "ok")

    def test_busy_close_is_rejected_and_closed_runtime_does_not_consume_input(self):
        entered, release = threading.Event(), threading.Event()
        outcomes = []

        def native_transcribe(audio, **kwargs):
            entered.set()
            if not release.wait(5):
                raise RuntimeError("测试没有释放模型边界")
            return iter([]), SimpleNamespace(language=None, language_probability=None)

        recognizer = Recognizer(RecognizerConfig(model=self.path))

        def run():
            try:
                outcomes.append(recognizer.transcribe(b"\x00\x00", format=AudioFormat()))
            except Exception as exc:
                outcomes.append(exc)

        with patch(
            "faster_whisper.BatchedInferencePipeline.transcribe", side_effect=native_transcribe
        ):
            worker = threading.Thread(target=run)
            worker.start()
            try:
                self.assertTrue(entered.wait(5))
                with self.assertRaises(ASRError) as caught:
                    recognizer.close()
                self.assertEqual(caught.exception.code, "RUNTIME_BUSY")
            finally:
                release.set()
                worker.join(5)
        self.assertEqual(outcomes[0].text, "")
        first_close = recognizer.close()
        self.assertIs(recognizer.close(), first_close)

        def never_read():
            self.fail("关闭运行时不能读取输入")
            yield b"\x00\x00"

        with self.assertRaises(ASRError) as caught:
            recognizer.transcribe_chunks(never_read(), format=AudioFormat())
        self.assertEqual(caught.exception.code, "RUNTIME_CLOSED")

    def test_three_entries_deliver_same_complete_audio_and_reuse_model(self):
        source = np.arange(525, dtype=np.int16)
        inputs = []

        def native_transcribe(audio, **kwargs):
            inputs.append(audio.copy())
            return iter([SimpleNamespace(text="same", start=0, end=1)]), SimpleNamespace(
                language="en", language_probability=None
            )

        path = self.path / "input.wav"
        sf.write(path, source, 16000, subtype="PCM_16")
        recognizer = Recognizer(RecognizerConfig(model=self.path))
        with patch(
            "faster_whisper.BatchedInferencePipeline.transcribe", side_effect=native_transcribe
        ):
            result = recognizer.transcribe(source, format=AudioFormat())
            chunked = recognizer.transcribe_chunks([source[:10], source[10:]], format=AudioFormat())
            file = recognizer.transcribe_file(path)
        self.assertEqual([result.text, chunked.text, file.text], ["same"] * 3)
        for actual in inputs:
            np.testing.assert_array_equal(actual, source.astype(np.float32) / 32768)
        self.assertEqual(recognizer.state, "ready")
        self.assertEqual(recognizer.close().status, "ok")

    def test_context_prepares_and_closes_without_suppressing_body_error(self):
        recognizer = Recognizer(RecognizerConfig(model=self.path))
        original = RuntimeError("调用方失败")
        with self.assertRaises(RuntimeError) as caught:
            with recognizer as ready:
                self.assertIs(ready, recognizer)
                self.assertEqual(ready.state, "ready")
                raise original
        self.assertIs(caught.exception, original)
        self.assertEqual(recognizer.state, "closed")
        self.assertEqual(recognizer.close_result.status, "ok")

    def test_gpu_close_attempts_all_owned_directories_and_preserves_body_error(self):
        cudnn = self.path / "cudnn"
        (cudnn / "bin").mkdir(parents=True)
        dll = cudnn / "bin/cudnn64_9.dll"
        dll.write_bytes(b"native-test")
        (cudnn / "native-manifest.json").write_text(
            json.dumps(
                {
                    "files": [
                        {
                            "path": "bin/cudnn64_9.dll",
                            "size_bytes": 11,
                            "sha256": hashlib.sha256(b"native-test").hexdigest(),
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        closed = []

        class Directory:
            def __init__(self, name):
                self.name = name

            def close(self):
                closed.append(self.name)
                if self.name == "cuda":
                    raise OSError("目录句柄释放失败")

        config = RecognizerConfig(
            model=self.path,
            device="cuda",
            cudnn_folder=cudnn,
            cuda_directory=self.path / "cuda",
        )
        recognizer = Recognizer(config)
        original = ValueError("正文错误")
        with (
            patch("os.add_dll_directory", side_effect=[Directory("cuda"), Directory("cudnn")]),
            patch("ctypes.WinDLL", return_value=object()),
        ):
            with self.assertRaises(ValueError) as caught:
                with recognizer:
                    raise original
        self.assertIs(caught.exception, original)
        self.assertEqual(closed, ["cuda", "cudnn"])
        self.assertEqual(recognizer.state, "closed")
        self.assertEqual(recognizer.close_result.status, "failed")
        self.assertEqual(recognizer.close_result.errors[0].code, "CLEANUP_FAILED")
        self.assertIs(recognizer.close(), recognizer.close_result)

    def test_unknown_model_backend_is_rejected_before_preparation(self):
        path = self.path / "model-manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest["backend"] = "foreign"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        recognizer = Recognizer(RecognizerConfig(model=self.path))
        with self.assertRaises(ASRError) as caught:
            recognizer.prepare()
        self.assertEqual(caught.exception.code, "BACKEND_INCOMPATIBLE")
        self.assertEqual(recognizer.state, "unprepared")

    def test_iterated_prefix_failure_does_not_prepare_or_publish_a_result(self):
        def failed_input():
            yield b"\x00\x00"
            raise OSError("来源中断")

        recognizer = Recognizer(RecognizerConfig(model=self.path))
        with patch("faster_whisper.WhisperModel", side_effect=AssertionError("不能加载模型")):
            with self.assertRaises(ASRError) as caught:
                recognizer.transcribe_chunks(failed_input(), format=AudioFormat())
        self.assertEqual(caught.exception.code, "AUDIO_READ_FAILED")
        self.assertEqual(caught.exception.operation, "read_chunks")
        self.assertEqual(recognizer.state, "unprepared")

    def test_prepare_failure_can_retry_and_late_model_failure_is_not_prefix_success(self):
        recognizer = Recognizer(RecognizerConfig(model=self.path))
        with patch("faster_whisper.WhisperModel", side_effect=RuntimeError("加载失败")):
            with self.assertRaises(ASRError) as caught:
                recognizer.prepare()
        self.assertEqual(caught.exception.code, "MODEL_LOAD")
        self.assertEqual(recognizer.state, "unprepared")
        recognizer.prepare()

        def segments():
            yield SimpleNamespace(text="前缀", start=0, end=1)
            raise RuntimeError("后续模型分段失败")

        with patch(
            "faster_whisper.BatchedInferencePipeline.transcribe",
            return_value=(segments(), SimpleNamespace(language="en", language_probability=None)),
        ):
            with self.assertRaises(ASRError) as caught:
                recognizer.transcribe(b"\x00\x00", format=AudioFormat())
        self.assertEqual(caught.exception.code, "INFERENCE_FAILED")
        self.assertEqual(recognizer.state, "ready")
        self.assertEqual(recognizer.close().status, "ok")

    def test_invalid_language_and_format_are_rejected_before_input_iteration(self):
        def never_read():
            self.fail("参数失败不应读取输入")
            yield b""

        recognizer = Recognizer(RecognizerConfig(model=self.path))
        with self.assertRaises(ASRError) as caught:
            recognizer.transcribe_chunks(never_read(), format=AudioFormat(), language=1)
        self.assertEqual(caught.exception.code, "INVALID_CONFIGURATION")
        with self.assertRaises(ASRError) as caught:
            recognizer.transcribe_chunks(never_read(), format=None)
        self.assertEqual(caught.exception.code, "INVALID_AUDIO_FORMAT")
