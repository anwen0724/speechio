"""Nemotron CPU 识别器与独立原生流的组装入口。"""

import inspect
from pathlib import Path

from realtime_asr.audio.conversion import as_audio
from realtime_asr.backends.stream import NativeStream
from realtime_asr.errors import ASRError
from realtime_asr.runtime.models.manifest import verify_manifest
from realtime_asr.runtime.models.stages import preparation_step
from realtime_asr.runtime.native.libraries import load_project_onnx_runtime


class NemotronEngine:
    """一个引擎拥有一个识别器；调用方按串行顺序访问不同流。"""

    def __init__(self, model_path):
        self.preparation_steps = []
        with preparation_step(self.preparation_steps, "verification"):
            self.manifest = verify_manifest(model_path)
            if self.manifest["backend"] != "nemotron":
                raise ASRError("BACKEND_INCOMPATIBLE", "verify_model", "模型不是 Nemotron 后端")
        with preparation_step(self.preparation_steps, "load"):
            self._load_recognizer()

    def _load_recognizer(self):
        """在明确的加载区间内选择原生库、展开参数并创建识别器。"""
        self._native_runtime, self.runtime_path = load_project_onnx_runtime()
        try:
            from sherpa_onnx import OnlineRecognizer
        except ImportError as exc:
            raise ASRError("DEPENDENCY_MISSING", "load_model", str(exc)) from exc
        self.parameters = {
            key: parameter.default
            for key, parameter in inspect.signature(
                OnlineRecognizer.from_transducer
            ).parameters.items()
            if parameter.default is not inspect.Parameter.empty
        }
        folder = Path(self.manifest["model_path"])
        self.parameters.update(
            {
                "tokens": str(folder / "tokens.txt"),
                "encoder": str(folder / "encoder.int8.onnx"),
                "decoder": str(folder / "decoder.int8.onnx"),
                "joiner": str(folder / "joiner.int8.onnx"),
                "num_threads": 1,
                "sample_rate": 16000,
                "feature_dim": 128,
                "enable_endpoint_detection": False,
                "provider": "cpu",
                "model_type": "",
            }
        )
        try:
            self._recognizer = OnlineRecognizer.from_transducer(**self.parameters)
        except Exception as exc:
            raise ASRError("MODEL_LOAD", "load_model", str(exc)) from exc

    def resolved_parameters(self):
        """记录实际工厂参数和统一调用参数，不加载额外热词或规则。"""
        return {
            **self.parameters,
            "normalize_audio": False,
            "use_prompt": True,
            "native_runtime_path": self.runtime_path,
        }

    def create_streaming_session(self, language=None, use_prompt=True):
        """创建本流语言和解码缓存，避免共享会话状态。"""
        return NativeStream(self._recognizer, language, use_prompt)

    def transcribe(self, audio, language=None, use_prompt=True, word_timestamps=False):
        """一次输入仍走显式结束与关闭；不以包装代替新增样本验证。"""
        array = as_audio(audio, allow_column=True)
        stream = self.create_streaming_session(language, use_prompt)
        try:
            stream.accept_audio(array)
            result = stream.finish()
        except BaseException as exc:
            try:
                stream.close()
            except Exception:
                # 保留原推理异常，清理错误作为独立证据附在该异常上。
                exc.cleanup = dict(stream.cleanup)
            raise
        else:
            stream.close()
            return result

    def close(self):
        """释放自有识别器引用；不承诺宿主进程资源立即回到原值。"""
        self._recognizer = None
        self._native_runtime = None
