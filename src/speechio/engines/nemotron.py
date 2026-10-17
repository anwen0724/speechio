"""Nemotron CPU 识别器与独立原生流的组装入口。"""

import inspect
from pathlib import Path

from speechio.audio.conversion import as_audio
from speechio.engines.stream import NativeStream
from speechio.errors import ASRError
from speechio.runtime.models.manifest import verify_manifest
from speechio.runtime.models.stages import preparation_step


class NemotronEngine:
    """一个引擎拥有一个识别器；调用方按串行顺序访问不同流。"""

    def __init__(self, config):
        self.preparation_steps = []
        options = config.engine_options or {}
        num_threads = options.get("num_threads", 1)
        if type(num_threads) is not int or num_threads <= 0:
            raise ASRError(
                "INVALID_CONFIGURATION", "configure_engine", "engine_options.num_threads 应为正整数"
            )
        self.num_threads = num_threads
        # 运行后端由调用方显式选择；库不检查系统显卡与驱动。
        provider = options.get("provider", "cpu")
        if not isinstance(provider, str) or not provider:
            raise ASRError(
                "INVALID_CONFIGURATION",
                "configure_engine",
                "engine_options.provider 应为非空字符串",
            )
        device_index = options.get("device", 0)
        if type(device_index) is not int or device_index < 0:
            raise ASRError(
                "INVALID_CONFIGURATION", "configure_engine", "engine_options.device 应为非负整数"
            )
        self.provider = provider
        self.device_index = device_index
        model_path = Path(config.model)
        if not model_path.is_dir():
            raise ASRError(
                "MODEL_INCOMPLETE",
                "verify_model",
                f"未找到 Nemotron 模型目录：{config.model}；"
                '先调用 install_models(<root>, "560ms") 准备模型',
            )
        with preparation_step(self.preparation_steps, "verification"):
            self.manifest = verify_manifest(model_path)
            if self.manifest["backend"] != "nemotron":
                raise ASRError("BACKEND_INCOMPATIBLE", "verify_model", "模型不是 Nemotron 后端")
        with preparation_step(self.preparation_steps, "load"):
            self._load_recognizer()

    def _load_recognizer(self):
        """在明确的加载区间内展开参数并创建识别器。"""
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
                "num_threads": self.num_threads,
                "sample_rate": 16000,
                "feature_dim": 128,
                "enable_endpoint_detection": False,
                "provider": self.provider,
                "device": self.device_index,
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
