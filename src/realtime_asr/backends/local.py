"""本地模型的组装与生命周期；GPU、路径和语言映射不进入通用调度。"""

import logging
from dataclasses import dataclass, replace
from os import PathLike
from pathlib import Path

from realtime_asr.backends.base import BaseTranscriptionEngine, EngineCapabilities
from realtime_asr.errors import ASRError
from realtime_asr.results import CloseError, CloseResult
from realtime_asr.runtime.native.gpu import CUDA_DIRECTORY, GPUContext

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True)
class RecognizerConfig:
    """显式本地配置；自定义或远程引擎不需要创建此对象。"""

    model_path: str | PathLike
    backend: str = "whisper"
    device: str = "cpu"
    compute_type: str | None = None
    device_index: int = 0
    cudnn_folder: str | PathLike | None = None
    cuda_directory: str | PathLike | None = None

    @property
    def capabilities(self):
        """内置能力的唯一声明位置，查询不加载模型。"""
        if self.backend not in ("whisper", "nemotron"):
            raise ASRError("INVALID_CONFIGURATION", "configure", "后端或设备组合无效")
        whisper = self.backend == "whisper"
        return EngineCapabilities(
            backend=self.backend,
            model=str(self.model_path),
            device=self.device,
            snapshot=whisper,
            streaming=not whisper,
            word_timestamps=whisper,
        )


def resolve_config(config):
    """保留已确认本地组合，固定路径，不触碰文件或加载 DLL。"""
    if not isinstance(config, RecognizerConfig):
        raise ASRError("INVALID_CONFIGURATION", "configure", "需要 RecognizerConfig")
    valid_index = isinstance(config.device_index, int) and not isinstance(config.device_index, bool)
    if (
        config.backend not in ("whisper", "nemotron")
        or config.device not in ("cpu", "cuda")
        or not valid_index
        or config.device_index < 0
        or (config.device == "cpu" and config.device_index != 0)
        or (config.backend == "nemotron" and config.device != "cpu")
    ):
        raise ASRError("INVALID_CONFIGURATION", "configure", "后端或设备组合无效")
    compute_type = "float16" if config.device == "cuda" else "int8"
    if config.compute_type not in (None, compute_type):
        raise ASRError("INVALID_CONFIGURATION", "configure", "本阶段计算类型与设备不匹配")
    if config.device == "cpu" and (
        config.cudnn_folder is not None or config.cuda_directory is not None
    ):
        raise ASRError("INVALID_CONFIGURATION", "configure", "CPU 不使用 GPU 路径")
    if config.device == "cuda" and config.cudnn_folder is None:
        raise ASRError("INVALID_CONFIGURATION", "configure", "CUDA 需要显式 cuDNN 目录")
    try:
        cudnn_folder = cuda_directory = None
        if config.device == "cuda":
            cudnn_folder = Path(config.cudnn_folder).resolve()
            cuda_directory = Path(config.cuda_directory or CUDA_DIRECTORY).resolve()
        return replace(
            config,
            model_path=Path(config.model_path).resolve(),
            compute_type=compute_type,
            cudnn_folder=cudnn_folder,
            cuda_directory=cuda_directory,
        )
    except (TypeError, ValueError, OSError) as exc:
        raise ASRError("INVALID_CONFIGURATION", "configure", "模型或运行库路径无效") from exc


def validate_local_pair(first, second):
    """只有本地 GPU 组装才检查进程运行库，共用实例由运行时管理。"""
    if not isinstance(first, LocalEngine) or not isinstance(second, LocalEngine):
        return
    left, right = first.config, second.config
    if left.device == right.device == "cuda" and (
        left.cuda_directory != right.cuda_directory or left.cudnn_folder != right.cudnn_folder
    ):
        raise ASRError("INVALID_CONFIGURATION", "configure", "CUDA 角色需要相同运行库目录")


class LocalEngine(BaseTranscriptionEngine):
    """隐藏本地准备、后端创建、参数映射与资源关闭，不拥有会话。"""

    def __init__(self, config):
        self.config = resolve_config(config)
        self._model = None
        self._gpu = None

    @property
    def capabilities(self):
        return self.config.capabilities

    def prepare(self):
        if self._model is not None:
            return
        try:
            if self.config.device == "cuda":
                self._gpu = GPUContext(self.config.cudnn_folder, self.config.cuda_directory)
                self._gpu.__enter__()
            if self.config.backend == "whisper":
                from realtime_asr.backends.whisper import WhisperEngine

                self._model = WhisperEngine(
                    self.config.model_path,
                    device=self.config.device,
                    compute_type=self.config.compute_type,
                    device_index=self.config.device_index,
                )
            else:
                from realtime_asr.backends.nemotron import NemotronEngine

                self._model = NemotronEngine(self.config.model_path)
        except BaseException as exc:
            cleanup = self.close()
            if cleanup.errors:
                exc.cleanup_errors = cleanup.errors
            if isinstance(exc, (ASRError, KeyboardInterrupt, SystemExit)):
                raise
            code = "DEPENDENCY_MISSING" if isinstance(exc, (ImportError, OSError)) else "MODEL_LOAD"
            raise ASRError(code, "load_model", str(exc)) from exc

    def transcribe(self, audio, language=None, *, role="final", word_timestamps=False):
        if self.config.backend == "whisper":
            mapped = {"zh-CN": "zh", "en-US": "en"}.get(language, language)
            # 候选使用完整识别参数，模型内部仍只维护最终与实时两种参数组。
            parameter_role = "final" if role == "candidate" else role
            result = self._model.transcribe(
                audio, language=mapped, role=parameter_role, word_timestamps=word_timestamps
            )
            if mapped is not None:
                result = replace(
                    result, language_source="hint" if result.reported_language else "unavailable"
                )
            return result
        mapped = {None: "auto", "zh": "zh-CN", "en": "en-US"}.get(language, language)
        return self._model.transcribe(audio, language=mapped)

    def create_streaming_session(self, language=None):
        mapped = {None: "auto", "zh": "zh-CN", "en": "en-US"}.get(language, language)
        return self._model.create_streaming_session(language=mapped)

    def close(self):
        """全部资源均尝试关闭；错误独立保留，引用始终清除。"""
        errors = []
        model, self._model = self._model, None
        gpu, self._gpu = self._gpu, None
        if model is not None:
            try:
                model.close()
            except Exception as exc:
                LOGGER.error("模型资源关闭失败：%s", exc)
                errors.append(CloseError("CLEANUP_FAILED", "close", str(exc)))
        if gpu is not None:
            try:
                gpu.__exit__(None, None, None)
            except Exception as exc:
                errors.append(CloseError("CLEANUP_FAILED", "close", str(exc)))
        return CloseResult(status="failed" if errors else "ok", errors=tuple(errors))
