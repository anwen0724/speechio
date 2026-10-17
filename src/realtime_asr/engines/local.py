"""本地模型的组装与生命周期；GPU、路径和语言映射不进入通用调度。"""

import logging
from collections.abc import Mapping
from dataclasses import dataclass, replace
from os import PathLike
from pathlib import Path

from realtime_asr.engines.base import BaseTranscriptionEngine, EngineCapabilities
from realtime_asr.engines.factory import available_engines, create_engine, normalize_engine
from realtime_asr.errors import ASRError
from realtime_asr.results import CloseError, CloseResult

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True)
class RecognizerConfig:
    """显式本地配置；model 可以是模型名或已准备的本地目录。"""

    model: str | PathLike
    engine: str = "whisper"
    device: str = "cpu"
    compute_type: str = "default"
    device_index: int = 0
    download_root: str | PathLike | None = None
    local_files_only: bool = False
    engine_options: Mapping[str, object] | None = None

    @property
    def capabilities(self):
        """内置能力的唯一声明位置，查询不加载模型。"""
        engine = normalize_engine(self.engine)
        if engine not in ("whisper", "nemotron"):
            raise ASRError("INVALID_CONFIGURATION", "configure", "引擎或设备组合无效")
        whisper = engine == "whisper"
        return EngineCapabilities(
            engine=engine,
            model=str(self.model),
            device=self.device,
            snapshot=whisper,
            streaming=not whisper,
            word_timestamps=whisper,
        )


def resolve_config(config):
    """校验本地组合；模型名保持字符串，已存在的目录才解析为路径。"""
    if not isinstance(config, RecognizerConfig):
        raise ASRError("INVALID_CONFIGURATION", "configure", "需要 RecognizerConfig")
    engine = normalize_engine(config.engine)
    if engine not in ("whisper", "nemotron"):
        raise ASRError(
            "BACKEND_INCOMPATIBLE",
            "configure",
            f"不支持的引擎 {engine}；可用：{', '.join(available_engines())}",
        )
    valid_index = isinstance(config.device_index, int) and not isinstance(config.device_index, bool)
    if (
        config.device not in ("cpu", "cuda")
        or not valid_index
        or config.device_index < 0
        or (config.device == "cpu" and config.device_index != 0)
        or not isinstance(config.compute_type, str)
        or not config.compute_type
        or type(config.local_files_only) is not bool
    ):
        raise ASRError("INVALID_CONFIGURATION", "configure", "引擎或设备组合无效")
    if not isinstance(config.model, (str, PathLike)):
        raise ASRError("INVALID_CONFIGURATION", "configure", "模型应为名字或目录")
    if config.engine_options is not None and not isinstance(config.engine_options, Mapping):
        raise ASRError("INVALID_CONFIGURATION", "configure", "engine_options 应为映射")
    try:
        model = str(config.model)
        if Path(model).is_dir():
            model = Path(model).resolve()
        download_root = (
            Path(config.download_root).resolve() if config.download_root is not None else None
        )
        engine_options = dict(config.engine_options) if config.engine_options is not None else None
        return replace(
            config,
            engine=engine,
            model=model,
            download_root=download_root,
            engine_options=engine_options,
        )
    except (TypeError, ValueError, OSError) as exc:
        raise ASRError("INVALID_CONFIGURATION", "configure", "模型或运行库路径无效") from exc


class LocalEngine(BaseTranscriptionEngine):
    """隐藏本地准备、后端创建、参数映射与资源关闭，不拥有会话。"""

    def __init__(self, config):
        self.config = resolve_config(config)
        self._model = None

    @property
    def capabilities(self):
        return self.config.capabilities

    def prepare(self):
        if self._model is not None:
            return
        try:
            self._model = create_engine(self.config)
        except BaseException as exc:
            cleanup = self.close()
            if cleanup.errors:
                exc.cleanup_errors = cleanup.errors
            if isinstance(exc, (ASRError, KeyboardInterrupt, SystemExit)):
                raise
            code = "DEPENDENCY_MISSING" if isinstance(exc, (ImportError, OSError)) else "MODEL_LOAD"
            raise ASRError(code, "load_model", str(exc)) from exc

    def transcribe(self, audio, language=None, *, role="final", word_timestamps=False):
        if self.config.engine == "whisper":
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
        if model is not None:
            try:
                model.close()
            except Exception as exc:
                LOGGER.error("模型资源关闭失败：%s", exc)
                errors.append(CloseError("CLEANUP_FAILED", "close", str(exc)))
        return CloseResult(status="failed" if errors else "ok", errors=tuple(errors))
