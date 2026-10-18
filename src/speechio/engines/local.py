"""本地模型的组装与生命周期；GPU、路径和语言映射不进入通用调度。"""

import logging
from collections.abc import Mapping
from dataclasses import replace
from os import PathLike
from pathlib import Path

from speechio.config import LocalModelConfig
from speechio.config import ModelConfig as ModelConfig
from speechio.engines.base import BaseTranscriptionEngine
from speechio.engines.factory import available_engines, create_engine, normalize_engine
from speechio.errors import ASRError
from speechio.results import CloseError, CloseResult

LOGGER = logging.getLogger(__name__)


def resolve_config(config):
    """固定目录的绝对位置并检查运行设置；存在性留到真正加载时检查。"""
    if not isinstance(config, LocalModelConfig):
        raise ASRError("INVALID_CONFIGURATION", "configure", "需要 LocalModelConfig")
    engine = normalize_engine(config.engine)
    if engine not in available_engines():
        raise ASRError(
            "BACKEND_INCOMPATIBLE",
            "configure",
            f"不支持的引擎 {engine}；可用：{', '.join(available_engines())}",
        )
    valid_index = isinstance(config.device_index, int) and not isinstance(config.device_index, bool)
    if (
        config.device not in ("cpu", "gpu")
        or not valid_index
        or config.device_index < 0
        or (config.device == "cpu" and config.device_index != 0)
        or (
            config.compute_type is not None
            and (not isinstance(config.compute_type, str) or not config.compute_type)
        )
    ):
        raise ASRError("INVALID_CONFIGURATION", "configure", "引擎或设备组合无效")
    if not isinstance(config.model_path, (str, PathLike)) or not str(config.model_path).strip():
        raise ASRError("INVALID_CONFIGURATION", "configure", "model_path 必须是模型目录")
    if config.engine_options is not None and not isinstance(config.engine_options, Mapping):
        raise ASRError("INVALID_CONFIGURATION", "configure", "engine_options 应为映射")
    try:
        engine_options = dict(config.engine_options) if config.engine_options is not None else None
        if (
            engine_options
            and {"device", "provider", "device_index", "compute_type"} & engine_options.keys()
        ):
            raise ASRError(
                "INVALID_CONFIGURATION", "configure", "设备和精度请使用 LocalModelConfig 的对应字段"
            )
        return replace(
            config,
            engine=engine,
            model_path=Path(config.model_path).expanduser().resolve(),
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
