"""一次识别的要求，与模型来源和音频格式分开。"""

from dataclasses import dataclass

from speechio.errors import STTError


@dataclass(frozen=True, kw_only=True)
class TranscriptionOptions:
    """language 不指定时由模型处理；词时间信息默认不请求。"""

    language: str | None = None
    word_timestamps: bool = False

    def __post_init__(self):
        if self.language is not None and (
            not isinstance(self.language, str) or not self.language.strip()
        ):
            raise STTError("INVALID_CONFIGURATION", "transcribe", "语言应为非空代码或 None")
        if type(self.word_timestamps) is not bool:
            raise STTError("INVALID_CONFIGURATION", "transcribe", "词时间戳开关应为布尔值")


def resolve_options(options):
    """在公开调用边界解析默认选项。"""
    if options is None:
        return TranscriptionOptions()
    if not isinstance(options, TranscriptionOptions):
        raise STTError("INVALID_CONFIGURATION", "transcribe", "需要 TranscriptionOptions")
    return options


def validate_task_timeout(seconds):
    """异步文件任务的总等待上限；不表示强制终止本地模型。"""
    from math import isfinite

    if (
        isinstance(seconds, bool)
        or not isinstance(seconds, (int, float))
        or not isfinite(seconds)
        or seconds <= 0
    ):
        raise STTError("INVALID_CONFIGURATION", "transcribe", "task_timeout_seconds 应为有限正数")
