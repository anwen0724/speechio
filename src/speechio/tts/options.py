"""每次合成的要求，与模型来源配置分开。"""

from dataclasses import dataclass
from math import isfinite

from speechio.errors import TTSError
from speechio.tts.voices import Voice


def validate_seconds(value, name, *, positive=False, optional=False):
    """只在公开时间边界校验，不允许布尔值或无穷等待参数。"""
    if value is None and optional:
        return
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(value)
        or (value <= 0 if positive else value < 0)
    ):
        raise TTSError("INVALID_CONFIGURATION", "configure", f"{name} 的秒数无效")


@dataclass(frozen=True, kw_only=True)
class SynthesisOptions:
    """未指定音色时保留模型原生默认行为，其余设置按模型能力校验。"""

    voice: str | Voice | None = None
    language: str | None = None
    speed: float = 1.0
    instruction: str | None = None
    sample_rate: int | None = None
    text_timestamps: str | None = None

    def __post_init__(self):
        if self.text_timestamps not in (None, "sentence", "word"):
            raise TTSError(
                "INVALID_INPUT", "synthesize", "text_timestamps 应为 None、sentence 或 word"
            )
        if (
            self.voice is not None
            and not isinstance(self.voice, Voice)
            and (not isinstance(self.voice, str) or not self.voice.strip())
        ):
            raise TTSError("INVALID_INPUT", "synthesize", "voice 应为非空音色 ID 或 Voice")
        for name in ("language", "instruction"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise TTSError("INVALID_INPUT", "synthesize", f"{name} 应为非空字符串或 None")
        if (
            isinstance(self.speed, bool)
            or not isinstance(self.speed, (int, float))
            or not isfinite(self.speed)
            or self.speed <= 0
            or (
                self.sample_rate is not None
                and (type(self.sample_rate) is not int or self.sample_rate <= 0)
            )
        ):
            raise TTSError("INVALID_INPUT", "synthesize", "语速或采样率无效")
