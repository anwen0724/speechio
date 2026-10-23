"""共享音频格式，保持原构造校验和默认值。"""

from dataclasses import dataclass

import numpy as np

from speechio.errors import STTError

SAMPLE_RATE = 16000


@dataclass(frozen=True)
class AudioFormat:
    """声明源音频的采样率、声道和幅度语义；字节输入固定采用小端。"""

    sample_rate: int = SAMPLE_RATE
    channels: int = 1
    sample_type: str = "int16"

    def __post_init__(self):
        if not _positive_integer(self.sample_rate):
            raise STTError("INVALID_SAMPLE_RATE", "convert_audio", "采样率应为正整数")
        if not _positive_integer(self.channels) or self.sample_type not in ("int16", "float32"):
            raise STTError("INVALID_AUDIO_FORMAT", "convert_audio", "声道或采样类型无效")


def _positive_integer(value):
    """公开整数参数不接受布尔值；不限制没有需求依据的最大值。"""
    return isinstance(value, (int, np.integer)) and not isinstance(value, bool) and value > 0
