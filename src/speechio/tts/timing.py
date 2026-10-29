"""合成文字在输出 PCM 时间轴上的位置，单位为秒。"""

from dataclasses import dataclass


@dataclass(frozen=True)
class WordTiming:
    """供应商或本地模型提供的一个字或词；时间从本次输出音频开头计算。"""

    text: str
    start_seconds: float
    end_seconds: float


@dataclass(frozen=True)
class TextTiming:
    """同一 segment_index 的后续事件更新该段；未结束时终点为 None。"""

    segment_index: int
    text: str
    start_seconds: float
    end_seconds: float | None = None
    words: tuple[WordTiming, ...] | None = None
