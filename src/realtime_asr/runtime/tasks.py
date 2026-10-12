"""会话与执行器之间的不可变输入，不携带会话可变状态或策略上下文。"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from realtime_asr.session.session import FinalTask


@dataclass(frozen=True)
class InferenceRequest:
    """固定任务与选项；优先级只描述调度，不描述语音优化用途。"""

    task: "FinalTask"
    pcm: bytes
    language: str | None
    word_timestamps: bool = False
    priority: Literal["final", "auxiliary"] = "final"
    submitted_at: float = 0.0
