"""会话与执行器之间的不可变输入，不携带会话可变状态或策略上下文。"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from realtime_asr.runtime.candidates import EMPTY_RETRY_SAMPLES

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
    role: Literal["final", "candidate"] = "final"
    retry_empty: bool = False

    @property
    def reserved_samples(self):
        """包括最多一次候选补零的预留量，不扩大真实音频覆盖。"""
        return len(self.pcm) // 2 + (EMPTY_RETRY_SAMPLES if self.retry_empty else 0)
