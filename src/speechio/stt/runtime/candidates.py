"""候选角色配置及固定模型输入策略，不持有会话可变状态。"""

from dataclasses import dataclass, field, replace
from math import isfinite

from speechio.errors import STTError
from speechio.models.config import LocalModelConfig, RemoteModelConfig


@dataclass(frozen=True, kw_only=True)
class OnDemandConfig:
    """显式开启候选，缺省共用主引擎；尾部长度使用秒。"""

    config: LocalModelConfig | RemoteModelConfig | None = None
    _adapter: object = field(default=None, init=False, repr=False, compare=False)
    tail_seconds: float = 3.0

    def __post_init__(self):
        if (
            self.config is not None
            and self._adapter is not None
            or self.config is not None
            and not isinstance(self.config, (LocalModelConfig, RemoteModelConfig))
            or isinstance(self.tail_seconds, bool)
            or not isinstance(self.tail_seconds, (int, float))
            or not isfinite(self.tail_seconds)
            or self.tail_seconds <= 0
        ):
            raise STTError("INVALID_CONFIGURATION", "configure_candidate", "候选模型或尾部秒数无效")

    @property
    def tail_samples(self):
        return max(1, int(self.tail_seconds * 16000))

    @classmethod
    def from_adapter(cls, adapter, *, tail_seconds=3.0):
        """使用已经构造的识别适配器；运行时负责共享与关闭。"""
        from speechio.adapters.contracts import validate_engine

        validate_engine(adapter)
        instance = cls(tail_seconds=tail_seconds)
        object.__setattr__(instance, "_adapter", adapter)
        return instance


EMPTY_RETRY_SAMPLES = 8000


def execute_candidate(model, request):
    """常规空结果仅补静音重试一次，真实音频及请求覆盖保持固定。"""
    result = model._candidate(
        request.pcm, request.language, word_timestamps=request.word_timestamps
    )
    attempts, added = 1, 0
    if request.retry_empty and not result.text.strip():
        attempts, added = 2, EMPTY_RETRY_SAMPLES
        try:
            result = model._candidate(
                request.pcm + bytes(added * 2),
                request.language,
                word_timestamps=request.word_timestamps,
            )
        except STTError as error:
            error.candidate_execution = {"attempts": attempts, "added_silence_samples": added}
            raise
    return replace(
        result,
        backend_metadata={
            **result.backend_metadata,
            "candidate_execution": {"attempts": attempts, "added_silence_samples": added},
        },
    )
