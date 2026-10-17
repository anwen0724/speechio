"""识别结果只包含后端信息，试次归属由验证入口补充。"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class WordTimestamp:
    """模型返回的词范围，位置相对于本次标准输入，不表示墙钟时间。"""

    text: str
    start_sample: int
    end_sample: int
    probability: float | None = None


@dataclass(frozen=True)
class Transcript:
    """保留原文、真实语言来源和可用分段，不伪造未知值。"""

    text: str = ""
    requested_language: str | None = None
    reported_language: str | None = None
    language_source: str = "unavailable"
    language_probability: float | None = None
    segments: list[dict] | None = None
    backend_metadata: dict = field(default_factory=dict)
    words: tuple[WordTimestamp, ...] | None = None
    timestamp_status: str = "not_requested"


@dataclass(frozen=True)
class CloseError:
    """一次资源释放失败，不携带评估试次或音频范围。"""

    code: str
    operation: str
    message: str


@dataclass(frozen=True)
class CloseResult:
    """记录首次关闭结论；引用清除不等于原生库或显存已经卸载。"""

    status: str = "ok"
    cleanup_method: str = "reference_drop"
    errors: tuple[CloseError, ...] = ()
