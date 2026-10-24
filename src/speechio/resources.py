"""模型和任务共享的资源释放结果。"""

from dataclasses import dataclass


@dataclass(frozen=True)
class CloseError:
    """一次资源释放失败，不携带评估试次或音频范围。"""

    code: str
    operation: str
    message: str


@dataclass(frozen=True)
class CloseResult:
    """关闭状态：pending 可继续确认，ok/failed 为最终结论。"""

    status: str = "ok"
    cleanup_method: str = "reference_drop"
    errors: tuple[CloseError, ...] = ()
