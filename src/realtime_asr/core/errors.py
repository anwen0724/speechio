"""提供稳定的失败分类，保留外部运行库原始异常。"""


class ASRError(RuntimeError):
    """本地操作失败；代码用于判断边界，消息用于定位原因。"""

    def __init__(self, code: str, operation: str, message: str):
        super().__init__(message)
        self.code = code
        self.operation = operation


def error_record(exc: Exception, operation: str, confirmed_range=None, unconfirmed_range=None):
    """将错误与最后确认的音频范围一起保存，不猜测失败调用是否生效。"""
    return {
        "operation": operation,
        "error_code": exc.code if isinstance(exc, ASRError) else "INFERENCE_FAILED",
        "exception_type": type(exc).__name__,
        "message": str(exc),
        "confirmed_range": confirmed_range,
        "unconfirmed_range": unconfirmed_range,
    }
