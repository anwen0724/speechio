"""取消请求保留独立时点，不等原生调用返回才记作请求。"""

from threading import Event
from time import perf_counter_ns


class CancellationSource(Event):
    """沿用事件等待接口，首次 set 记录请求的单调时间。"""

    def __init__(self):
        super().__init__()
        self.requested_at_ns = None

    def set(self):
        """重复取消不改写第一次请求时点。"""
        if self.requested_at_ns is None:
            self.requested_at_ns = perf_counter_ns()
        super().set()
