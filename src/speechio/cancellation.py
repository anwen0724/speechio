"""单次任务的协作取消；请求取消不代表底层执行已经退出。"""

from threading import Event, Lock


class CancellationToken:
    """可跨线程请求取消；每个任务使用一个令牌，不支持重置。"""

    def __init__(self):
        self._event = Event()
        self._lock = Lock()
        self._callbacks = set()
        self._callback_errors = []
        self._notes = []

    def _add_note(self, note):
        with self._lock:
            self._notes.append(note)

    def _annotate(self, error):
        with self._lock:
            notes = tuple(self._notes)
        for note in notes:
            if note not in getattr(error, "__notes__", ()):
                error.add_note(note)

    @property
    def cancel_requested(self):
        return self._event.is_set()

    def cancel(self):
        """幂等发出请求；内部回调只唤醒等待或中断传输，不等待资源释放。"""
        with self._lock:
            if self._event.is_set():
                return
            self._event.set()
            callbacks = tuple(self._callbacks)
            self._callbacks.clear()
        for callback in callbacks:
            self._invoke(callback)

    def _invoke(self, callback):
        try:
            callback()
        except Exception as exc:
            # 一个适配器的中断失败不能阻止其他等待者收到取消。
            with self._lock:
                self._callback_errors.append(type(exc).__name__)

    def wait_cancelled(self, timeout=None):
        return self._event.wait(timeout)

    def _subscribe(self, callback):
        with self._lock:
            cancelled = self._event.is_set()
            if not cancelled:
                self._callbacks.add(callback)
        if cancelled:
            self._invoke(callback)

        def unsubscribe():
            with self._lock:
                self._callbacks.discard(callback)

        return unsubscribe
