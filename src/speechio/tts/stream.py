"""公开合成任务：接收音频、限制缓冲并管理结束与取消。"""

from collections import deque
from threading import Condition, Event, Lock, Thread
from time import monotonic

from speechio.audio.results import AudioChunk, validate_pcm
from speechio.errors import TTSError
from speechio.resources import CloseError, CloseResult
from speechio.tts.options import validate_seconds


class SynthesisStream:
    """一个发送方与一个读取方可并行工作，关闭不会释放共享引擎。"""

    def __init__(self, backend, *, incremental, max_buffer_seconds):
        validate_pcm(b"", backend.format)
        self._backend = backend
        self._format = backend.format
        self._incremental = incremental
        self._state = "accepting" if incremental else "finishing"
        self._condition = Condition()
        self._sender = Lock()
        self._queue = deque()
        self._buffer_bytes = 0
        self._max_bytes = max_buffer_seconds * self.format.sample_rate * 2
        self._has_text = not incremental
        self._error = None
        self._drained = False
        self._close_requested = False
        self._done = Event()
        self._cancel_done = Event()
        self._cancel_done.set()
        self._cancel_thread = None
        self._cancel_error = None
        self._close_result = None
        self._worker = Thread(target=self._receive, name="speechio-tts-audio", daemon=True)
        self._worker.start()

    @property
    def format(self):
        return self._format

    @property
    def state(self):
        with self._condition:
            return self._state

    @property
    def closed(self):
        return (
            self._done.is_set()
            and self._cancel_done.is_set()
            and (self._close_result is not None and self._close_result.status == "ok")
        )

    @property
    def released(self):
        """供拥有者判断任务槽是否可复用；音频未读尽时仍保持占用。"""
        return self.closed and (self._drained or self._close_requested)

    def send_text(self, text):
        if not isinstance(text, str):
            raise TTSError("INVALID_INPUT", "send_text", "文本片段应为字符串")
        with self._sender:
            with self._condition:
                if not self._incremental or self._state != "accepting":
                    raise TTSError("INPUT_CLOSED", "send_text", "该任务已不能追加文本")
                if not text:
                    return
                self._has_text = self._has_text or bool(text.strip())
            try:
                self._backend.send_text(text)
            except Exception as exc:
                self._fail(exc, "send_text")
                raise self._error from None

    def finish_input(self):
        """结束输入只发送一次；不等待服务生成剩余音频。"""
        with self._sender:
            with self._condition:
                if self._state in ("finishing", "completed"):
                    return
                if self._error is not None:
                    raise self._error
                if not self._has_text:
                    error = TTSError("INVALID_INPUT", "finish_input", "没有有效合成文本")
                else:
                    error = None
                    self._state = "finishing"
            if error is not None:
                self._fail(error, "finish_input")
                raise error
            try:
                self._backend.finish_input()
            except Exception as exc:
                self._fail(exc, "finish_input")
                raise self._error from None

    def _fail(self, exc, operation):
        with self._condition:
            if self._state not in ("failed", "cancelled"):
                self._state = "failed"
                self._error = (
                    exc
                    if isinstance(exc, TTSError)
                    else TTSError("SYNTHESIS_FAILED", operation, "合成后端操作失败")
                )
                self._queue.clear()
                self._buffer_bytes = 0
                self._condition.notify_all()
        self._request_cancel()

    def _receive(self):
        """专用接收线程只缓冲未读音频；失败时不继续交付残余音频。"""
        try:
            while True:
                with self._condition:
                    if self._state in ("cancelled", "failed"):
                        break
                chunk = self._backend.read()
                with self._condition:
                    if self._state in ("cancelled", "failed"):
                        break
                    if chunk is None:
                        if self._state != "finishing":
                            raise TTSError("PROTOCOL_ERROR", "read", "输入未结束时后端提前完成")
                        self._state = "completed"
                        self._condition.notify_all()
                        break
                    if not isinstance(chunk, AudioChunk) or chunk.format != self.format:
                        raise TTSError("PROTOCOL_ERROR", "read", "音频块格式不一致")
                    if getattr(self._backend, "pull_based", False):
                        # 本地生成可以等待消费者；保持每次入队不超过配置容量。
                        offset = 0
                        capacity = max(2, int(self._max_bytes) // 2 * 2)
                        while offset < len(chunk.data):
                            data = chunk.data[offset : offset + capacity]
                            while self._buffer_bytes + len(data) > capacity and self._state not in (
                                "cancelled",
                                "failed",
                            ):
                                self._condition.wait()
                            if self._state in ("cancelled", "failed"):
                                break
                            self._queue.append(AudioChunk(data, self.format))
                            self._buffer_bytes += len(data)
                            offset += len(data)
                            self._condition.notify_all()
                    else:
                        if self._buffer_bytes + len(chunk.data) > self._max_bytes:
                            raise TTSError("BUFFER_EXCEEDED", "read", "待读取音频超过缓冲上限")
                        if chunk.data:
                            self._queue.append(chunk)
                            self._buffer_bytes += len(chunk.data)
                            self._condition.notify_all()
        except BaseException as exc:
            self._fail(exc, "read")
        finally:
            # 取消可能仍在发送控制消息，连接不能先于该操作释放。
            self._cancel_done.wait()
            try:
                result = self._backend.close()
                while isinstance(result, CloseResult) and result.status == "pending":
                    # 后端保留执行中的资源；这里只在后台查询最终释放结论。
                    self._done.wait(0.05)
                    result = self._backend.close()
                self._close_result = result if isinstance(result, CloseResult) else CloseResult()
                if self._cancel_error is not None:
                    self._close_result = CloseResult(status="failed", errors=(self._cancel_error,))
            except BaseException:
                self._close_result = CloseResult(
                    status="failed",
                    errors=(CloseError("CLOSE_FAILED", "close_stream", "合成流释放失败"),),
                )
            with self._condition:
                self._done.set()
                self._condition.notify_all()

    def read(self, *, timeout=None):
        """None 只表示成功读尽；读取超时不会改变任务状态。"""
        validate_seconds(timeout, "timeout", optional=True)
        deadline = None if timeout is None else monotonic() + timeout
        with self._condition:
            while True:
                if self._error is not None:
                    raise self._error
                if self._queue:
                    chunk = self._queue.popleft()
                    self._buffer_bytes -= len(chunk.data)
                    self._condition.notify_all()
                    return chunk
                if self._state == "completed" and self._done.is_set():
                    self._drained = True
                    return None
                remaining = None if deadline is None else deadline - monotonic()
                if remaining is not None and remaining <= 0:
                    raise TTSError("READ_TIMEOUT", "read", "本次音频读取等待超时")
                self._condition.wait(remaining)

    def _request_cancel(self):
        with self._condition:
            if self._cancel_thread is None:
                self._cancel_done.clear()
                self._cancel_thread = Thread(
                    target=self._cancel_backend, name="speechio-tts-cancel", daemon=True
                )
                self._cancel_thread.start()

    def _cancel_backend(self):
        try:
            self._backend.cancel()
        except BaseException:
            self._cancel_error = CloseError("CLOSE_FAILED", "cancel", "后端取消请求失败")
        finally:
            self._cancel_done.set()

    def cancel(self):
        with self._condition:
            if self._state in ("completed", "failed", "cancelled"):
                return
            self._state = "cancelled"
            self._error = TTSError("CANCELLED", "read", "合成任务已取消")
            self._queue.clear()
            self._buffer_bytes = 0
            self._condition.notify_all()
        self._request_cancel()

    def close(self, *, timeout=5.0):
        validate_seconds(timeout, "timeout")
        self._close_requested = True
        self.cancel()
        with self._condition:
            self._queue.clear()
            self._buffer_bytes = 0
        deadline = monotonic() + timeout
        if not self._done.wait(timeout) or not self._cancel_done.wait(
            max(0, deadline - monotonic())
        ):
            return CloseResult(status="pending", cleanup_method="deferred")
        return self._close_result

    def wait_closed(self):
        """拥有者的后台清理使用；不让调用线程无限等待。"""
        self._done.wait()
        self._cancel_done.wait()
        return self._close_result

    def __iter__(self):
        return self

    def __next__(self):
        chunk = self.read()
        if chunk is None:
            raise StopIteration
        return chunk

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        result = self.close()
        if result.status != "ok":
            if exc is not None:
                exc.add_note(f"合成流关闭状态：{result.status}")
            else:
                raise TTSError("CLOSE_FAILED", "close_stream", f"合成流关闭状态：{result.status}")
