"""公开合成任务：接收音频、限制缓冲并管理结束与取消。"""

import asyncio
from collections import deque
from threading import Event, Lock, Thread
from time import monotonic

from speechio._async import AsyncCondition, run_blocking
from speechio.audio.results import AudioChunk, validate_pcm
from speechio.errors import TTSError
from speechio.resources import CloseError, CloseResult
from speechio.tts.options import validate_seconds
from speechio.tts.timing import TextTiming


class SynthesisStream:
    """一个发送方与一个读取方可并行工作，关闭不会释放共享引擎。"""

    def __init__(self, backend, *, incremental, max_buffer_seconds, text_timestamps=None):
        validate_pcm(b"", backend.format)
        if text_timestamps is not None and not callable(getattr(backend, "read_event", None)):
            raise TTSError(
                "PROTOCOL_ERROR", "open_stream", "声明文字时间能力的后端须提供 read_event"
            )
        self._backend = backend
        self._text_timestamps = text_timestamps
        self._read_mode = None
        self._format = backend.format
        self._incremental = incremental
        self._state = "accepting" if incremental else "finishing"
        self._condition = AsyncCondition()
        self._sender = Lock()
        self._async_sender = asyncio.Lock()
        self._reader = Lock()
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
        self._unsubscribe_cancel = None
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
                chunk = (
                    self._backend.read_event()
                    if self._text_timestamps is not None
                    else self._backend.read()
                )
                with self._condition:
                    if self._state in ("cancelled", "failed"):
                        break
                    if chunk is None:
                        if self._state != "finishing":
                            raise TTSError("PROTOCOL_ERROR", "read", "输入未结束时后端提前完成")
                        self._state = "completed"
                        self._condition.notify_all()
                        break
                    if isinstance(chunk, TextTiming):
                        # 音频容量不能约束无音频的元信息；事件队列另设数量上限。
                        if len(self._queue) >= 1024:
                            raise TTSError("BUFFER_EXCEEDED", "read", "待读取合成事件超过缓冲上限")
                        self._queue.append(chunk)
                        self._condition.notify_all()
                        continue
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
            # 中断请求不等于发送调用已经返回；关闭须等发送方退出后接管后端。
            self._sender.acquire()
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
            finally:
                self._sender.release()
            with self._condition:
                self._done.set()
                self._condition.notify_all()

    def read(self, *, timeout=None):
        """只读取音频，跳过文字事件；不能与事件读取方式混用。"""
        return self._read("audio", timeout)

    def read_event(self, *, timeout=None):
        """按顺序读取 AudioChunk 或 TextTiming；None 表示成功读尽。"""
        return self._read("events", timeout)

    def events(self):
        """逐个交付事件，同一段文字可能在结束时收到更新。"""
        while (event := self.read_event()) is not None:
            yield event

    def _read(self, mode, timeout):
        if not self._reader.acquire(blocking=False):
            raise TTSError("INVALID_INPUT", "read", "同一合成流只允许一个读取方")
        try:
            return self._read_next(mode, timeout)
        finally:
            self._reader.release()

    def _read_next(self, mode, timeout):
        """两种读取方式共用一条队列和同一个等待截止时间。"""
        validate_seconds(timeout, "timeout", optional=True)
        deadline = None if timeout is None else monotonic() + timeout
        with self._condition:
            if self._read_mode is not None and self._read_mode != mode:
                raise TTSError("INVALID_INPUT", "read", "同一合成流不能混用音频读取和事件读取")
            self._read_mode = mode
            while True:
                if self._error is not None:
                    raise self._error
                if self._queue:
                    chunk = self._queue.popleft()
                    if isinstance(chunk, AudioChunk):
                        self._buffer_bytes -= len(chunk.data)
                    self._condition.notify_all()
                    if mode == "audio" and isinstance(chunk, TextTiming):
                        continue
                    return chunk
                if self._state == "completed" and self._done.is_set():
                    self._drained = True
                    self._unbind_cancellation()
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
            if self._state in ("failed", "cancelled") or self._drained:
                return
            finished = self._done.is_set()
            self._state = "cancelled"
            self._drained = True
            self._error = TTSError("CANCELLED", "read", "合成任务已取消")
            self._queue.clear()
            self._buffer_bytes = 0
            self._condition.notify_all()
        if not finished:
            self._request_cancel()
        self._unbind_cancellation()

    def _bind_cancellation(self, token):
        """令牌持续控制整条流，直到读尽或取消；结束后解除引用。"""
        self._unsubscribe_cancel = token._subscribe(self.cancel)
        if self._drained:
            self._unbind_cancellation()

    def _unbind_cancellation(self):
        if self._unsubscribe_cancel is not None:
            self._unsubscribe_cancel()
            self._unsubscribe_cancel = None

    def close(self, *, timeout=5.0):
        validate_seconds(timeout, "timeout")
        self._close_requested = True
        self.cancel()
        self._unbind_cancellation()
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

    async def _read_async(self, mode, timeout):
        """只在事件循环恢复后消费；取消等待不会遗留后台读取者。"""
        validate_seconds(timeout, "timeout", optional=True)
        if not self._reader.acquire(blocking=False):
            raise TTSError("INVALID_INPUT", "read", "同一合成流只允许一个读取方")
        deadline = None if timeout is None else monotonic() + timeout
        try:
            while True:
                with self._condition:
                    try:
                        return self._read_next(mode, 0)
                    except TTSError as exc:
                        if exc.code != "READ_TIMEOUT":
                            raise
                    remaining = None if deadline is None else deadline - monotonic()
                    if remaining is not None and remaining <= 0:
                        raise TTSError("READ_TIMEOUT", "read", "本次音频读取等待超时")
                    wake = self._condition.subscribe()
                try:
                    await asyncio.wait_for(wake, remaining)
                except TimeoutError:
                    raise TTSError("READ_TIMEOUT", "read", "本次音频读取等待超时") from None
                finally:
                    self._condition.unsubscribe(wake)
        finally:
            self._reader.release()

    async def read_async(self, *, timeout=None):
        return await self._read_async("audio", timeout)

    async def read_event_async(self, *, timeout=None):
        return await self._read_async("events", timeout)

    async def send_text_async(self, text):
        # 有副作用的发送无法回滚；取消等待意味着取消整条合成流。
        try:
            async with self._async_sender:
                await run_blocking(self.send_text, text, on_cancel=self.cancel)
        except asyncio.CancelledError:
            self.cancel()
            raise

    async def finish_input_async(self):
        try:
            async with self._async_sender:
                await run_blocking(self.finish_input, on_cancel=self.cancel)
        except asyncio.CancelledError:
            self.cancel()
            raise

    async def close_async(self, *, timeout=5.0):
        return await run_blocking(self.close, timeout=timeout)

    def __aiter__(self):
        return self

    async def __anext__(self):
        chunk = await self.read_async()
        if chunk is None:
            raise StopAsyncIteration
        return chunk

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        result = await self.close_async()
        if result.status != "ok":
            if exc is not None:
                exc.add_note(f"合成流关闭状态：{result.status}")
            else:
                raise TTSError("CLOSE_FAILED", "close_stream", f"合成流关闭状态：{result.status}")
