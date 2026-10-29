"""异步与同步共用执行流程；等待者退出不转移底层资源所有权。"""

import asyncio
from contextvars import ContextVar, copy_context
from functools import wraps
from inspect import Parameter, signature
from threading import Condition, Thread

from speechio.cancellation import CancellationToken

_current = ContextVar("speechio_cancellation", default=None)


def current_cancellation():
    context = _current.get()
    return None if context is None else context[0]


class AsyncCondition(Condition):
    """同步状态锁兼容异步通知；等待线程不会代替调用方消费队列。"""

    def __init__(self):
        super().__init__()
        self._async_waiters = set()

    def subscribe(self):
        """调用者持有条件锁，保证检查状态和登记通知之间没有空隙。"""
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        self._async_waiters.add((loop, future))
        return future

    def unsubscribe(self, future):
        with self:
            self._async_waiters = {item for item in self._async_waiters if item[1] is not future}

    def notify_all(self):
        super().notify_all()
        for loop, future in tuple(self._async_waiters):
            try:
                loop.call_soon_threadsafe(self._wake, future)
            except RuntimeError:
                self._async_waiters.discard((loop, future))

    @staticmethod
    def _wake(future):
        if not future.done():
            future.set_result(None)


def check_cancelled():
    """在输入、模型调用和结果交付边界检查当前任务。"""
    context = _current.get()
    if context is not None:
        token, error_type, operation = context
        if token.cancel_requested:
            error = error_type("CANCELLED", operation, "任务已取消")
            token._annotate(error)
            for failure in token._callback_errors:
                error.add_note(f"适配器中断请求失败：{failure}；资源仍需等待底层退出后清理")
            raise error


def cancellable(error_type, *, discard=None):
    """公共同步方法建立任务范围；嵌套准备与适配器调用复用同一令牌。"""

    def decorate(method):
        @wraps(method)
        def call(*args, **kwargs):
            cancellation = kwargs.get("cancellation")
            inherited = _current.get()
            token = (
                cancellation
                if cancellation is not None
                else (inherited[0] if inherited else CancellationToken())
            )
            if not isinstance(token, CancellationToken):
                raise error_type(
                    "INVALID_INPUT", method.__name__, "cancellation 需要 CancellationToken"
                )
            context = _current.set((token, error_type, method.__name__))
            try:
                check_cancelled()
                result = method(*args, **kwargs)
                if token.cancel_requested and discard is not None:
                    discard(result)
                check_cancelled()
                return result
            except Exception as exc:
                if token.cancel_requested and getattr(exc, "code", None) != "CANCELLED":
                    error = error_type("CANCELLED", method.__name__, "任务已取消")
                    token._annotate(error)
                    for note in getattr(exc, "__notes__", ()):
                        if note not in getattr(error, "__notes__", ()):
                            error.add_note(note)
                    raise error from None
                raise
            finally:
                _current.reset(context)

        call._cancelled_result = discard
        call._cancellation_error = error_type
        return call

    return decorate


class AdapterCalls:
    """接入时检查签名；旧适配器不接收新参数，也不因 TypeError 重试。"""

    def __init__(self, adapter):
        self._methods = {}
        for name in (
            "prepare",
            "transcribe",
            "transcribe_file",
            "transcribe_url",
            "create_streaming_session",
            "synthesize",
            "clone_voice",
            "list_voices",
            "stream",
            "open_stream",
        ):
            method = getattr(adapter, name, None)
            if not callable(method):
                continue
            try:
                parameters = signature(method).parameters
                accepts = "cancellation" in parameters or any(
                    p.kind == Parameter.VAR_KEYWORD for p in parameters.values()
                )
            except (ValueError, TypeError):
                accepts = False
            self._methods[name] = method, accepts

    def call(self, name, *args, **kwargs):
        check_cancelled()
        method, accepts = self._methods[name]
        context = _current.get()
        if accepts and context is not None:
            kwargs["cancellation"] = context[0]
        result = method(*args, **kwargs)
        # 流的创建结果必须先交还拥有者登记，才能处理创建期间的取消。
        if name not in ("stream", "open_stream", "create_streaming_session"):
            check_cancelled()
        return result


async def run_blocking(method, *args, cancellation=None, on_cancel=None, discard=None, **kwargs):
    """库拥有的工作线程完成同步流程；取消 await 不会提前释放执行资源。"""
    loop = asyncio.get_running_loop()
    future = loop.create_future()
    context = copy_context()

    def deliver(result, error):
        if future.done():
            if error is None and discard is not None:
                Thread(
                    target=discard, args=(result,), name="speechio-cancel-cleanup", daemon=True
                ).start()
        else:
            if error is None:
                future.set_result(result)
            else:
                future.set_exception(error)

    def execute():
        try:
            result, error = context.run(method, *args, **kwargs), None
        except BaseException as exc:
            result, error = None, exc
        try:
            loop.call_soon_threadsafe(deliver, result, error)
        except RuntimeError:
            # 事件循环已退出，后台执行仍由同步对象负责收尾。
            if error is None and discard is not None:
                discard(result)

    Thread(target=execute, name="speechio-operation", daemon=True).start()
    try:
        return await future
    except asyncio.CancelledError as exc:
        if cancellation is not None:
            cancellation.cancel()
            cancellation._annotate(exc)
        if on_cancel is not None:
            on_cancel()
        if (
            discard is not None
            and future.done()
            and not future.cancelled()
            and future.exception() is None
        ):
            Thread(
                target=discard, args=(future.result(),), name="speechio-cancel-cleanup", daemon=True
            ).start()
        raise


async def run_operation(method, *args, cancellation=None, **kwargs):
    """异步入口自动创建令牌，显式令牌与 Task.cancel 共用同步控制流程。"""
    token = cancellation if cancellation is not None else CancellationToken()
    result = await run_blocking(
        lambda: method(*args, cancellation=token, **kwargs),
        cancellation=token,
        discard=getattr(method, "_cancelled_result", None),
    )
    # 取消可能发生在工作线程完成之后、事件循环交付之前。
    if token.cancel_requested:
        discard = getattr(method, "_cancelled_result", None)
        if discard is not None:
            await run_blocking(discard, result)
        error = method._cancellation_error("CANCELLED", method.__name__, "任务已取消")
        token._annotate(error)
        raise error
    return result


def wait_cancellable(seconds):
    """轮询间隔可被取消唤醒；不会撤销已提交的远程任务。"""
    token = current_cancellation()
    if token is None:
        from time import sleep

        sleep(seconds)
    else:
        token.wait_cancelled(seconds)
        check_cancelled()
