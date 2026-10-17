"""原始 stdin 的可停止读取；借用标准流，不创建音频处理状态。"""

import io
import os
import sys
from threading import Event


def stdin_blocks(stream, format, check):
    """管道空闲时仍检查事件端故障，不阻塞命令关闭模型。"""
    block_bytes = 1024 * format.channels * (2 if format.sample_type == "int16" else 4)
    try:
        descriptor = stream.fileno()
    except (AttributeError, io.UnsupportedOperation):
        descriptor = None
    available = _pipe_reader(descriptor, block_bytes) if descriptor is not None else None
    if available is None:
        # 普通文件和调用方提供的有限流保持原读取语义。
        read = getattr(stream, "read1", stream.read)
        while chunk := read(block_bytes):
            check()
            yield chunk
        return
    pause = Event()
    while True:
        check()
        count = available()
        if count is None:
            return
        if count == 0:
            pause.wait(0.05)
            continue
        chunk = os.read(descriptor, min(count, block_bytes))
        if not chunk:
            return
        yield chunk


def _pipe_reader(descriptor, block_bytes):
    """只查询借用描述符的可读性，不更改阻塞模式或关闭句柄。"""
    if sys.platform != "win32":
        import select

        return lambda: block_bytes if select.select([descriptor], [], [], 0.05)[0] else 0
    import ctypes
    import msvcrt
    from ctypes import wintypes

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetFileType.argtypes = [wintypes.HANDLE]
    kernel.GetFileType.restype = wintypes.DWORD
    handle = msvcrt.get_osfhandle(descriptor)
    if kernel.GetFileType(handle) != 3:  # FILE_TYPE_PIPE。
        return None
    kernel.PeekNamedPipe.argtypes = [
        wintypes.HANDLE,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.LPVOID,
        ctypes.POINTER(wintypes.DWORD),
        wintypes.LPVOID,
    ]
    kernel.PeekNamedPipe.restype = wintypes.BOOL

    def available():
        count = wintypes.DWORD()
        if kernel.PeekNamedPipe(handle, None, 0, None, ctypes.byref(count), None):
            return count.value
        error = ctypes.get_last_error()
        if error == 109:  # ERROR_BROKEN_PIPE：上游已结束输入。
            return None
        raise ctypes.WinError(error)

    return available
