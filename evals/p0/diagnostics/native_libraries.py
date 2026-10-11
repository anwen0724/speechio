"""评估进程中实际已加载原生库的路径观测。"""

import ctypes
import os


def loaded_libraries():
    """只读枚举本进程实际加载路径；驱动安装路径不能代替此证据。"""
    if os.name != "nt":
        return []
    from ctypes import wintypes

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.EnumProcessModules.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(wintypes.HMODULE),
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
    ]
    psapi.GetModuleFileNameExW.argtypes = [
        wintypes.HANDLE,
        wintypes.HMODULE,
        wintypes.LPWSTR,
        wintypes.DWORD,
    ]
    process = kernel.GetCurrentProcess()
    modules, needed = (wintypes.HMODULE * 2048)(), wintypes.DWORD()
    if not psapi.EnumProcessModules(process, modules, ctypes.sizeof(modules), ctypes.byref(needed)):
        raise ctypes.WinError(ctypes.get_last_error())
    paths = []
    for handle in modules[: needed.value // ctypes.sizeof(wintypes.HMODULE)]:
        buffer = ctypes.create_unicode_buffer(32768)
        psapi.GetModuleFileNameExW(process, handle, buffer, len(buffer))
        lower = buffer.value.lower()
        if any(name in lower for name in ("cudnn", "cublas", "onnxruntime", "nvcuda", "cudart")):
            paths.append(buffer.value)
    return paths
