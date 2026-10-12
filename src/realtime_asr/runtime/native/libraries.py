"""原生库准备只在模型创建阶段执行，不修改全局 PATH。"""

import ctypes
import importlib.util
import os
from pathlib import Path

from realtime_asr.errors import ASRError


def load_project_onnx_runtime():
    """Windows 显式加载项目 ONNX Runtime，避免系统旧 DLL 抢先匹配。"""
    if os.name != "nt":
        return None, None
    spec = importlib.util.find_spec("onnxruntime")
    if spec is None:
        raise ASRError("DEPENDENCY_MISSING", "load_model", "缺少项目 ONNX Runtime")
    path = Path(spec.origin).parent / "capi" / "onnxruntime.dll"
    try:
        handle = ctypes.WinDLL(str(path))
    except OSError as exc:
        raise ASRError("BACKEND_INCOMPATIBLE", "load_model", str(exc)) from exc
    return handle, str(path.resolve())
