"""读取实际运行环境，不从设计文档推导已安装能力。"""

import csv
import importlib.metadata
import os
import platform
import struct
import subprocess
import sys
from datetime import UTC, datetime
from uuid import uuid4


def gpu_devices():
    """使用驱动工具读取 GPU；失败原因与没有设备分别返回。"""
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=15, check=True)
        devices = []
        for index, name, memory, driver in csv.reader(result.stdout.splitlines()):
            devices.append(
                {
                    "device_index": int(index),
                    "name": name.strip(),
                    "memory_total_bytes": int(memory.strip()) * 1024 * 1024,
                    "driver_version": driver.strip(),
                    "runtime_cuda_version": None,
                    "runtime_cudnn_version": None,
                }
            )
        return devices, None
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        return [], f"{type(exc).__name__}: {exc}"


def inspect_environment():
    """记录当前解释器与包；GPU DLL 的实际加载另由 GPU 运行记录证明。"""
    import psutil

    dependencies, missing = {}, []
    for name in (
        "numpy",
        "faster-whisper",
        "ctranslate2",
        "sherpa-onnx",
        "onnxruntime",
        "psutil",
        "scipy",
    ):
        try:
            dependencies[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            missing.append({"name": name, "reason": "当前解释器未安装"})
    cpu = platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER")
    if os.name == "nt":
        import winreg

        try:
            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            ) as key:
                cpu = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        except OSError:
            pass
    gpus, gpu_reason = gpu_devices()
    unknown = [
        {
            "name": "runtime_cuda_version/runtime_cudnn_version",
            "reason": "尚未进行实际 GPU 推理与 DLL 追踪",
        }
    ]
    if not cpu:
        unknown.append({"name": "cpu_name", "reason": "系统接口未提供"})
    if gpu_reason:
        unknown.append({"name": "gpus", "reason": gpu_reason})
    return {
        "environment_id": uuid4().hex,
        "recorded_at": datetime.now(UTC).isoformat(),
        "os_version": platform.platform(),
        "os_build": platform.version(),
        "python_version": platform.python_version(),
        "python_path": str(sys.executable),
        "python_bits": struct.calcsize("P") * 8,
        "cpu_name": cpu,
        "logical_processors": os.cpu_count(),
        "memory_total_bytes": psutil.virtual_memory().total,
        "dependencies": dependencies,
        "gpus": gpus,
        "unknown_fields": unknown,
        "missing_dependencies": missing,
    }
