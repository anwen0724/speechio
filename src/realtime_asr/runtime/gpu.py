"""项目自有 cuDNN 准备与进程内 DLL 选择。"""

import ctypes
import hashlib
import json
import os
import zipfile
from pathlib import Path

from realtime_asr.core.errors import ASRError
from realtime_asr.runtime.models.manifest import check_file, sha256_file

CUDNN_VERSION = "9.16.0.29"
CUDA_DIRECTORY = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin")


def verify_runtime(folder, manifest):
    """目录存在不能代替完整清单；清单路径必须属于所选运行库。"""
    files = manifest.get("files", [])
    if not files or not any(item["path"] == "bin/cudnn64_9.dll" for item in files):
        raise ASRError("DEPENDENCY_MISSING", "prepare_gpu", "cuDNN 清单缺少必需 DLL")
    for item in files:
        path = (folder / item["path"]).resolve()
        if not path.is_relative_to(folder):
            raise ASRError("MODEL_INTEGRITY", "prepare_gpu", "原生库清单路径越界")
        check_file(path, item["size_bytes"], item["sha256"])


def prepare_cudnn(runtime_root):
    """显式获取 NVIDIA 原生库包，校验来源哈希后展开到项目目录。"""
    import requests

    root = Path(runtime_root).resolve()
    folder = root / "cudnn" / CUDNN_VERSION
    manifest_path = folder / "native-manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        verify_runtime(folder, manifest)
        return folder
    root.mkdir(parents=True, exist_ok=True)
    metadata = requests.get(
        f"https://pypi.org/pypi/nvidia-cudnn-cu12/{CUDNN_VERSION}/json", timeout=(15, 30)
    )
    metadata.raise_for_status()
    source = next(item for item in metadata.json()["urls"] if "win_amd64" in item["filename"])
    wheel = root / source["filename"]
    if not wheel.exists():
        partial = wheel.with_suffix(wheel.suffix + ".partial")
        with requests.get(source["url"], stream=True, timeout=(15, 30)) as response:
            response.raise_for_status()
            with partial.open("wb") as output:
                for block in response.iter_content(1024 * 1024):
                    output.write(block)
        check_file(partial, source["size"], source["digests"]["sha256"])
        partial.rename(wheel)
    check_file(wheel, source["size"], source["digests"]["sha256"])
    bin_folder = folder / "bin"
    bin_folder.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(wheel) as package:
        for name in package.namelist():
            if "/bin/" in name and name.endswith(".dll"):
                target = bin_folder / Path(name).name
                with package.open(name) as source_file:
                    digest = hashlib.file_digest(source_file, "sha256").hexdigest()
                if target.exists():
                    check_file(target, package.getinfo(name).file_size, digest)
                else:
                    with package.open(name) as source_file, target.open("xb") as output:
                        import shutil

                        shutil.copyfileobj(source_file, output)
    files = [
        {
            "path": p.relative_to(folder).as_posix(),
            "size_bytes": p.stat().st_size,
            "sha256": sha256_file(p),
        }
        for p in sorted(bin_folder.glob("*.dll"))
    ]
    if not files:
        raise ASRError("MODEL_PREPARE", "prepare_gpu", "NVIDIA 包没有 Windows DLL")
    manifest_path.write_text(
        json.dumps(
            {
                "package": "nvidia-cudnn-cu12",
                "version": CUDNN_VERSION,
                "source": source["url"],
                "archive_sha256": source["digests"]["sha256"],
                "files": files,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return folder


class GPUContext:
    """显式加载所选 CUDA 与 cuDNN，句柄在模型使用期间保持有效。"""

    def __init__(self, cudnn_folder, cuda_directory=CUDA_DIRECTORY):
        self.cudnn_folder = Path(cudnn_folder).resolve()
        self.cuda_directory = Path(cuda_directory).resolve()
        self._directories = []
        self._libraries = []

    def __enter__(self):
        if os.name != "nt":
            raise ASRError("BACKEND_INCOMPATIBLE", "prepare_gpu", "本入口只验证 Windows GPU")
        manifest_file = self.cudnn_folder / "native-manifest.json"
        if not manifest_file.is_file():
            raise ASRError("DEPENDENCY_MISSING", "prepare_gpu", "项目 cuDNN 尚未显式准备")
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        verify_runtime(self.cudnn_folder, manifest)
        try:
            for directory in (self.cuda_directory, self.cudnn_folder / "bin"):
                self._directories.append(os.add_dll_directory(str(directory)))
            paths = [
                self.cuda_directory / "cublasLt64_12.dll",
                self.cuda_directory / "cublas64_12.dll",
            ]
            paths.extend(sorted((self.cudnn_folder / "bin").glob("*.dll")))
            for path in paths:
                self._libraries.append(ctypes.WinDLL(str(path)))
        except OSError as exc:
            self.__exit__(None, None, None)
            raise ASRError("DEPENDENCY_MISSING", "prepare_gpu", str(exc)) from exc
        return self

    def __exit__(self, exc_type, exc, tb):
        for directory in self._directories:
            directory.close()
        self._directories.clear()
        self._libraries.clear()
