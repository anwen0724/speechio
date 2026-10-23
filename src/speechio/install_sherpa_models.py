"""把固定 Nemotron 权重安装到显式目录；推理启动时不下载。"""

import argparse
import sys
from pathlib import Path

from speechio.errors import ASRError
from speechio.model_manifests import nemotron_manifests
from speechio.models.files import check_model_files
from speechio.models.manifest import sha256_file
from speechio.models.prepare import prepare_nemotron


class ModelInstallError(RuntimeError):
    """模型无法校验或安全安装。"""


MODEL_MANIFESTS = nemotron_manifests()


def manifests_for_selection(selection):
    """把模型选择解析为固定清单；未知名称直接报可选项。"""
    value = str(selection).strip().lower()
    if value == "all":
        return tuple(MODEL_MANIFESTS.items())
    if value in MODEL_MANIFESTS:
        return ((value, MODEL_MANIFESTS[value]),)
    raise ModelInstallError(
        f"未知模型选择 {selection!r}；可选：{', '.join(MODEL_MANIFESTS)} 或 all"
    )


def verify_archive(path, manifest):
    """安装前核对压缩包大小和 SHA-256。"""
    path = Path(path)
    if not path.is_file():
        raise ModelInstallError(f"压缩包不存在：{path}")
    size = path.stat().st_size
    if size != manifest.archive_size_bytes:
        raise ModelInstallError(
            f"压缩包大小不符：{path}（{size}，期望 {manifest.archive_size_bytes}）"
        )
    digest = sha256_file(path)
    if digest.lower() != manifest.archive_sha256.lower():
        raise ModelInstallError(f"压缩包 SHA-256 不符：{path}")


def install_models(root, model="all", *, offline=False):
    """安装所选模型并返回名称到目录的映射；已有有效模型直接复用。"""
    root = Path(root).expanduser()
    if root.is_symlink():
        raise ModelInstallError(f"模型根目录不能是符号链接：{root}")
    root.mkdir(parents=True, exist_ok=True)
    installed = {}
    for name, manifest in manifests_for_selection(model):
        destination = root / manifest.model_id
        if destination.is_symlink():
            raise ModelInstallError(f"目标目录不能是符号链接：{destination}")
        if destination.is_dir() and offline:
            check_model_files("nemotron", destination)
            installed[name] = destination
            continue
        if offline:
            raise ModelInstallError(f"离线安装缺少模型 {name}：{destination}")
        prepare_nemotron(destination, allow_download=True, variant=name)
        installed[name] = destination
    return installed


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(
        prog="asr-install-sherpa-models", description="安装固定 Nemotron 权重"
    )
    parser.add_argument("--root", type=Path, required=True, help="模型根目录")
    parser.add_argument("--model", choices=(*MODEL_MANIFESTS, "all"), default="all")
    parser.add_argument("--offline", action="store_true", help="只校验已有模型，不联网")
    args = parser.parse_args(argv)
    try:
        installed = install_models(args.root, args.model, offline=args.offline)
    except (ModelInstallError, ASRError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    for name, path in installed.items():
        print(f"{name}: {path}")
    return 0
