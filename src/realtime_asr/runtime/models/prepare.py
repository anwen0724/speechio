"""显式获取模型；推理模块只接收已经校验的本地目录。"""

import shutil
import tarfile
from pathlib import Path
from urllib.request import urlopen

from realtime_asr.errors import ASRError
from realtime_asr.runtime.models.manifest import (
    MANIFEST_NAME,
    check_file,
    create_manifest,
    sha256_file,
    verify_manifest,
)
from realtime_asr.runtime.models.pinned import DETECTION_MODELS, WHISPER_MODEL, nemotron_model


def prepare_model(
    backend, folder, *, allow_download=False, variant=None, local_snapshot=None, local_source=None
):
    """统一准备本地资源；身份、缓存和文件规则不进入 CLI 或会话。"""
    if (
        (variant is not None and backend != "nemotron")
        or (local_snapshot is not None and backend != "whisper")
        or (local_source is not None and backend not in DETECTION_MODELS)
    ):
        raise ASRError("INVALID_CONFIGURATION", "prepare_model", "准备参数不属于所选资源")
    if backend == "whisper":
        expected_id = WHISPER_MODEL["id"]
    elif backend == "nemotron":
        expected_id = nemotron_model(variant or "560ms")["id"]
    elif backend in DETECTION_MODELS:
        expected_id = DETECTION_MODELS[backend]["id"]
    else:
        raise ASRError("BACKEND_INCOMPATIBLE", "prepare_model", "资源种类不受支持")
    folder = Path(folder)
    if (folder / MANIFEST_NAME).is_file():
        manifest = verify_manifest(folder)
        if (manifest["backend"], manifest["model_id"]) != (backend, expected_id):
            raise ASRError("MODEL_INTEGRITY", "prepare_model", "已有清单不是所选模型身份")
        return manifest
    if backend == "whisper":
        return prepare_whisper(folder, allow_download=allow_download, local_snapshot=local_snapshot)
    if backend == "nemotron":
        return prepare_nemotron(folder, allow_download=allow_download, variant=variant or "560ms")
    return prepare_detection(
        backend, folder, allow_download=allow_download, local_source=local_source
    )


def prepare_detection(backend, folder, *, allow_download=False, local_source=None):
    """固定小型 ONNX 资源先核对再命名，不导入任何检测运行库。"""
    folder = Path(folder).resolve()
    model = DETECTION_MODELS[backend]
    if (folder / MANIFEST_NAME).exists():
        manifest = verify_manifest(folder)
        if manifest["backend"] != backend:
            raise ASRError("MODEL_INTEGRITY", "prepare_model", "已有清单不是所选资源")
        return manifest
    folder.mkdir(parents=True, exist_ok=True)
    for name, expected in model["files"].items():
        target = folder / name
        if target.exists():
            check_file(target, *expected)
            continue
        if local_source is not None:
            source = Path(local_source)
            source = source if source.is_file() and len(model["files"]) == 1 else source / name
            check_file(source, *expected)
            shutil.copyfile(source, target)
        elif allow_download:
            partial = target.with_suffix(target.suffix + ".partial")
            with urlopen(model["urls"][name], timeout=30) as response, partial.open("wb") as output:
                shutil.copyfileobj(response, output)
            check_file(partial, *expected)
            partial.replace(target)
        else:
            raise ASRError(
                "MODEL_INCOMPLETE", "prepare_model", "缺少固定文件，提供本地来源或明确 --download"
            )
    return create_manifest(
        folder,
        backend=backend,
        model_id=model["id"],
        source=model["source"],
        revision=model["revision"],
    )


def prepare_whisper(folder, *, allow_download=False, local_snapshot=None):
    """复用固定本地快照，或在显式授权获取时解析不可变仓库修订。"""
    folder = Path(folder).resolve()
    if (folder / MANIFEST_NAME).exists():
        return verify_manifest(folder)
    folder.mkdir(parents=True, exist_ok=True)
    source = WHISPER_MODEL["source"]
    if local_snapshot is not None:
        snapshot = Path(local_snapshot).resolve()
        revision = snapshot.name
        if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
            raise ASRError("MODEL_PREPARE", "prepare_model", "本地快照应位于不可变修订目录")
        _copy_snapshot(snapshot, folder)
    elif allow_download:
        from huggingface_hub import HfApi, snapshot_download

        revision = HfApi().model_info(source).sha
        staging = folder.parent / f".{folder.name}.download-{revision}"
        snapshot_download(repo_id=source, revision=revision, local_dir=staging)
        _copy_snapshot(staging, folder)
    else:
        raise ASRError("MODEL_INCOMPLETE", "prepare_model", "离线准备缺少已固定的模型清单")
    return create_manifest(
        folder, backend="whisper", model_id=WHISPER_MODEL["id"], source=source, revision=revision
    )


def _copy_snapshot(snapshot, folder):
    """远端先落到独立缓存；冲突核对通过后才复制，未知目标不覆盖。"""
    files = [path for path in snapshot.iterdir() if path.is_file()]
    for path in files:
        target = folder / path.name
        if target.exists() and sha256_file(target) != sha256_file(path):
            raise ASRError("MODEL_INTEGRITY", "prepare_model", f"已有文件不符：{target}")
    for path in files:
        target = folder / path.name
        if not target.exists():
            shutil.copyfile(path, target)


def prepare_nemotron(folder, *, allow_download=False, variant="560ms"):
    """固定发布包先校验，再安全展开；已有模型只做离线核对。"""
    folder = Path(folder).resolve()
    model = nemotron_model(variant)
    model_id = model["id"]
    url = f"https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/{model_id}.tar.bz2"
    if (folder / MANIFEST_NAME).exists():
        manifest = verify_manifest(folder)
        if manifest.get("variant", "560ms") != variant:
            raise ASRError("MODEL_INTEGRITY", "prepare_model", "已有模型不是所选变体")
        return manifest
    if not allow_download:
        raise ASRError("MODEL_INCOMPLETE", "prepare_model", "离线准备缺少完整 Nemotron 模型")
    folder.parent.mkdir(parents=True, exist_ok=True)
    archive = folder.parent / f"{model_id}.tar.bz2"
    if not archive.exists():
        partial = archive.with_suffix(archive.suffix + ".partial")
        import requests

        with requests.get(url, stream=True, timeout=(15, 30)) as response:
            response.raise_for_status()
            with partial.open("wb") as output:
                for block in response.iter_content(chunk_size=1024 * 1024):
                    output.write(block)
        check_file(partial, *model["archive"])
        partial.rename(archive)
    check_file(archive, *model["archive"])
    extracted = folder.parent / model_id
    if not extracted.exists():
        with tarfile.open(archive, "r:bz2") as package:
            package.extractall(folder.parent, filter="data")
    # 已存在的展开目录只校验，不用重新解包覆盖未知内容。
    for name, expected in model["files"].items():
        check_file(extracted / name, *expected)
    if extracted != folder:
        if folder.exists():
            raise ASRError("MODEL_PREPARE", "prepare_model", "目标目录已存在，保留已有文件")
        extracted.rename(folder)
    return create_manifest(
        folder,
        backend="nemotron",
        model_id=model_id,
        source=url,
        revision=model_id,
        variant=variant,
    )
