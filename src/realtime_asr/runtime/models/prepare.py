"""显式获取模型；推理模块只接收已经校验的本地目录。"""

import shutil
import tarfile
from pathlib import Path

from realtime_asr.core.errors import ASRError
from realtime_asr.runtime.models.manifest import (
    MANIFEST_NAME,
    check_file,
    create_manifest,
    sha256_file,
    verify_manifest,
)
from realtime_asr.runtime.models.pinned import NEMOTRON_ARCHIVE, NEMOTRON_ID, NEMOTRON_URL


def prepare_whisper(folder, *, allow_download=False, local_snapshot=None):
    """复用固定本地快照，或在显式授权获取时解析不可变仓库修订。"""
    folder = Path(folder).resolve()
    if (folder / MANIFEST_NAME).exists():
        return verify_manifest(folder)
    folder.mkdir(parents=True, exist_ok=True)
    source = "Systran/faster-whisper-tiny"
    if local_snapshot is not None:
        snapshot = Path(local_snapshot).resolve()
        revision = snapshot.name
        if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
            raise ASRError("MODEL_PREPARE", "prepare_model", "本地快照应位于不可变修订目录")
        for path in snapshot.iterdir():
            if path.is_file():
                target = folder / path.name
                if target.exists() and sha256_file(target) != sha256_file(path):
                    raise ASRError("MODEL_INTEGRITY", "prepare_model", f"已有文件不符：{target}")
                if not target.exists():
                    shutil.copyfile(path, target)
    elif allow_download:
        from huggingface_hub import HfApi, snapshot_download

        revision = HfApi().model_info(source).sha
        snapshot_download(repo_id=source, revision=revision, local_dir=folder)
    else:
        raise ASRError("MODEL_INCOMPLETE", "prepare_model", "离线准备缺少已固定的模型清单")
    return create_manifest(
        folder, backend="whisper", model_id="tiny", source=source, revision=revision
    )


def prepare_nemotron(folder, *, allow_download=False):
    """固定发布包先校验，再安全展开；已有模型只做离线核对。"""
    folder = Path(folder).resolve()
    if (folder / MANIFEST_NAME).exists():
        return verify_manifest(folder)
    if not allow_download:
        raise ASRError("MODEL_INCOMPLETE", "prepare_model", "离线准备缺少完整 Nemotron 模型")
    folder.parent.mkdir(parents=True, exist_ok=True)
    archive = folder.parent / f"{NEMOTRON_ID}.tar.bz2"
    if not archive.exists():
        partial = archive.with_suffix(archive.suffix + ".partial")
        import requests

        with requests.get(NEMOTRON_URL, stream=True, timeout=(15, 30)) as response:
            response.raise_for_status()
            with partial.open("wb") as output:
                for block in response.iter_content(chunk_size=1024 * 1024):
                    output.write(block)
        check_file(partial, *NEMOTRON_ARCHIVE)
        partial.rename(archive)
    check_file(archive, *NEMOTRON_ARCHIVE)
    with tarfile.open(archive, "r:bz2") as package:
        package.extractall(folder.parent, filter="data")
    extracted = folder.parent / NEMOTRON_ID
    if extracted != folder:
        if folder.exists():
            raise ASRError("MODEL_PREPARE", "prepare_model", "目标目录已存在，保留已有文件")
        extracted.rename(folder)
    return create_manifest(
        folder,
        backend="nemotron",
        model_id=NEMOTRON_ID,
        source=NEMOTRON_URL,
        revision=NEMOTRON_ID,
    )
