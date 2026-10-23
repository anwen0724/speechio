"""显式获取模型；推理模块只接收已经校验的本地目录。"""

import json
import shutil
import tarfile
import zipfile
from os import PathLike
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import urlopen

from speechio.errors import STTError, TTSError
from speechio.models.files import TTS_REPOSITORIES, check_model_files, check_tts_model_files
from speechio.models.manifest import (
    MANIFEST_NAME,
    check_file,
    create_manifest,
    verify_manifest,
)
from speechio.models.pinned import DETECTION_MODELS, nemotron_model


def prepare_model(
    backend, folder, *, allow_download=False, variant=None, local_snapshot=None, local_source=None
):
    """统一准备本地资源；身份、缓存和文件规则不进入 CLI 或会话。"""
    if (
        (variant is not None and backend != "nemotron")
        or (local_snapshot is not None and backend != "whisper")
        or (local_source is not None and backend not in DETECTION_MODELS)
    ):
        raise STTError("INVALID_CONFIGURATION", "prepare_model", "准备参数不属于所选资源")
    if backend == "whisper":
        return prepare_whisper(folder, allow_download=allow_download, local_snapshot=local_snapshot)
    if backend == "nemotron":
        return prepare_nemotron(folder, allow_download=allow_download, variant=variant or "560ms")
    if backend not in DETECTION_MODELS:
        raise STTError("BACKEND_INCOMPATIBLE", "prepare_model", "资源种类不受支持")
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
            raise STTError("MODEL_INTEGRITY", "prepare_model", "已有清单不是所选资源")
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
            raise STTError(
                "MODEL_INCOMPLETE", "prepare_model", "缺少固定文件，提供本地来源或明确 --download"
            )
    return create_manifest(
        folder,
        backend=backend,
        model_id=model["id"],
        source=model["source"],
        revision=model["revision"],
    )


def prepare_whisper(folder, *, allow_download=False, local_snapshot=None, model_id="tiny"):
    """旧资源入口只委托显式下载或检查已有目录，不再生成哈希。"""
    if local_snapshot is not None:
        raise STTError(
            "INVALID_CONFIGURATION", "prepare_model", "已有模型请直接通过 model_path 加载"
        )
    path = (
        download_model_files("whisper", model_id, folder)
        if allow_download
        else check_model_files("whisper", folder)
    )
    return {"backend": "whisper", "model_id": model_id, "model_path": str(path)}


def prepare_nemotron(folder, *, allow_download=False, variant="560ms"):
    """兼容已有安装命令；下载与本地文件检查都不执行模型哈希校验。"""
    path = (
        download_model_files("nemotron", variant, folder)
        if allow_download
        else check_model_files("nemotron", folder)
    )
    return {
        "backend": "nemotron",
        "model_id": nemotron_model(variant)["id"],
        "model_path": str(path),
    }


def _download_source(engine, model_id):
    """名称仅用于下载；同一来源的文件始终保存到明确的目标目录。"""
    if not isinstance(model_id, str) or not model_id.strip():
        raise STTError("INVALID_CONFIGURATION", "download_model", "model_id 必须是非空模型标识")
    if engine == "nemotron":
        identity = nemotron_model(model_id)["id"]
        return (
            identity,
            f"https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/{identity}.tar.bz2",
        )
    if engine != "whisper":
        raise STTError("BACKEND_INCOMPATIBLE", "download_model", "下载只支持 whisper 和 nemotron")
    names = (
        "tiny",
        "tiny.en",
        "base",
        "base.en",
        "small",
        "small.en",
        "medium",
        "medium.en",
        "large-v1",
        "large-v2",
        "large-v3",
    )
    sources = {name: f"Systran/faster-whisper-{name}" for name in names}
    sources["large-v3-turbo"] = "mobiuslabsgmbh/faster-whisper-large-v3-turbo"
    sources["turbo"] = sources["large-v3-turbo"]
    sources["large"] = sources["large-v3"]
    if model_id in sources:
        return model_id, sources[model_id]
    if model_id.count("/") == 1 and all(model_id.split("/")):
        return model_id, model_id
    raise STTError(
        "INVALID_CONFIGURATION",
        "download_model",
        "未知 Whisper 模型；请使用已知名称或提供组织名/仓库名",
    )


def download_model_files(engine, model_id, destination):
    """显式交付本地模型目录；临时下载失败不改变目标，不计算文件哈希。"""
    is_tts = engine in TTS_REPOSITORIES
    error = TTSError if is_tts else STTError
    check_files = check_tts_model_files if is_tts else check_model_files
    if is_tts:
        if model_id not in TTS_REPOSITORIES[engine]:
            raise error("INVALID_CONFIGURATION", "download_model", "该适配器不支持所选模型标识")
        identity = source = model_id
    else:
        identity, source = _download_source(engine, model_id)
    if not isinstance(destination, (str, PathLike)) or not str(destination).strip():
        raise error("INVALID_CONFIGURATION", "download_model", "destination 必须是模型保存目录")
    folder = Path(destination).expanduser().absolute()
    if folder.is_symlink():
        raise error("MODEL_CONFLICT", "download_model", f"目标目录不能是符号链接：{folder}")
    folder = folder.resolve()
    if folder.exists():
        record = folder / MANIFEST_NAME
        if record.is_file():
            try:
                saved = json.loads(record.read_text(encoding="utf-8"))
            except (ValueError, OSError) as exc:
                raise error(
                    "MODEL_CONFLICT", "download_model", f"无法确认已有模型身份：{folder}"
                ) from exc
            if saved.get("backend") == engine and saved.get("source") == source:
                return check_files(engine, folder)
        if not folder.is_dir() or any(folder.iterdir()):
            raise error(
                "MODEL_CONFLICT",
                "download_model",
                f"目标已有文件，无法确认是所选模型，未覆盖：{folder}",
            )
    folder.parent.mkdir(parents=True, exist_ok=True)
    try:
        with TemporaryDirectory(prefix=f".{folder.name}.download-", dir=folder.parent) as temporary:
            staging = Path(temporary)
            payload = staging / "model"
            payload.mkdir()
            if engine == "whisper" or is_tts:
                from huggingface_hub import HfApi, snapshot_download

                revision = HfApi().model_info(source).sha
                if not revision:
                    raise error("MODEL_DOWNLOAD", "download_model", "下载来源未返回具体版本")
                snapshot_download(
                    repo_id=source,
                    revision=revision,
                    local_dir=payload,
                    allow_patterns=(
                        ["*.json", "*.safetensors", "*.txt"]
                        if engine == "qwen_tts"
                        else ["config.json", "kokoro-v1_1-zh.pth", "voices/*.pt"]
                        if engine == "kokoro"
                        else ["*.json", "model.bin", "vocabulary.txt"]
                    ),
                )
            else:
                revision = identity
                archive = staging / "model.tar.bz2"
                with urlopen(source, timeout=30) as response, archive.open("wb") as output:
                    shutil.copyfileobj(response, output)
                    length = response.headers.get("Content-Length")
                if length is not None and archive.stat().st_size != int(length):
                    raise error(
                        "MODEL_DOWNLOAD", "download_model", "下载未完成，文件长度与响应不符"
                    )
                with tarfile.open(archive, "r:bz2") as package:
                    package.extractall(payload, filter="data")
                if (payload / identity).is_dir():
                    payload = payload / identity
            if engine == "kokoro":
                _download_english_resources(payload, staging)
            check_files(engine, payload)
            # 只记录下载身份，供重复下载确认目录；加载不依赖这份记录。
            record = {
                "backend": engine,
                "model_id": identity,
                "source": source,
                "revision": revision,
            }
            (payload / MANIFEST_NAME).write_text(
                json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            if folder.exists():
                folder.rmdir()  # 只允许移除空目录；下载期间出现的新文件也不会被覆盖。
            payload.rename(folder)
        return folder
    except (STTError, TTSError):
        raise
    except Exception as exc:
        raise error(
            "MODEL_DOWNLOAD", "download_model", f"模型下载失败，目标文件未覆盖：{exc}"
        ) from exc


def _download_english_resources(payload, staging):
    """Kokoro 的英文分词资源也在显式下载阶段交付；运行时不安装包。"""
    name = "en_core_web_sm-3.8.0"
    source = f"https://github.com/explosion/spacy-models/releases/download/{name}/{name}-py3-none-any.whl"
    archive = staging / "english.whl"
    with urlopen(source, timeout=60) as response, archive.open("wb") as output:
        shutil.copyfileobj(response, output)
    prefix = f"en_core_web_sm/{name}/"
    destination = payload / "english"
    with zipfile.ZipFile(archive) as package:
        for entry in package.infolist():
            if not entry.filename.startswith(prefix) or entry.is_dir():
                continue
            target = (destination / entry.filename[len(prefix) :]).resolve()
            if not target.is_relative_to(destination.resolve()):
                raise TTSError("MODEL_DOWNLOAD", "download_model", "语言资源路径无效")
            target.parent.mkdir(parents=True, exist_ok=True)
            with package.open(entry) as source_file, target.open("wb") as output:
                shutil.copyfileobj(source_file, output)
    if not (destination / "config.cfg").is_file():
        raise TTSError("MODEL_DOWNLOAD", "download_model", "下载的英文资源不完整")
