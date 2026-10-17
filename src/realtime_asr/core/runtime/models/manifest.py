"""保存不可变模型身份及文件完整性。"""

import hashlib
import json
from pathlib import Path
from uuid import uuid4

from realtime_asr.core.errors import ASRError
from realtime_asr.core.runtime.models.pinned import DETECTION_MODELS, nemotron_model

MANIFEST_NAME = "model-manifest.json"
WHISPER_FILES = {name: None for name in ("model.bin", "config.json", "tokenizer.json")}


def required_files(backend, variant=None):
    """同一份固定文件规则用于建立清单和离线校验。"""
    if backend == "whisper":
        return WHISPER_FILES
    if backend == "nemotron":
        return nemotron_model(variant or "560ms")["files"]
    if backend in DETECTION_MODELS:
        return DETECTION_MODELS[backend]["files"]
    raise ASRError("BACKEND_INCOMPATIBLE", "verify_model", "资源清单种类不受支持")


def sha256_file(path):
    """分块计算文件哈希，不把大型模型复制到内存。"""
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def check_file(path, size, digest):
    """加载前检查文件，区分缺失与完整性错误。"""
    path = Path(path)
    if not path.is_file():
        raise ASRError("MODEL_INCOMPLETE", "verify_model", f"缺少文件：{path}")
    actual_size, actual_hash = path.stat().st_size, sha256_file(path)
    if actual_size != size or actual_hash != digest:
        raise ASRError("MODEL_INTEGRITY", "verify_model", f"文件校验不符：{path}")
    return actual_size, actual_hash


def create_manifest(folder, *, backend, model_id, source, revision, variant=None):
    """首次准备后建立模型记录；已有清单不可被当前文件覆盖。"""
    folder = Path(folder).resolve()
    if (folder / MANIFEST_NAME).exists():
        manifest = verify_manifest(folder)
        if (manifest["backend"], manifest["model_id"], manifest["revision"]) != (
            backend,
            model_id,
            revision,
        ):
            raise ASRError("MODEL_INTEGRITY", "prepare_model", "已有清单的模型身份不同")
        return manifest
    required = required_files(backend, variant)
    for name, expected in required.items():
        path = folder / name
        if not path.is_file():
            raise ASRError("MODEL_INCOMPLETE", "prepare_model", f"缺少文件：{path}")
        if expected:
            check_file(path, *expected)
    files = []
    for path in sorted(folder.rglob("*")):
        if (
            not path.is_file()
            or ".cache" in path.relative_to(folder).parts
            or path.name.endswith(".partial")
        ):
            continue
        size, digest = path.stat().st_size, sha256_file(path)
        files.append(
            {
                "path": path.relative_to(folder).as_posix(),
                "size_bytes": size,
                "sha256": digest,
                "expected_size_bytes": size,
                "expected_sha256": digest,
                "verified": True,
            }
        )
    manifest = {
        "model_record_id": uuid4().hex,
        "backend": backend,
        "model_id": model_id,
        "source": source,
        "revision": revision,
        "model_path": str(folder),
        "files": files,
    }
    if backend == "nemotron" and variant is not None:
        manifest["variant"] = variant
    with (folder / MANIFEST_NAME).open("x", encoding="utf-8") as output:
        json.dump(manifest, output, ensure_ascii=False, indent=2)
    return manifest


def verify_manifest(folder):
    """离线核对已准备模型的逐文件大小和 SHA-256。"""
    folder = Path(folder).resolve()
    try:
        manifest = json.loads((folder / MANIFEST_NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ASRError("MODEL_INCOMPLETE", "verify_model", "模型清单缺失或无法读取") from exc
    if manifest.get("backend") not in ("whisper", "nemotron", *DETECTION_MODELS):
        raise ASRError("BACKEND_INCOMPATIBLE", "verify_model", "模型清单后端不受支持")
    if manifest["backend"] == "nemotron":
        variant = manifest.get("variant", "560ms")
        model = nemotron_model(variant)
        if variant != "560ms" and manifest["model_id"] != model["id"]:
            raise ASRError("MODEL_INTEGRITY", "verify_model", "Nemotron 变体与模型身份不一致")
        required = model["files"]
    else:
        required = required_files(manifest["backend"])
        if manifest["backend"] in DETECTION_MODELS:
            model = DETECTION_MODELS[manifest["backend"]]
            if (manifest["model_id"], manifest["revision"]) != (model["id"], model["revision"]):
                raise ASRError("MODEL_INTEGRITY", "verify_model", "固定检测资源身份不一致")
    declared = {item["path"]: item for item in manifest.get("files", [])}
    if not required.keys() <= declared.keys():
        raise ASRError("MODEL_INCOMPLETE", "verify_model", "清单遗漏必需模型文件")
    for name, expected in required.items():
        if (
            expected
            and (declared[name]["expected_size_bytes"], declared[name]["expected_sha256"])
            != expected
        ):
            raise ASRError("MODEL_INTEGRITY", "verify_model", f"固定模型校验值不符：{name}")
    for item in manifest["files"]:
        path = (folder / item["path"]).resolve()
        if not path.is_relative_to(folder):
            raise ASRError("MODEL_INTEGRITY", "verify_model", "模型清单路径越界")
        check_file(path, item["expected_size_bytes"], item["expected_sha256"])
    manifest["model_path"] = str(folder)
    return manifest
