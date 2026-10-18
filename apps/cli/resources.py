"""独立下载 ASR 模型，以及保留检测资源的准备入口；不启动识别。"""

import argparse
import json
import sys
from pathlib import Path

from speechio.errors import ASRError
from speechio.models import download_model
from speechio.runtime.models.files import MODEL_FILES, check_model_files
from speechio.runtime.models.manifest import verify_manifest
from speechio.runtime.models.pinned import DETECTION_MODELS
from speechio.runtime.models.prepare import prepare_model

KINDS = {
    "whisper": "whisper",
    "nemotron": "nemotron",
    "vad": "silero-vad",
    "openwakeword": "openwakeword",
}


def parser():
    command = argparse.ArgumentParser(prog="asr models")
    actions = command.add_subparsers(dest="action", required=True)
    actions.add_parser("list", help="列出支持的模型类型，不扫描硬盘或加载模型")
    download = actions.add_parser("download", help="下载指定 ASR 模型到明确目录")
    download.add_argument("--engine", choices=tuple(MODEL_FILES), required=True)
    download.add_argument("--model-id", required=True)
    download.add_argument("--destination", type=Path, required=True)
    prepare = actions.add_parser("prepare", help="准备检测资源，或检查已有 ASR 模型目录")
    prepare.add_argument("--path", type=Path, required=True)
    prepare.add_argument("--download", action="store_true")
    prepare.add_argument("--kind", choices=tuple(KINDS), required=True)
    prepare.add_argument("--model-id")
    prepare.add_argument("--local-source", type=Path)
    verify = actions.add_parser("verify", help="检查 ASR 必需文件；检测资源保留原有校验")
    verify.add_argument("--path", type=Path, required=True)
    verify.add_argument("--engine", choices=tuple(MODEL_FILES))
    return command


def model_catalog():
    """仅描述适配器文件要求，不把一份模型的固定哈希作为支持边界。"""
    result = {
        engine: {"backend": engine, "files": list(files)} for engine, files in MODEL_FILES.items()
    }
    for kind in ("vad", "openwakeword"):
        model = DETECTION_MODELS[KINDS[kind]]
        result[kind] = {
            "backend": KINDS[kind],
            "id": model["id"],
            "revision": model["revision"],
            "files": list(model["files"]),
        }
    return result


def _execute(args):
    """命令只映射参数；下载和加载所需文件检查由核心包完成。"""
    if args.action == "list":
        return {"resources": model_catalog()}
    if args.action == "download":
        path = download_model(
            engine=args.engine, model_id=args.model_id, destination=args.destination
        )
        return {"model_path": str(path)}
    if args.action == "verify":
        engine = args.engine
        if engine is None:
            try:
                engine = json.loads(
                    (args.path / "model-manifest.json").read_text(encoding="utf-8")
                )["backend"]
            except (OSError, ValueError, KeyError) as exc:
                raise ASRError(
                    "INVALID_CONFIGURATION", "verify", "ASR 目录没有类型记录，请显式填写 --engine"
                ) from exc
        if engine in MODEL_FILES:
            return {"model_path": str(check_model_files(engine, args.path)), "engine": engine}
        return {"manifest": verify_manifest(args.path)}
    backend = KINDS[args.kind]
    if backend in MODEL_FILES:
        if args.local_source is not None:
            raise ASRError(
                "INVALID_CONFIGURATION", "prepare", "已有 ASR 模型直接通过 model_path 加载"
            )
        if args.download:
            if args.model_id is None:
                raise ASRError(
                    "INVALID_CONFIGURATION", "prepare", "下载 ASR 模型必须提供 --model-id"
                )
            path = download_model(engine=backend, model_id=args.model_id, destination=args.path)
        else:
            path = check_model_files(backend, args.path)
        return {"model_path": str(path), "engine": backend}
    return {
        "manifest": prepare_model(
            backend, args.path, allow_download=args.download, local_source=args.local_source
        )
    }


def main(argv=None, *, stdout=None):
    """资源命令使用 JSON 输出，错误有明确分类和退出码。"""
    output = stdout or sys.stdout
    if stdout is None and hasattr(output, "reconfigure"):
        output.reconfigure(encoding="utf-8")
    args = parser().parse_args(argv)
    try:
        value = {"operation": args.action, "status": "ok", **_execute(args)}
        code = 0
    except Exception as exc:
        error = (
            exc
            if isinstance(exc, ASRError)
            else ASRError("RESOURCE_PREPARE_FAILED", args.action, str(exc))
        )
        value = {
            "operation": error.operation,
            "status": "failed",
            "code": error.code,
            "message": error.message,
        }
        code = 1
    output.write(json.dumps(value, ensure_ascii=False) + "\n")
    return code
