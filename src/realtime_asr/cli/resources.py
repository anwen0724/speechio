"""离线运行资源的明确准备入口，不加载识别或检测模型。"""

import argparse
import json
import os
import sys
from pathlib import Path

from realtime_asr.core.errors import ASRError
from realtime_asr.core.runtime.models.manifest import WHISPER_FILES, verify_manifest
from realtime_asr.core.runtime.models.pinned import DETECTION_MODELS, WHISPER_MODEL, nemotron_model
from realtime_asr.core.runtime.models.prepare import prepare_model
from realtime_asr.core.runtime.native.gpu import CUDNN_VERSION, prepare_cudnn, verify_runtime

KINDS = {
    "whisper": "whisper",
    "nemotron": "nemotron",
    "vad": "silero-vad",
    "openwakeword": "openwakeword",
}


def parser(gpu):
    command = argparse.ArgumentParser(prog="asr gpu-runtime" if gpu else "asr models")
    actions = command.add_subparsers(dest="action", required=True)
    if not gpu:
        actions.add_parser("list", help="列出准备资源及固定身份，不加载模型")
    prepare = actions.add_parser("prepare", help="明确准备或离线复用资源")
    prepare.add_argument("--root" if gpu else "--path", type=Path, required=True)
    prepare.add_argument("--download", action="store_true")
    if not gpu:
        prepare.add_argument("--kind", choices=tuple(KINDS), required=True)
        prepare.add_argument("--variant", choices=("560ms", "1120ms", "80ms"))
        prepare.add_argument("--local-snapshot", type=Path)
        prepare.add_argument("--local-source", type=Path)
    verify = actions.add_parser("verify", help="只检查文件与清单")
    verify.add_argument("--path", type=Path, required=True)
    return command


def model_catalog():
    """固定准备身份来自同一份数据，目录里出现其他文件不增加支持声明。"""
    result = {
        "whisper": {
            "backend": "whisper",
            **WHISPER_MODEL,
            "files": list(WHISPER_FILES),
            "download_extra": "models",
        },
        "nemotron": {
            "backend": "nemotron",
            "variants": {
                variant: nemotron_model(variant)["id"] for variant in ("560ms", "1120ms", "80ms")
            },
            "download_extra": "models",
        },
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


def gpu_resource(args):
    """默认离线校验；联网获取与实际 GPU 推理分别执行。"""
    path = args.root / "cudnn" / CUDNN_VERSION if args.action == "prepare" else args.path
    manifest_path = path / "native-manifest.json"
    if args.action == "prepare" and not manifest_path.is_file():
        if not args.download:
            raise ASRError(
                "DEPENDENCY_MISSING", "prepare_gpu", "缺少已准备库，联网需明确 --download"
            )
        if os.name != "nt":
            raise ASRError("BACKEND_INCOMPATIBLE", "prepare_gpu", "当前准备入口只支持 Windows")
        path = prepare_cudnn(args.root)
        manifest_path = path / "native-manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ASRError("DEPENDENCY_MISSING", "verify_gpu", "原生库清单缺失或无效") from exc
    verify_runtime(path, manifest)
    return {"path": str(path.resolve()), "manifest": manifest, "model_inference_verified": False}


def main(argv=None, *, gpu=False, stdout=None):
    """资源命令使用 JSON 输出，错误有稳定分类和明确退出码。"""
    output = stdout or sys.stdout
    if stdout is None and hasattr(output, "reconfigure"):
        output.reconfigure(encoding="utf-8")
    args = parser(gpu).parse_args(argv)
    try:
        if gpu:
            result = {"resource": gpu_resource(args)}
        elif args.action == "list":
            result = {"resources": model_catalog()}
        else:
            result = {
                "manifest": verify_manifest(args.path)
                if args.action == "verify"
                else prepare_model(
                    KINDS[args.kind],
                    args.path,
                    allow_download=args.download,
                    variant=args.variant,
                    local_snapshot=args.local_snapshot,
                    local_source=args.local_source,
                )
            }
        value = {"operation": args.action, "status": "ok", **result}
        code = 0
    except Exception as exc:
        error = (
            exc
            if isinstance(exc, ASRError)
            else ASRError(
                "OPTIONAL_DEPENDENCY_MISSING"
                if isinstance(exc, ImportError)
                else "RESOURCE_PREPARE_FAILED",
                args.action,
                str(exc),
            )
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
