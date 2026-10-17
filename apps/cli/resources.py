"""离线模型资源的明确准备入口，不加载识别或检测模型。"""

import argparse
import json
import sys
from pathlib import Path

from speechio.errors import ASRError
from speechio.runtime.models.manifest import WHISPER_FILES, verify_manifest
from speechio.runtime.models.pinned import DETECTION_MODELS, WHISPER_MODEL
from speechio.runtime.models.prepare import prepare_model

KINDS = {
    "whisper": "whisper",
    "vad": "silero-vad",
    "openwakeword": "openwakeword",
}


def parser():
    command = argparse.ArgumentParser(prog="asr models")
    actions = command.add_subparsers(dest="action", required=True)
    actions.add_parser("list", help="列出准备资源及固定身份，不加载模型")
    prepare = actions.add_parser("prepare", help="明确准备或离线复用资源")
    prepare.add_argument("--path", type=Path, required=True)
    prepare.add_argument("--download", action="store_true")
    prepare.add_argument("--kind", choices=tuple(KINDS), required=True)
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


def main(argv=None, *, stdout=None):
    """资源命令使用 JSON 输出，错误有稳定分类和明确退出码。"""
    output = stdout or sys.stdout
    if stdout is None and hasattr(output, "reconfigure"):
        output.reconfigure(encoding="utf-8")
    args = parser().parse_args(argv)
    try:
        if args.action == "list":
            result = {"resources": model_catalog()}
        else:
            result = {
                "manifest": verify_manifest(args.path)
                if args.action == "verify"
                else prepare_model(
                    KINDS[args.kind],
                    args.path,
                    allow_download=args.download,
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
