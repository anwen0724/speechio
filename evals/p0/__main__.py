"""P0 开发验证命令；不是最终产品 CLI。"""

import argparse
import json
from pathlib import Path

from realtime_asr.core.errors import ASRError


def parser():
    """解析显式准备和运行操作，不在帮助或导入时初始化模型。"""
    root = argparse.ArgumentParser(description="P0 本地 ASR 运行与语料基线")
    root.add_argument("--runtime-root", type=Path, default=Path(".runtime"))
    commands = root.add_subparsers(dest="command", required=True)
    commands.add_parser("environment")
    prepare = commands.add_parser("prepare-models")
    prepare.add_argument("--allow-download", action="store_true")
    prepare.add_argument("--local-whisper-snapshot", type=Path)
    commands.add_parser("prepare-gpu")
    corpus = commands.add_parser("prepare-corpus")
    corpus.add_argument("--folder", type=Path, default=Path("data/corpus/p0"))
    validate = commands.add_parser("validate-corpus")
    validate.add_argument("--manifest", type=Path, default=Path("data/corpus/p0/manifest.json"))
    validate.add_argument("--exploratory", action="store_true")
    run = commands.add_parser("run")
    run.add_argument("--manifest", type=Path, default=Path("data/corpus/p0/manifest.json"))
    run.add_argument("--output", type=Path, default=Path("artifacts/p0"))
    run.add_argument("--exploratory", action="store_true")
    run.add_argument("--matrix", help="逗号分隔，例如 M01,M03；未选择项记录 skipped")
    report = commands.add_parser("report")
    report.add_argument("folder", type=Path, help="已有运行目录，只读取保存的机器记录")
    return root


def main():
    """边界错误返回非零退出码，模型运行不触发准备下载。"""
    arguments = parser().parse_args()
    try:
        match arguments.command:
            case "report":
                from evals.p0.saved_report import rebuild_report

                result = rebuild_report(arguments.folder)
            case "environment":
                from evals.p0.diagnostics.environment import inspect_environment

                result = inspect_environment()
            case "prepare-models":
                from realtime_asr.core.runtime.models.prepare import (
                    prepare_nemotron,
                    prepare_whisper,
                )

                result = [
                    prepare_whisper(
                        arguments.runtime_root / "models/whisper-tiny",
                        allow_download=arguments.allow_download,
                        local_snapshot=arguments.local_whisper_snapshot,
                    ),
                    prepare_nemotron(
                        arguments.runtime_root / "models/nemotron",
                        allow_download=arguments.allow_download,
                    ),
                ]
            case "prepare-gpu":
                from realtime_asr.core.runtime.native.gpu import prepare_cudnn

                result = {"folder": str(prepare_cudnn(arguments.runtime_root))}
            case "prepare-corpus":
                from evals.p0.corpus.public import prepare_public_corpus

                result = {"manifest": str(prepare_public_corpus(arguments.folder))}
            case "validate-corpus":
                from evals.p0.corpus.manifest import load_corpus

                corpus, _ = load_corpus(arguments.manifest, allow_unreviewed=arguments.exploratory)
                result = {
                    "corpus_version": corpus["corpus_version"],
                    "samples": len(corpus["samples"]),
                    "annotation_checked": all(
                        item["annotation_checked"] for item in corpus["samples"]
                    ),
                }
            case "run":
                from evals.p0.runner import run_matrix

                selected = set(arguments.matrix.split(",")) if arguments.matrix else None
                if selected and not selected <= {f"M{index:02}" for index in range(1, 11)}:
                    raise ValueError("矩阵编号应为 M01 至 M10")
                result = run_matrix(
                    arguments.manifest,
                    arguments.runtime_root,
                    arguments.output,
                    exploratory=arguments.exploratory,
                    selected=selected,
                )
                return 0 if result["status"].startswith("passed") else 2
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (ASRError, OSError, ValueError) as exc:
        from realtime_asr.core.errors import error_record

        print(json.dumps(error_record(exc, arguments.command), ensure_ascii=False, indent=2))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
