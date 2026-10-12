"""显式启动真实 P3 验收，不启动麦克风，不执行正式 CER／WER。"""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

from evals.p3.runner import check_configuration, configurations


def main():
    parser = argparse.ArgumentParser(description="P3 真实实时识别验收")
    parser.add_argument("--project", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/p3/realtime-check/summary.json")
    )
    parser.add_argument("--configuration", action="append")
    args = parser.parse_args()
    project = args.project.resolve()
    baseline_path = project / "artifacts/p2/session-check/summary.json"
    saved = json.loads(baseline_path.read_text(encoding="utf-8"))
    baseline = {
        (backend["backend"], session["sample_id"]): next(
            e["result"]["text"] for e in session["events"] if e["kind"] == "final"
        )
        for backend in saved["backends"]
        for session in backend["sessions"]
    }
    choices = configurations(project)
    selected = args.configuration or list(choices)
    report = {
        "python": sys.version,
        "platform": platform.platform(),
        "p2_baseline_sha256": hashlib.sha256(baseline_path.read_bytes()).hexdigest(),
        "p0_formal_acceptance": "pending_human_review",
        "configurations": [],
        "scope": "paced_1x_cn_en_continuous_and_30s_repeat_per_configuration",
        "versions": {
            name: importlib.metadata.version(name)
            for name in (
                "numpy",
                "faster-whisper",
                "ctranslate2",
                "sherpa-onnx",
                "onnxruntime",
                "psutil",
                "soxr",
            )
        },
        "model_records": {},
    }
    for name in ("whisper-tiny", "nemotron"):
        manifest_path = project / ".runtime/models" / name / "model-manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        report["model_records"][name] = {
            key: manifest[key] for key in ("model_record_id", "model_id", "backend", "revision")
        }
        report["model_records"][name]["manifest_sha256"] = hashlib.sha256(
            manifest_path.read_bytes()
        ).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        for name in selected:
            config, realtime, backend = choices[name]
            result = check_configuration(name, config, realtime, backend, project, baseline, 30)
            report["configurations"].append(result)
            print(f"{name}: 5 个单元，中文、英文连续输入及 30 秒长输入通过", flush=True)
        report["status"] = "ok"
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = str(exc)
        raise
    finally:
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
