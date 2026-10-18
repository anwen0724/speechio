"""在项目外用 -I 运行，验证安装包公开会话接口及依赖边界。"""

import argparse
import importlib.metadata
import importlib.util
import json
import sys
import threading
import wave
from dataclasses import asdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("base", "whisper"), required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--audio", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    before = {t.ident for t in threading.enumerate()}
    import speechio as asr

    optional = ("soundfile", "soxr", "faster_whisper", "sherpa_onnx", "onnxruntime", "pyaudio")
    loaded = [name for name in optional if name in sys.modules]
    module_path = Path(asr.__file__).resolve()
    if "site-packages" not in module_path.parts or loaded:
        raise AssertionError("没有使用无可选库副作用的独立安装包")
    if {t.ident for t in threading.enumerate()} != before:
        raise AssertionError("根导入创建了线程")
    if any(importlib.util.find_spec(name) is not None for name in ("soundfile", "soxr", "pyaudio")):
        raise AssertionError("标准音频验收环境不应安装 audio 或麦克风依赖")
    report = {
        "stage": args.stage,
        "python": sys.executable,
        "cwd": str(Path.cwd()),
        "installed_module": str(module_path),
        "optional_loaded_by_import": loaded,
        "import_created_threads": False,
        "versions": {p: importlib.metadata.version(p) for p in ("numpy", "speechio")},
    }
    runtime = asr.ASRRuntime(asr.ModelConfig(engine="whisper", model_path=args.model_path))
    if {t.ident for t in threading.enumerate()} != before:
        raise AssertionError("创建未准备运行时创建了线程")
    try:
        if args.stage == "base":
            try:
                runtime.prepare()
            except asr.ASRError as exc:
                if exc.code != "DEPENDENCY_MISSING":
                    raise
                report["missing_model_dependency"] = exc.code
            else:
                raise AssertionError("基础环境意外具有模型依赖")
        else:
            saved = json.loads(args.baseline.read_text(encoding="utf-8"))
            expected = next(
                check["result"]["text"]
                for backend in saved["backends"]
                if backend["backend"] == "whisper"
                for check in backend["checks"]
                if check["sample_id"] == "C07-1" and check["entry"] == "array"
            )
            with wave.open(str(args.audio)) as source:
                pcm = source.readframes(source.getnframes())
            runtime.prepare()
            session = runtime.create_session(asr.SessionConfig(language="en"))
            tasks = []
            for _ in range(2):
                session.start()
                for offset in range(0, len(pcm), 1001):
                    session.feed_audio(pcm[offset : offset + 1001])
                tasks.append(session.end_turn())
            events = [session.next_event(timeout=30) for _ in range(4)]
            for index, task in enumerate(tasks):
                final, completed = events[index * 2 : index * 2 + 2]
                if final is None or completed is None:
                    raise AssertionError("缺少独立安装事件")
                if (final.kind, completed.kind) != ("final", "completed"):
                    raise AssertionError("独立安装事件顺序错误")
                if final.turn != task.turn or final.result.text != expected:
                    raise AssertionError("独立安装归属或正文错误")
            report["events"] = [asdict(event) for event in events]
            report["same_p1_text"] = True
            report["session_close"] = asdict(session.close())
    finally:
        report["close"] = asdict(runtime.close(timeout=30))
    if report["close"]["status"] != "ok":
        raise AssertionError("独立安装运行时关闭失败")
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{args.stage}: 独立安装验收通过")


if __name__ == "__main__":
    main()
