"""项目外 -I 运行，验证实际安装包和无 audio 依赖的实时接口。"""

import argparse
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
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--audio", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    before = {t.ident for t in threading.enumerate()}
    import realtime_asr as asr

    module = Path(asr.__file__).resolve()
    optional = ("faster_whisper", "sherpa_onnx", "onnxruntime", "soxr", "soundfile", "pyaudio")
    if "site-packages" not in module.parts or any(name in sys.modules for name in optional):
        raise AssertionError("独立根导入路径或可选库边界错误")
    if any(importlib.util.find_spec(name) for name in ("soxr", "soundfile", "pyaudio")):
        raise AssertionError("无 audio 验收环境安装了音频依赖")
    runtime = asr.ASRRuntime(
        asr.ModelConfig(model=args.model),
        realtime=asr.RealtimeConfig(initial_delay=0, interval=10),
    )
    if {t.ident for t in threading.enumerate()} != before:
        raise AssertionError("未准备运行时启动了线程")
    report = {
        "stage": args.stage,
        "cwd": str(Path.cwd()),
        "installed_module": str(module),
        "events": [],
    }
    try:
        if args.stage == "base":
            try:
                runtime.prepare()
            except asr.ASRError as exc:
                if exc.code != "DEPENDENCY_MISSING":
                    raise
                report["missing_model_dependency"] = exc.code
            else:
                raise AssertionError("基础包意外加载了模型")
        else:
            saved = json.loads(args.baseline.read_text(encoding="utf-8"))
            expected = next(
                e["result"]["text"]
                for b in saved["backends"]
                if b["backend"] == "whisper"
                for s in b["sessions"]
                if s["sample_id"] == "C07-1"
                for e in s["events"]
                if e["kind"] == "final"
            )
            with wave.open(str(args.audio)) as source:
                pcm = source.readframes(source.getnframes())
            runtime.prepare()
            session = runtime.create_session(asr.SessionConfig(realtime=True, language="en"))
            session.start()
            session.feed_audio(pcm[:8192])
            observation = session.next_event(timeout=30)
            if observation is None or (observation.kind, observation.source) != (
                "realtime",
                "snapshot",
            ):
                raise AssertionError("独立安装没有实时观察")
            report["events"].append(asdict(observation))
            for offset in range(8192, len(pcm), 1001):
                session.feed_audio(pcm[offset : offset + 1001])
            task = session.end_turn()
            final, completed = session.next_event(timeout=30), session.next_event(timeout=30)
            if (
                (final.kind, completed.kind) != ("final", "completed")
                or final.result.text != expected
                or final.turn != task.turn
            ):
                raise AssertionError("独立安装最终事件与基线不同")
            report["events"].extend((asdict(final), asdict(completed)))
            report["same_p2_text"] = True
            session.close()
    finally:
        report["close"] = asdict(runtime.close(timeout=30))
    if report["close"]["status"] != "ok":
        raise AssertionError("独立安装关闭失败")
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{args.stage}: P3 独立安装验证通过", flush=True)


if __name__ == "__main__":
    main()
