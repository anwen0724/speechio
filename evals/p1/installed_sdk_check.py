"""在独立 wheel 环境中直接执行；不导入项目评估辅助对象。"""

import argparse
import importlib.metadata
import importlib.util
import json
import sys
import wave
from dataclasses import asdict
from pathlib import Path

import numpy as np

import realtime_asr
from realtime_asr import ASRError, AudioConverter, AudioFormat, Recognizer, RecognizerConfig


def expect_error(code, call):
    """记录所选路径的真实失败，不用异常全文判断行为。"""
    try:
        call()
    except ASRError as exc:
        if exc.code != code:
            raise AssertionError(f"预期 {code}，实际 {exc.code}") from exc
        return {"code": exc.code, "operation": exc.operation}
    raise AssertionError(f"预期失败 {code}，却正常返回")


def main():
    """先验收基础安装，再验收不安装 audio 组的 Whisper 标准音频。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=("base", "whisper"))
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--audio", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    module_path = Path(realtime_asr.__file__).resolve()
    if "site-packages" not in module_path.parts:
        raise AssertionError("没有使用独立安装的 wheel")
    optional = ("soundfile", "soxr", "faster_whisper", "sherpa_onnx", "onnxruntime", "pyaudio")
    loaded = [name for name in optional if name in sys.modules]
    if loaded:
        raise AssertionError(f"核心导入加载了可选运行库：{loaded}")
    if any(importlib.util.find_spec(name) is not None for name in ("soundfile", "soxr", "pyaudio")):
        raise AssertionError("此检查环境应没有 audio 或麦克风依赖")
    converter = AudioConverter(AudioFormat())
    frames = converter.feed(b"\x01\x00\xff\xff") + converter.finish()
    converter.close()
    if b"".join(f.pcm for f in frames) != b"\x01\x00\xff\xff":
        raise AssertionError("基础包标准 PCM 路径没有保留原值")
    report = {
        "stage": args.stage,
        "python": sys.executable,
        "cwd": str(Path.cwd()),
        "installed_module": str(module_path),
        "optional_loaded_by_import": loaded,
        "versions": {p: importlib.metadata.version(p) for p in ("numpy", "realtime-asr")},
    }
    converter = AudioConverter(AudioFormat(48000))
    try:
        report["missing_soxr"] = expect_error(
            "DEPENDENCY_MISSING", lambda: converter.feed(b"\x00\x00")
        )
    finally:
        converter.close()
    recognizer = Recognizer(RecognizerConfig(model=args.model))
    try:
        report["missing_soundfile"] = expect_error(
            "DEPENDENCY_MISSING", lambda: recognizer.transcribe_file("missing.wav")
        )
        if args.stage == "base":
            report["missing_whisper"] = expect_error("DEPENDENCY_MISSING", recognizer.prepare)
        else:
            with wave.open(str(args.audio)) as source:
                raw = source.readframes(source.getnframes())
            result = recognizer.transcribe(
                np.frombuffer(raw, dtype="<i2"), format=AudioFormat(), language="en"
            )
            report["real_result"] = asdict(result)
    finally:
        report["close"] = asdict(recognizer.close())
    if report["close"]["status"] != "ok":
        raise AssertionError("独立运行时关闭失败")
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
