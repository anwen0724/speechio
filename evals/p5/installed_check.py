"""项目外的安装检查；不向被测 Python 传入工作区模块路径。"""

import argparse
import json
import os
import subprocess
from pathlib import Path


def run(project, environment, outside, *, mode):
    """基础包检查轻量导入；麦克风组检查实际入口与无 SoundFile 运行。"""
    interpreter = environment / "Scripts/python.exe"
    executable = environment / "Scripts/asr.exe"
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["PYTHONIOENCODING"] = "utf-8"
    report = {"mode": mode, "cwd": str(outside), "commands": []}
    probe = subprocess.run(
        [
            str(interpreter),
            "-I",
            "-c",
            "import json,sys,importlib.util,realtime_asr; "
            "print(json.dumps({'package':realtime_asr.__file__,"
            "'heavy_imports':[m for m in ('pyaudio','soundfile','soxr','faster_whisper',"
            "'sherpa_onnx','onnxruntime') if m in sys.modules],"
            "'soundfile_installed':importlib.util.find_spec('soundfile') is not None,"
            "'exports':realtime_asr.__all__}))",
        ],
        cwd=outside,
        env=env,
        capture_output=True,
        timeout=30,
    )
    if probe.returncode:
        raise AssertionError(probe.stderr.decode("utf-8"))
    report["import"] = json.loads(probe.stdout)
    if report["import"]["heavy_imports"]:
        raise AssertionError("基础导入加载了重依赖")
    if report["import"]["soundfile_installed"] != (mode == "file"):
        raise AssertionError("SoundFile 安装边界不符合本次验证")
    if not Path(report["import"]["package"]).is_relative_to(environment):
        raise AssertionError("包没有来自指定安装环境")
    for prefix in ([str(executable)], [str(interpreter), "-I", "-m", "realtime_asr"]):
        result = subprocess.run(
            [*prefix, "--help"], cwd=outside, env=env, capture_output=True, timeout=15
        )
        if result.returncode or b"devices" not in result.stdout:
            raise AssertionError("帮助入口不可用")
    report["help_entries"] = ["console_script", "isolated_module"]
    if mode == "base":
        return report
    options = ["--model", str(project / ".runtime/models/whisper-tiny"), "--output", "jsonl"]
    with (project / "data/corpus/p0/C01-1.wav").open("rb") as stream:
        import wave

        with wave.open(stream) as audio:
            pcm = audio.readframes(audio.getnframes())
    commands = [
        (["devices", "--output", "jsonl"], None),
        (["stdin", *options, "--language", "zh-CN"], pcm),
        (["mic", *options, "--mode", "manual", "--duration", "1"], None),
        (
            [
                "mic",
                *options,
                "--vad-model",
                str(project / ".runtime/models/vad/silero-v6.onnx"),
                "--duration",
                "1",
            ],
            None,
        ),
    ]
    if mode == "file":
        commands = [
            (
                [
                    "file",
                    str(project / "data/corpus/p0/C01-1.wav"),
                    str(project / "artifacts/p5/real-check/C01-stereo-48k.wav"),
                    *options,
                    "--language",
                    "zh-CN",
                ],
                None,
            )
        ]
    for args, input_bytes in commands:
        result = subprocess.run(
            [str(executable), *args],
            cwd=outside,
            env=env,
            input=input_bytes,
            capture_output=True,
            timeout=90,
        )
        if result.returncode:
            raise AssertionError(result.stderr.decode("utf-8"))
        records = [json.loads(line) for line in result.stdout.decode("utf-8").splitlines()]
        if args[0] == "stdin":
            finals = [r["event"] for r in records if r.get("event", {}).get("kind") == "final"]
            baseline = json.loads(
                (project / "artifacts/p5/real-check/cases.json").read_text("utf-8")
            )
            expected = baseline[0]["events"][-2]["result"]["text"]
            if len(finals) != 1 or finals[0]["audio_samples"] != len(pcm) // 2:
                raise AssertionError("独立 stdin 没有交付完整音频")
            if finals[0]["result"]["text"] != expected:
                raise AssertionError("独立 stdin 与已验证正文不同")
        if args[0] == "file":
            finals = [r["event"] for r in records if r.get("event", {}).get("kind") == "final"]
            baseline = json.loads(
                (project / "artifacts/p5/real-check/cases.json").read_text("utf-8")
            )
            expected = [e["result"]["text"] for e in baseline[6]["events"] if e["kind"] == "final"]
            if [e["result"]["text"] for e in finals] != expected:
                raise AssertionError("独立文件序列与已验证正文不同")
        report["commands"].append(
            {"args": args, "records": records, "exit_code": result.returncode}
        )
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("environment", type=Path)
    parser.add_argument("outside", type=Path)
    parser.add_argument("--mode", choices=("base", "microphone", "file"), required=True)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[2]
    report = run(project, args.environment.resolve(), args.outside.resolve(), mode=args.mode)
    path = project / f"artifacts/p5/installed-{args.mode}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(args.mode, "installed verification passed")


if __name__ == "__main__":
    main()
