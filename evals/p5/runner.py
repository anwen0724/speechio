"""公开语料与有限真实采集；命令通过实际入口运行并对照 SDK。"""

import json
import os
import subprocess
import sys
from dataclasses import asdict

import numpy as np
import soundfile as sf

from evals.p4.runner import read_audio
from realtime_asr import AudioFormat, Recognizer, RecognizerConfig, list_input_devices


def run(project, folder):
    """三条模型路径分别覆盖文件、序列、stdin 与自动模式。"""
    cpu = RecognizerConfig(model=project / ".runtime/models/whisper-tiny")
    gpu = RecognizerConfig(
        model=cpu.model, device="cuda", cudnn_folder=project / ".runtime/cudnn/9.16.0.29"
    )
    native = RecognizerConfig(model=project / ".runtime/models/nemotron", engine="nemotron")
    chinese = project / "data/corpus/p0/C01-1.wav"
    stereo = folder / "C01-stereo-48k.wav"
    voice = read_audio(chinese)
    expanded = np.repeat(voice, 3)
    sf.write(stereo, np.column_stack((expanded, expanded)), 48000, subtype="PCM_16")
    reports = []
    for name, config in (("whisper-cpu", cpu), ("whisper-cuda", gpu), ("nemotron", native)):
        options = [
            "--model",
            str(config.model),
            "--backend",
            config.engine,
            "--device",
            config.device,
            "--output",
            "jsonl",
        ]
        if config.cudnn_folder is not None:
            options += ["--cudnn-folder", str(config.cudnn_folder)]
        with Recognizer(config) as recognizer:
            for sample_id, language in (("C01-1", "zh-CN"), ("C07-1", "en")):
                path = project / f"data/corpus/p0/{sample_id}.wav"
                pcm = read_audio(path)
                expected = recognizer.transcribe(pcm, format=AudioFormat(), language=language).text
                for command in ("file", "stdin"):
                    args = [
                        command,
                        *([str(path)] if command == "file" else []),
                        *options,
                        "--language",
                        language,
                    ]
                    report = invoke(project, args, pcm.tobytes() if command == "stdin" else None)
                    finals = [e for e in report["events"] if e["kind"] == "final"]
                    if [e["result"]["text"] for e in finals] != [expected]:
                        raise AssertionError("CLI 最终正文与同步 SDK 不同")
                    report["matches_sdk"] = True
                    report["configuration"] = name
                    report["sample_id"] = sample_id
                    reports.append(report)
                args = [
                    "file",
                    str(path),
                    *options,
                    "--language",
                    language,
                    "--mode",
                    "auto",
                    "--vad-model",
                    str(project / ".runtime/models/vad/silero-v6.onnx"),
                    "--webrtc-mode",
                    "0" if language == "en" else "3",
                ]
                report = invoke(project, args)
                finals = [e for e in report["events"] if e["kind"] == "final"]
                if not finals:
                    raise AssertionError("真实自动文件没有生成段")
                for event in finals:
                    boundary = event["boundary"]
                    segment = pcm[boundary["start_sample"] : boundary["end_sample"]]
                    expected_segment = recognizer.transcribe(
                        segment, format=AudioFormat(), language=language
                    ).text
                    if event["result"]["text"] != expected_segment:
                        raise AssertionError("CLI 自动最终与同范围 SDK 不同")
                report["matches_sdk"] = True
                report["configuration"] = name
                report["sample_id"] = sample_id
                reports.append(report)
            expected_sequence = [
                recognizer.transcribe_file(p, language="zh-CN").text for p in (chinese, stereo)
            ]
        report = invoke(
            project, ["file", str(chinese), str(stereo), *options, "--language", "zh-CN"]
        )
        finals = [e for e in report["events"] if e["kind"] == "final"]
        if [e["result"]["text"] for e in finals] != expected_sequence:
            raise AssertionError("不同格式的文件序列与 SDK 不同")
        report["matches_sdk"] = True
        report["configuration"] = name
        report["sample_id"] = "sequence-16k-mono-48k-stereo"
        reports.append(report)
        print(name, "file/stdin/automatic/sequence verified", flush=True)
    devices = [asdict(d) for d in list_input_devices()]
    captures = []
    # 索引来自已实际枚举的设备；此处不假设名称能唯一定位硬件。
    for device in devices:
        if device["index"] not in (1, 2):
            continue
        args = [
            "mic",
            "--model",
            str(cpu.model),
            "--input-device",
            str(device["index"]),
            "--mode",
            "manual",
            "--duration",
            "2",
            "--output",
            "jsonl",
        ]
        report = invoke(project, args)
        if not any(e["kind"] == "realtime" for e in report["events"]):
            raise AssertionError("真实麦克风没有基本实时观察")
        captures.append(report)
        print(device["index"], "microphone real model verified", flush=True)
    native_report = invoke(
        project,
        [
            "mic",
            "--model",
            str(native.model),
            "--backend",
            "nemotron",
            "--mode",
            "manual",
            "--duration",
            "2",
            "--output",
            "jsonl",
        ],
    )
    if not any(e["source"] == "native" for e in native_report["events"]):
        raise AssertionError("真实麦克风没有原生观察")
    captures.append(native_report)
    (folder / "cases.json").write_text(
        json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (folder / "microphone.json").write_text(
        json.dumps(captures, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {
        "file_and_stdin_cases": len(reports),
        "matches_sdk": True,
        "microphone_cases": len(captures),
        "devices": devices,
        "human_spoken_quality_checked": False,
        "automatic_gain": False,
    }


def invoke(project, args, input_bytes=None):
    """标准输入按原字节递交，stdout 只解析正式 CLI 的 JSONL。"""
    process = subprocess.run(
        [sys.executable, "-m", "realtime_asr", *args],
        cwd=project,
        input=input_bytes,
        capture_output=True,
        timeout=90,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )
    if process.returncode:
        raise AssertionError(f"CLI exit={process.returncode}: {process.stderr.decode('utf-8')}")
    records = [json.loads(line) for line in process.stdout.decode("utf-8").splitlines() if line]
    events = [r["event"] for r in records if r["record_type"] == "event"]
    if records[-1].get("status") != "stopped" or records[-1]["exit_code"] != 0:
        raise AssertionError("CLI 没有正常停止状态")
    return {
        "args": args,
        "records": records,
        "events": events,
        "stderr": process.stderr.decode("utf-8"),
        "exit_code": process.returncode,
    }


def check_broken_pipe(project, *, idle_stdin=False):
    """真实下游在准备消息后关闭管道，检查产品清理和进程退出码。"""
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "realtime_asr",
            *(
                ["stdin", "--realtime"]
                if idle_stdin
                else ["file", str(project / "data/corpus/p0/C01-1.wav")]
            ),
            "--model",
            str(project / ".runtime/models/whisper-tiny"),
            "--output",
            "jsonl",
        ],
        cwd=project,
        stdin=subprocess.PIPE if idle_stdin else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )
    try:
        first = json.loads(process.stdout.readline())
        records = [first]
        if idle_stdin:
            while records[-1].get("status") != "ready":
                records.append(json.loads(process.stdout.readline()))
            process.stdin.write(b"\x01\x00" * 1024)
            process.stdin.flush()
            while records[-1].get("event", {}).get("kind") != "realtime":
                records.append(json.loads(process.stdout.readline()))
        process.stdout.close()
        if idle_stdin:
            # 第二个前缀会触发下一次输出，随后保持上游打开但空闲。
            process.stdin.write(b"\x01\x00" * 1024)
            process.stdin.flush()
        code = process.wait(timeout=30)
        diagnostic = process.stderr.read().decode("utf-8")
        if code != 1 or "Traceback" in diagnostic:
            raise AssertionError(f"关闭输出管道后 exit={code}: {diagnostic}")
        return {
            "records": records,
            "idle_stdin": idle_stdin,
            "exit_code": code,
            "stderr": diagnostic,
        }
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        process.stdout.close()
        process.stderr.close()
        if process.stdin is not None:
            process.stdin.close()
