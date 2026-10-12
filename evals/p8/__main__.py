"""P8 边界整理的真实本地回归；复用已有输入和入口验证工具。"""

import argparse
import json
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf

from evals.p7.__main__ import (
    ModelMeasurement,
    early_checks,
    model_configs,
    network,
    run_case,
)
from realtime_asr import ASRRuntime, EndpointConfig, RealtimeConfig, SessionConfig


def matrix(folder):
    """对照三个实际模型路径，保留正文、时间戳及策略的原有行为。"""
    previous = json.loads(Path("artifacts/p7/real-check/matrix.json").read_text(encoding="utf-8"))
    rows = []
    for name, config in model_configs():
        measurement = ModelMeasurement()
        if config.backend == "whisper":
            from faster_whisper import BatchedInferencePipeline

            instrument = patch.object(
                BatchedInferencePipeline,
                "transcribe",
                measurement.wrap(BatchedInferencePipeline.transcribe),
            )
        else:
            instrument = nullcontext()
        with (
            instrument,
            ASRRuntime(
                config, realtime=RealtimeConfig(mode=config.capabilities.realtime_mode)
            ) as runtime,
        ):
            # 首次 CUDA 推理需要初始化计算资源。预热单元也执行真实最终流程，
            # 再对照持续使用的实时行为；不把冷启动耗时隐藏到核心实现里。
            warm_audio, _ = sf.read("data/corpus/p0/C01-1.wav", dtype="int16")
            warm = runtime.create_session(SessionConfig(language="zh"))
            try:
                warm.start()
                warm.feed_audio(warm_audio)
                warm.finish()
                final = warm.get_event(timeout=60)
                completed = warm.get_event(timeout=60)
                if final.kind != "final" or not final.result.text or completed.kind != "completed":
                    raise AssertionError("真实预热单元失败")
            finally:
                warm.close()
            for language, filename in (("zh", "C01-1.wav"), ("en", "C07-1.wav")):
                pcm, _ = sf.read(f"data/corpus/p0/{filename}", dtype="int16")
                cases = [
                    ("periodic", pcm, False, None),
                    (
                        "dynamic",
                        np.concatenate((pcm, np.zeros(16000, np.int16))),
                        True,
                        EndpointConfig(dynamic_silence=True),
                    ),
                ]
                if config.backend == "whisper" and language == "en":
                    cases.append(
                        (
                            "punctuation",
                            np.concatenate(
                                (pcm, np.zeros(1600, np.int16), pcm, np.zeros(16000, np.int16))
                            ),
                            True,
                            EndpointConfig(punctuation_split=True, dynamic_silence=True),
                        )
                    )
                for case, source, automatic, endpoint in cases:
                    before = len(measurement.calls)
                    row = run_case(
                        runtime,
                        source,
                        language=language,
                        name=case,
                        automatic=automatic,
                        endpoint=endpoint,
                    )
                    row["backend"] = name
                    row["model_calls"] = (
                        measurement.calls[before:] if config.backend == "whisper" else None
                    )
                    old = next(
                        item
                        for item in previous
                        if item["backend"] == name
                        and item["language"] == language
                        and item["name"] == case
                    )
                    for field in ("final_texts", "timestamp_status", "ranges"):
                        if row[field] != old[field]:
                            raise AssertionError(f"{name}/{language}/{case}: {field} 改变")
                    if case == "punctuation" and not row["optimization"].get("punctuation_splits"):
                        raise AssertionError("英文真实句末未切段")
                    rows.append(row)
                    (folder / "matrix.json").write_text(
                        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
                    )
                    print(f"passed local {name} {language} {case}", flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("matrix", "early", "network", "remote", "cli"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/p8/real-check"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.mode == "remote":
        from evals.p8.remote import run

        (args.output / "remote.json").write_text(
            json.dumps(run(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
    else:
        {"matrix": matrix, "early": early_checks, "network": network, "cli": local_cli}[args.mode](
            args.output
        )


def local_cli(folder):
    """真实本地文件命令完成完整最终；原生实时先排空再完成。"""
    baseline = json.loads((folder / "matrix.json").read_text(encoding="utf-8"))
    rows = []
    for name, config in model_configs():
        if config.device == "cuda":
            continue
        process = subprocess.run(
            [
                sys.executable,
                "-m",
                "realtime_asr",
                "file",
                "--model",
                str(config.model_path),
                "--backend",
                config.backend,
                "--language",
                "zh",
                "--realtime",
                "--output",
                "jsonl",
                "data/corpus/p0/C01-1.wav",
            ],
            capture_output=True,
            encoding="utf-8",
            timeout=60,
            check=True,
        )
        records = [json.loads(line) for line in process.stdout.splitlines()]
        events = [record["event"] for record in records if record["record_type"] == "event"]
        final = next(event for event in events if event["kind"] == "final")
        expected = next(
            row
            for row in baseline
            if row["backend"] == name and row["language"] == "zh" and row["name"] == "periodic"
        )
        if (
            final["result"]["text"] != expected["final_texts"][0]
            or final["audio_samples"] != 119040
        ):
            raise AssertionError(f"{name} 本地 CLI 最终结果与实际 SDK 不一致")
        if events[-1]["kind"] != "completed":
            raise AssertionError("本地 CLI 没有正常完成")
        if config.backend == "nemotron" and not any(event["stream_finished"] for event in events):
            raise AssertionError("Nemotron CLI 没有排空原生流")
        rows.append({"backend": name, "events": events, "exit_code": process.returncode})
    (folder / "local-cli.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
