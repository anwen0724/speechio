"""同公开音频的策略 A/B，记录范围与成本，不代替人工质量评分。"""

import argparse
import json
import platform
import subprocess
import sys
from collections import Counter
from dataclasses import asdict
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from threading import Event, Thread
from time import monotonic, sleep
from unittest.mock import patch

import numpy as np
import soundfile as sf

from evals.p6.network import running_server
from realtime_asr import (
    ASRRuntime,
    EndpointConfig,
    RealtimeConfig,
    Recognizer,
    RecognizerConfig,
    SegmentationConfig,
    SessionConfig,
    StabilizationConfig,
    VadConfig,
)
from realtime_asr.client import RemoteClient
from realtime_asr.server import ServerConfig


def digest(audio):
    return sha256(audio.astype(np.float32).tobytes()).hexdigest()


class ModelMeasurement:
    """完整执行真实 Whisper 调用，记录外部推理耗时与输入摘要。"""

    def __init__(self):
        self.calls = []

    def wrap(self, original):
        def measured(pipeline, audio, **options):
            started = monotonic()
            segments, info = original(pipeline, audio, **options)
            complete = list(segments)
            self.calls.append(
                {
                    "samples": len(audio),
                    "digest": digest(audio),
                    "role": "snapshot" if options["beam_size"] == 3 else "final",
                    "word_timestamps": options["word_timestamps"],
                    "execution_ms": (monotonic() - started) * 1000,
                }
            )
            return iter(complete), info

        return measured


def run_case(runtime, pcm, *, language, name, automatic=False, endpoint=None):
    """按源样本节奏递交，独立消费结果；网络和模型耗时不作为音频时钟。"""
    config = SessionConfig(
        language=language, realtime=True, word_timestamps=True, stabilization=StabilizationConfig()
    )
    if automatic:
        session = runtime.create_auto_session(
            config=config,
            vad=VadConfig(model_path=Path(".runtime/models/vad/silero-v6.onnx"), webrtc_mode=0),
            endpoint=endpoint,
        )
    else:
        session = runtime.create_session(config)
        session.start()
    events, errors = [], []
    input_done = Event()
    completed = Event()
    started = monotonic()

    def consume():
        try:
            while True:
                event = session.get_event(timeout=0.05)
                if event is not None:
                    if event.error is not None:
                        raise AssertionError(asdict(event))
                    events.append({"received_at": monotonic(), **asdict(event)})
                    if not automatic and event.kind == "completed":
                        completed.set()
                if input_done.is_set() and (
                    session.pending_segments == 0 if automatic else completed.is_set()
                ):
                    break
        except Exception as error:
            errors.append(str(error))

    worker = Thread(target=consume, name="ASRP7EvalConsumer")
    worker.start()
    try:
        for offset in range(0, len(pcm), 1024):
            if errors:
                raise AssertionError(errors)
            session.feed_audio(pcm[offset : offset + 1024])
            sleep(max(0, started + min(offset + 1024, len(pcm)) / 16000 - monotonic()))
        ended = monotonic()
        if automatic:
            session.finish_input()
        else:
            session.finish()
        input_done.set()
        worker.join(60)
        if worker.is_alive() or errors:
            raise AssertionError(errors or "消费线程未完成")
        realtime = [event for event in events if event["kind"] == "realtime"]
        finals = [event for event in events if event["kind"] == "final"]
        if not finals or not any(event["result"]["text"] for event in realtime):
            raise AssertionError("缺少真实实时或最终正文")
        ranges = []
        for event in finals:
            boundary = event["boundary"]
            start = boundary["start_sample"] if boundary else 0
            end = boundary["end_sample"] if boundary else len(pcm)
            if not 0 <= start < end <= len(pcm) or event["audio_samples"] != end - start:
                raise AssertionError("实际最终范围错误")
            for word in event["result"]["words"] or []:
                if not 0 <= word["start_sample"] <= word["end_sample"] <= end - start:
                    raise AssertionError("真实词时间超出本段")
            ranges.append(
                {
                    "start": start,
                    "end": end,
                    "digest": digest(pcm[start:end].astype(np.float32) / 32768),
                }
            )
        result = {
            "name": name,
            "language": language,
            "input_samples": len(pcm),
            "elapsed_ms": (monotonic() - started) * 1000,
            "finish_to_last_final_ms": max(0, (finals[-1]["received_at"] - ended) * 1000),
            "first_nonempty_realtime_ms": (
                next(event["received_at"] for event in realtime if event["result"]["text"])
                - started
            )
            * 1000,
            "trigger_counts": dict(Counter(event["trigger_reason"] for event in realtime)),
            "stable_observations": sum(
                bool(event["stabilization"]["stable_text"]) for event in realtime
            ),
            "max_retained_observations": max(
                event["stabilization"]["retained_observations"] for event in realtime
            ),
            "final_texts": [event["result"]["text"] for event in finals],
            "timestamp_status": [event["result"]["timestamp_status"] for event in finals],
            "word_counts": [len(event["result"]["words"] or []) for event in finals],
            "finalizations": [event["finalization"] for event in finals],
            "boundaries": [event["boundary"] for event in finals],
            "ranges": ranges,
            "optimization": session.optimization_status if automatic else {},
            "events": events,
        }
        if result["max_retained_observations"] > 128:
            raise AssertionError("稳定历史超过规格容量")
        return result
    finally:
        session.close()
        input_done.set()
        completed.set()
        worker.join(5)


def model_configs():
    """三条已准备路径由一个位置声明，局部验证沿用相同环境。"""
    whisper = Path(".runtime/models/whisper-tiny").resolve()
    configs = (
        ("whisper-cpu", RecognizerConfig(model_path=whisper)),
        (
            "whisper-cuda",
            RecognizerConfig(
                model_path=whisper,
                device="cuda",
                cudnn_folder=Path(".runtime/cudnn/9.16.0.29").resolve(),
            ),
        ),
        (
            "nemotron",
            RecognizerConfig(
                model_path=Path(".runtime/models/nemotron").resolve(), backend="nemotron"
            ),
        ),
    )
    return configs


def matrix(folder):
    """两种语言分别比较基线与优化，句末使用连续的两段录音。"""
    corpus = {
        language: sf.read(f"data/corpus/p0/{filename}", dtype="int16")[0]
        for language, filename in (("zh", "C01-1.wav"), ("en", "C07-1.wav"))
    }
    rows = []
    for backend, config in model_configs():
        native = config.backend == "nemotron"
        modes = (False,) if native else (False, True)
        baseline = {}
        for acoustic in modes:
            measurement = ModelMeasurement()
            if native:
                from contextlib import nullcontext

                instrument = nullcontext()
            else:
                from faster_whisper import BatchedInferencePipeline

                instrument = patch.object(
                    BatchedInferencePipeline,
                    "transcribe",
                    measurement.wrap(BatchedInferencePipeline.transcribe),
                )
            with (
                instrument,
                ASRRuntime(
                    config,
                    realtime=RealtimeConfig(
                        mode="native" if native else "snapshot",
                        acoustic_boundaries=acoustic,
                    ),
                ) as runtime,
            ):
                for language, pcm in corpus.items():
                    plans = [("acoustic" if acoustic else "periodic", False, None)]
                    if not acoustic:
                        plans.extend(
                            (
                                ("vad-baseline", True, EndpointConfig()),
                                ("early", True, EndpointConfig(early_transcription_silence=0.2)),
                                ("dynamic", True, EndpointConfig(dynamic_silence=True)),
                            )
                        )
                    for name, automatic, endpoint in plans:
                        before = len(measurement.calls)
                        padded = (
                            np.concatenate((pcm, np.zeros(16000, np.int16))) if automatic else pcm
                        )
                        result = run_case(
                            runtime,
                            padded,
                            language=language,
                            name=name,
                            automatic=automatic,
                            endpoint=endpoint,
                        )
                        result["backend"] = backend
                        result["model_calls"] = measurement.calls[before:] if not native else None
                        if not automatic:
                            if acoustic:
                                if result["final_texts"] != baseline[language]:
                                    raise AssertionError("声学调度改变完整最终正文")
                            else:
                                baseline[language] = result["final_texts"]
                        if not native and any(
                            status != "reported" for status in result["timestamp_status"]
                        ):
                            raise AssertionError("Whisper 实际词时间不可用")
                        if native and any(
                            status != "unavailable" for status in result["timestamp_status"]
                        ):
                            raise AssertionError("Nemotron 伪造词时间能力")
                        rows.append(result)
                        (folder / "matrix.json").write_text(
                            json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
                        )
                        print(
                            f"passed {backend} {language} {name}; "
                            f"stable={result['stable_observations']} units={len(result['ranges'])}",
                            flush=True,
                        )
                if not native and not acoustic:
                    for language, pcm in corpus.items():
                        before = len(measurement.calls)
                        result = run_case(
                            runtime,
                            np.concatenate(
                                (pcm, np.zeros(1600, np.int16), pcm, np.zeros(16000, np.int16))
                            ),
                            language=language,
                            name="punctuation",
                            automatic=True,
                            endpoint=EndpointConfig(punctuation_split=True, dynamic_silence=True),
                        )
                        result["backend"] = backend
                        result["model_calls"] = measurement.calls[before:]
                        result["punctuation_triggered"] = bool(
                            result["optimization"].get("punctuation_splits")
                        )
                        rows.append(result)
                        (folder / "matrix.json").write_text(
                            json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
                        )
                        print(
                            f"sentence {backend} {language}: {result['optimization']}", flush=True
                        )
    return rows


def sentence_checks(folder):
    """只重测本次修改影响的中英文句末路径，替换已保存的对应记录。"""
    from faster_whisper import BatchedInferencePipeline

    existing = folder / "matrix.json"
    rows = json.loads(existing.read_text(encoding="utf-8")) if existing.exists() else []
    rows = [row for row in rows if row["name"] != "punctuation"]
    for backend, config in model_configs():
        if config.backend != "whisper":
            continue
        measurement = ModelMeasurement()
        with (
            patch.object(
                BatchedInferencePipeline,
                "transcribe",
                measurement.wrap(BatchedInferencePipeline.transcribe),
            ),
            ASRRuntime(config, realtime=RealtimeConfig()) as runtime,
        ):
            for language, filename in (("zh", "C01-1.wav"), ("en", "C07-1.wav")):
                pcm, _ = sf.read(f"data/corpus/p0/{filename}", dtype="int16")
                before = len(measurement.calls)
                result = run_case(
                    runtime,
                    np.concatenate((pcm, np.zeros(1600, np.int16), pcm, np.zeros(16000, np.int16))),
                    language=language,
                    name="punctuation",
                    automatic=True,
                    endpoint=EndpointConfig(punctuation_split=True, dynamic_silence=True),
                )
                result["backend"] = backend
                result["model_calls"] = measurement.calls[before:]
                result["punctuation_triggered"] = bool(
                    result["optimization"].get("punctuation_splits")
                )
                rows.append(result)
                existing.write_text(
                    json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                print(f"sentence {backend} {language}: {result['optimization']}", flush=True)
    return rows


def network(folder):
    """真实 HTTP 词时间、远程 WS 稳定以及 Chrome 采集展示。"""
    config = RecognizerConfig(model_path=Path(".runtime/models/whisper-tiny").resolve())
    pcm, _ = sf.read("data/corpus/p0/C01-1.wav", dtype="int16")
    with Recognizer(config) as recognizer:
        expected = recognizer.transcribe_file("data/corpus/p0/C01-1.wav", word_timestamps=True)
    server = ServerConfig(
        recognizer=config, vad_model=str(Path(".runtime/models/vad/silero-v6.onnx").resolve())
    )
    with running_server(server) as url:
        with RemoteClient(url) as client:
            result = client.transcribe_file("data/corpus/p0/C01-1.wav", word_timestamps=True)
            if (
                result["result"]["text"] != expected.text
                or result["result"]["timestamp_status"] != "reported"
            ):
                raise AssertionError("真实远程词时间与 SDK 不一致")
            cli = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "realtime_asr",
                    "remote",
                    "--url",
                    url,
                    "--word-timestamps",
                    "--output",
                    "jsonl",
                    "file",
                    "data/corpus/p0/C01-1.wav",
                ],
                capture_output=True,
                encoding="utf-8",
                check=True,
            )
            cli_result = json.loads(cli.stdout)
            if cli_result["result"]["timestamp_status"] != "reported":
                raise AssertionError("远程 CLI 未传递词时间选项")
            (folder / "remote-cli.json").write_text(cli.stdout, encoding="utf-8")
            connection = client.connect(stable_text=True, word_timestamps=True, language="zh")
            connection.start()
            start = monotonic()
            for offset in range(0, len(pcm), 1600):
                connection.feed_audio(pcm[offset : offset + 1600].tobytes())
                sleep(max(0, start + min(offset + 1600, len(pcm)) / 16000 - monotonic()))
            connection.finish()
            connection.close()
            messages = []
            while message := connection.get_message(timeout=0):
                messages.append(message)
            if not any(
                message.get("event", {}).get("stabilization", {}).get("stable_text")
                for message in messages
                if message.get("event", {}).get("stabilization")
            ):
                raise AssertionError("真实远程没有结构化稳定正文")
            (folder / "network.json").write_text(
                json.dumps({"http": result, "messages": messages}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        subprocess.run(
            [
                "node",
                "evals/p7/browser.mjs",
                url,
                str(folder.resolve()),
                str(Path("data/corpus/p0/C01-1.wav").resolve()),
            ],
            check=True,
        )


def early_checks(folder):
    """真实双 VAD 找到静音后暂停送入，核对精确复用与新增音频失效。"""
    pcm, _ = sf.read("data/corpus/p0/C01-1.wav", dtype="int16")
    source = np.concatenate((pcm, np.zeros(32000, np.int16)))
    source = np.pad(source, (0, (-len(source)) % 512))
    voiced_frames = [pcm[offset : offset + 512] for offset in range(0, len(pcm) - 511, 512)]
    voice = max(voiced_frames, key=lambda frame: float(np.mean(frame.astype(np.float64) ** 2)))
    rows = []
    for name, config in model_configs():
        with ASRRuntime(config) as runtime:
            for resume in (False, True):
                session = runtime.create_auto_session(
                    config=SessionConfig(language="zh", word_timestamps=True),
                    vad=VadConfig(
                        model_path=Path(".runtime/models/vad/silero-v6.onnx"), webrtc_mode=0
                    ),
                    segmentation=SegmentationConfig(post_speech_silence=2),
                    endpoint=EndpointConfig(early_transcription_silence=0.2),
                )
                try:
                    for offset in range(0, len(source), 512):
                        session.feed_audio(source[offset : offset + 512])
                        if session.optimization_status.get("early_submitted"):
                            break
                    else:
                        raise AssertionError("真实语音未进入提前推理静音条件")
                    deadline = monotonic() + 60
                    while not session.optimization_status.get("early_completed"):
                        if monotonic() >= deadline:
                            raise AssertionError("真实提前模型未完成")
                        sleep(0.01)
                    if resume:
                        session.feed_audio(voice)
                    session.finish()
                    final = session.get_event(timeout=60)
                    if (
                        final is None
                        or final.kind != "final"
                        or final.source != ("final" if resume else "early-final")
                    ):
                        raise AssertionError("真实提前结果复用资格错误")
                    if session.get_event(timeout=60).kind != "completed":
                        raise AssertionError("真实提前验证未终结")
                    rows.append(
                        {
                            "backend": name,
                            "resume": resume,
                            "final": asdict(final),
                            "optimization": session.optimization_status,
                        }
                    )
                    print(f"passed real early reuse {name} resume={resume}", flush=True)
                finally:
                    session.close()
    (folder / "early-exact.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return rows


def main():
    parser = argparse.ArgumentParser(description="P7 真实模型验证，需要已准备模型和公开 WAV")
    parser.add_argument("--output", type=Path, default=Path("artifacts/p7/real-check"))
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--network-only", action="store_true")
    modes.add_argument("--early-only", action="store_true")
    modes.add_argument("--sentence-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.early_only:
        early_checks(args.output)
        return
    if args.sentence_only:
        sentence_checks(args.output)
        from evals.p7.report import summarize

        summarize(args.output)
        return
    rows = [] if args.network_only else matrix(args.output)
    network(args.output)
    if rows:
        early_checks(args.output)
    if rows:
        from evals.p7.report import summarize

        summarize(args.output)
    report = {
        "status": "passed",
        "utc": datetime.now(UTC).isoformat(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cases": len(rows),
        "formal_quality": "not_accepted_pending_human_corpus_review",
        "limits": ["single-machine configurations", "public browser source, not human microphone"],
        "case_summaries": [
            {key: value for key, value in row.items() if key not in ("events", "model_calls")}
            for row in rows
        ],
    }
    (args.output / "verification.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("P7 real verification passed", flush=True)


if __name__ == "__main__":
    main()
