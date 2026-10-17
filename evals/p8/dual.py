"""P8.1 双实时的真实模型准备与联合验证，独立于正式运行入口。"""

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from threading import Thread
from time import perf_counter, sleep

import numpy as np
import psutil
import soundfile as sf

from apps.client import RemoteClient
from apps.server import ServerConfig
from evals.p6.network import running_server
from realtime_asr import (
    ASRRuntime,
    BaseTranscriptionEngine,
    RealtimeConfig,
    RealtimeMergeConfig,
    Recognizer,
    RecognizerConfig,
    SessionConfig,
    StabilizationConfig,
    VadConfig,
)
from realtime_asr.engines.local import LocalEngine
from realtime_asr.runtime.models.prepare import prepare_nemotron

OUTPUT = Path("artifacts/p8-1")


class MeasuredStream:
    """在引擎契约边界计量输入，所有识别和关闭仍由真实本地流执行。"""

    def __init__(self, actual):
        self.actual = actual
        self.samples = 0
        self.digest = hashlib.sha256()
        self.finished = False
        self.closed = False

    def accept_audio(self, audio, sample_rate=None):
        self.samples += len(audio)
        self.digest.update(audio.astype("<f4", copy=False).tobytes())
        self.actual.accept_audio(audio, sample_rate)

    def decode(self):
        self.actual.decode()

    def get_result(self):
        return self.actual.get_result()

    def finish(self):
        result = self.actual.finish()
        self.finished = True
        return result

    def close(self):
        self.actual.close()
        self.closed = self.actual.state == "closed"

    def record(self):
        return {
            "samples": self.samples,
            "digest": self.digest.hexdigest(),
            "finished": self.finished,
            "closed": self.closed,
        }


class MeasuredEngine(BaseTranscriptionEngine):
    """装饰真实本地引擎，记录输入与物理退出，不替换模型输出。"""

    def __init__(self, config):
        self.actual = LocalEngine(config)
        self.streams = []
        self.whole = []
        self.close_result = None

    @property
    def capabilities(self):
        return self.actual.capabilities

    def prepare(self):
        self.actual.prepare()

    def transcribe(self, audio, language=None, *, role="final", word_timestamps=False):
        self.whole.append(
            {
                "samples": len(audio),
                "digest": hashlib.sha256(audio.astype("<f4", copy=False).tobytes()).hexdigest(),
            }
        )
        return self.actual.transcribe(audio, language, role=role, word_timestamps=word_timestamps)

    def create_streaming_session(self, language=None):
        stream = MeasuredStream(self.actual.create_streaming_session(language))
        self.streams.append(stream)
        return stream

    def close(self):
        self.close_result = self.actual.close()
        return self.close_result


def model_config(variant):
    return RecognizerConfig(
        model=Path(f".runtime/models/nemotron-{variant}").resolve(), engine="nemotron"
    )


def run_case(pcm, language, *, dual, automatic=False):
    """按源时钟递交，独立消费原始观察；推理完成时刻不充当音频位置。"""
    accurate, fast = (
        MeasuredEngine(model_config("1120ms")),
        MeasuredEngine(model_config("80ms")) if dual else None,
    )
    process = psutil.Process()
    baseline_rss = process.memory_info().rss
    with ASRRuntime(
        engine=accurate,
        realtime=RealtimeConfig(mode="native"),
        ultrafast=RealtimeConfig(mode="native", engine=fast) if fast else None,
    ) as runtime:
        config = SessionConfig(
            language=language,
            realtime=True,
            merge=RealtimeMergeConfig() if dual else None,
            stabilization=StabilizationConfig(),
        )
        if automatic:
            session = runtime.create_auto_session(
                config=config,
                vad=VadConfig(model_path=Path(".runtime/models/vad/silero-v6.onnx"), webrtc_mode=0),
            )
        else:
            session = runtime.create_session(config)
            session.start()
        events, errors = [], []
        started = perf_counter()
        peak_rss = process.memory_info().rss
        source_end = None

        def consume():
            try:
                while True:
                    event = session.get_event(timeout=0.05)
                    if event is not None:
                        if event.error:
                            raise AssertionError(asdict(event))
                        events.append(
                            {**asdict(event), "received_ms": (perf_counter() - started) * 1000}
                        )
                    ended = source_end is not None
                    if ended and (
                        session.pending_segments == 0
                        if automatic
                        else any(event["kind"] == "completed" for event in events)
                    ):
                        return
            except Exception as error:
                errors.append(str(error))

        consumer = Thread(target=consume, name="ASRDualEvalConsumer")
        consumer.start()
        feeds = []
        try:
            for offset in range(0, len(pcm), 1024):
                if errors:
                    raise AssertionError(errors)
                before = perf_counter()
                session.feed_audio(pcm[offset : offset + 1024])
                feeds.append((perf_counter() - before) * 1000)
                peak_rss = max(peak_rss, process.memory_info().rss)
                sleep(max(0, started + min(offset + 1024, len(pcm)) / 16000 - perf_counter()))
            session.finish_input() if automatic else session.finish()
            source_end = perf_counter()
            consumer.join(60)
            if consumer.is_alive() or errors:
                raise AssertionError(errors or "双实时消费未完成")
            finals = [event for event in events if event["kind"] == "final"]
            if not finals:
                raise AssertionError("缺少完整最终结果")
            source_frames = []
            for event in finals:
                boundary = event["boundary"]
                left = boundary["start_sample"] if boundary else 0
                right = boundary["end_sample"] if boundary else len(pcm)
                source = pcm[left:right].astype("<f4") / 32768
                source_frames.append(
                    {"samples": len(source), "digest": hashlib.sha256(source.tobytes()).hexdigest()}
                )
            if accurate.whole != source_frames:
                raise AssertionError("真实最终输入不是对应完整音频")
            for engine in (accurate, fast):
                if engine is None:
                    continue
                records = [stream.record() for stream in engine.streams]
                if [
                    {key: record[key] for key in ("samples", "digest")} for record in records
                ] != source_frames:
                    raise AssertionError("真实原生角色的增量或尾部不完整")
                if not all(record["finished"] and record["closed"] for record in records):
                    raise AssertionError("真实流没有结束排空并关闭")
            first = {}
            for event in events:
                if event["kind"] == "realtime" and event["result"]["text"]:
                    first.setdefault(event["realtime_role"], event["received_ms"])
                    if event["realtime_role"] == "ultrafast" and event["stabilization"] is not None:
                        raise AssertionError("极速进入了准确稳定证据")
            if "accurate" not in first or (dual and "ultrafast" not in first):
                raise AssertionError("真实角色没有非空正文")
            merged = [
                event["merge"] for event in events if event["kind"] == "realtime" and event["merge"]
            ]
            row = {
                "language": language,
                "dual": dual,
                "automatic": automatic,
                "input_samples": len(pcm),
                "first_nonempty_ms": first,
                "finish_to_last_final_ms": max(
                    0, finals[-1]["received_ms"] - (source_end - started) * 1000
                ),
                "max_feed_ms": max(feeds),
                "rss_before": baseline_rss,
                "rss_peak": peak_rss,
                "final_texts": [event["result"]["text"] for event in finals],
                "matched_merges": sum(result["matched"] for result in merged),
                "suffix_observations": sum(bool(result["ultrafast_suffix"]) for result in merged),
                "events": events,
                "whole": accurate.whole,
                "streams": {
                    "accurate": [stream.record() for stream in accurate.streams],
                    "ultrafast": [stream.record() for stream in fast.streams] if fast else [],
                },
            }
        finally:
            session.close()
            source_end = perf_counter()
            consumer.join(5)
    row["closed"] = asdict(runtime.close_result)
    row["engine_closes"] = {
        "accurate": asdict(accurate.close_result),
        "ultrafast": asdict(fast.close_result) if fast else None,
    }
    return row


def matrix():
    """每种配置使用独立进程，避免模型缓存和分配器残留污染资源比较。"""
    rows = []
    cases = [(language, dual, False) for language in ("zh", "en") for dual in (False, True)]
    cases.append(("zh", True, True))
    for language, dual, automatic in cases:
        output = OUTPUT / f"case-{language}-{int(dual)}-{int(automatic)}.json"
        command = [
            sys.executable,
            "-m",
            "evals.p8.dual",
            "case",
            "--language",
            language,
            "--case-output",
            str(output),
        ]
        if dual:
            command.append("--dual")
        if automatic:
            command.append("--automatic")
        subprocess.run(command, check=True, timeout=90)
        row = json.loads(output.read_text(encoding="utf-8"))
        if dual and not automatic and row["final_texts"] != rows[-1]["final_texts"]:
            raise AssertionError("双实时改变同音频的最终正文")
        rows.append(row)
        (OUTPUT / "matrix.json").write_text(
            json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(
            f"passed isolated dual={dual} {language} automatic={automatic}; "
            f"first={row['first_nonempty_ms']}",
            flush=True,
        )


def network():
    """真实模型经 HTTP、WebSocket、远程 CLI 和 Chrome 输入消费双路。"""
    pcm, _ = sf.read("data/corpus/p0/C01-1.wav", dtype="int16")
    config = ServerConfig(
        recognizer=model_config("1120ms"),
        realtime_config=RealtimeConfig(mode="native"),
        ultrafast_config=RealtimeConfig(mode="native", model=model_config("80ms")),
    )
    with running_server(config) as url:
        with RemoteClient(url) as client:
            response = client.http.get("/api/v1/capabilities")
            response.raise_for_status()
            capability = response.json()
            if not capability["dual_realtime"]:
                raise AssertionError("服务没有声明双实时")
            whole = client.transcribe(pcm.tobytes(), language="zh")
            session = client.connect(mode="manual", language="zh", dual_realtime=True)
            session.start()
            session.feed_audio(pcm.tobytes())
            session.finish()
            messages = []
            while not any(
                message.get("event", {}).get("kind") == "completed" for message in messages
            ):
                message = session.get_message(timeout=60)
                if message is None or message["type"] == "error":
                    raise AssertionError(message)
                messages.append(message)
            session.close()
            roles = {
                message["event"]["realtime_role"]
                for message in messages
                if message.get("event", {}).get("kind") == "realtime"
            }
            if roles != {"accurate", "ultrafast"}:
                raise AssertionError("真实网络丢失一路原始结果")
            final = next(
                message["event"]
                for message in messages
                if message.get("event", {}).get("kind") == "final"
            )
            if final["result"]["text"] != whole["result"]["text"]:
                raise AssertionError("双实时网络改变完整最终正文")
            command = [
                sys.executable,
                "-m",
                "realtime_asr",
                "remote",
                "--url",
                url,
                "--language",
                "zh",
                "--output",
                "jsonl",
                "stdin",
                "--realtime",
                "--dual-realtime",
            ]
            process = subprocess.run(
                command, input=pcm.tobytes(), capture_output=True, check=True, timeout=60
            )
            cli = [json.loads(line) for line in process.stdout.decode("utf-8").splitlines()]
            cli_roles = {
                message["event"]["realtime_role"]
                for message in cli
                if message.get("event", {}).get("kind") == "realtime"
            }
            if cli_roles != roles:
                raise AssertionError("远程 CLI 没有交付两路")
            cli_final = next(
                message["event"]
                for message in cli
                if message.get("event", {}).get("kind") == "final"
            )
            if cli_final["result"]["text"] != final["result"]["text"]:
                raise AssertionError("远程 CLI 最终正文不同")
        subprocess.run(
            [
                "node",
                "evals/p7/browser.mjs",
                url,
                str(OUTPUT.resolve()),
                str(Path("data/corpus/p0/C01-1.wav").resolve()),
                "dual",
            ],
            check=True,
            timeout=90,
        )
    (OUTPUT / "network.json").write_text(
        json.dumps(
            {"capabilities": capability, "http": whole, "websocket": messages, "cli": cli},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print("passed actual dual network, remote CLI and Chrome", flush=True)


def local_cli():
    """真实本地 stdin 接通两种权重，角色选择与结构化输出符合 SDK。"""
    pcm, _ = sf.read("data/corpus/p0/C01-1.wav", dtype="int16")
    command = [
        sys.executable,
        "-m",
        "realtime_asr",
        "stdin",
        "--model",
        str(model_config("1120ms").model),
        "--backend",
        "nemotron",
        "--ultrafast-model",
        str(model_config("80ms").model),
        "--realtime",
        "--dual-realtime",
        "--language",
        "zh",
        "--output",
        "jsonl",
    ]
    completed = subprocess.run(
        command, input=pcm.tobytes(), capture_output=True, check=True, timeout=60
    )
    records = [json.loads(line) for line in completed.stdout.decode("utf-8").splitlines()]
    events = [record["event"] for record in records if record["record_type"] == "event"]
    roles = {event["realtime_role"] for event in events if event["kind"] == "realtime"}
    if roles != {"accurate", "ultrafast"} or events[-1]["kind"] != "completed":
        raise AssertionError("本地 CLI 没有完成双实时")
    final = next(event for event in events if event["kind"] == "final")
    expected = json.loads((OUTPUT / "models.json").read_text(encoding="utf-8"))[0]["recordings"][0][
        "result"
    ]["text"]
    if final["result"]["text"] != expected or final["audio_samples"] != len(pcm):
        raise AssertionError("本地 CLI 最终输入或正文不一致")
    (OUTPUT / "local-cli.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("passed actual local dual CLI", flush=True)


def legacy():
    """三条原有路径走实际运行时，检查正文、词时间状态及资源关闭。"""
    from evals.p7.__main__ import model_configs

    pcm, _ = sf.read("data/corpus/p0/C01-1.wav", dtype="int16")
    baseline = json.loads(Path("artifacts/p8/real-check/matrix.json").read_text(encoding="utf-8"))
    rows = []
    for name, config in model_configs():
        with ASRRuntime(
            config, realtime=RealtimeConfig(mode=config.capabilities.realtime_mode)
        ) as runtime:
            session = runtime.create_session(
                SessionConfig(language="zh", realtime=True, word_timestamps=True)
            )
            session.start()
            session.feed_audio(pcm)
            session.finish()
            events = []
            while not events or events[-1]["kind"] != "completed":
                event = session.get_event(timeout=60)
                if event is None or event.error is not None:
                    raise AssertionError(event)
                events.append(asdict(event))
            session.close()
        final = next(event for event in events if event["kind"] == "final")
        old = next(
            row
            for row in baseline
            if row["backend"] == name and row["language"] == "zh" and row["name"] == "periodic"
        )
        if (
            final["result"]["text"] != old["final_texts"][0]
            or final["result"]["timestamp_status"] != old["timestamp_status"][0]
            or runtime.close_result.status != "ok"
        ):
            raise AssertionError("原有真实模型路径发生偏离")
        rows.append({"backend": name, "events": events, "close": asdict(runtime.close_result)})
        print(f"passed legacy actual runtime {name}", flush=True)
    (OUTPUT / "legacy.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def prepare_models():
    """校验已经下载的固定包，明确目标目录，再建立离线模型清单。"""
    root = Path(".runtime/models").resolve()
    rows = []
    for variant in ("1120ms", "80ms"):
        target = (root / f"nemotron-{variant}").resolve()
        if target.parent != root:
            raise AssertionError("模型目标不在预定目录")
        manifest = prepare_nemotron(target, allow_download=True, variant=variant)
        config = RecognizerConfig(model=target, engine="nemotron")
        with Recognizer(config) as recognizer:
            recordings = []
            for language, filename in (("zh", "C01-1.wav"), ("en", "C07-1.wav")):
                started = perf_counter()
                result = recognizer.transcribe_file(f"data/corpus/p0/{filename}", language=language)
                if not result.text:
                    raise AssertionError("新变体实际识别没有正文")
                recordings.append(
                    {
                        "language": language,
                        "result": asdict(result),
                        "elapsed_ms": (perf_counter() - started) * 1000,
                    }
                )
        if recognizer.close_result.status != "ok":
            raise AssertionError("新变体未正常关闭")
        rows.append(
            {
                "variant": variant,
                "manifest": manifest,
                "recordings": recordings,
                "close": asdict(recognizer.close_result),
            }
        )
        (OUTPUT / "models.json").write_text(
            json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"verified actual model {variant}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "matrix", "network", "cli", "case", "legacy"))
    parser.add_argument("--language", choices=("zh", "en"), default="zh")
    parser.add_argument("--dual", action="store_true")
    parser.add_argument("--automatic", action="store_true")
    parser.add_argument("--case-output", type=Path)
    args = parser.parse_args()
    if args.mode == "case" and args.case_output is None:
        parser.error("case 需要 --case-output")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if args.mode == "case":
        filename = "C01-1.wav" if args.language == "zh" else "C07-1.wav"
        pcm, _ = sf.read(f"data/corpus/p0/{filename}", dtype="int16")
        if args.automatic:
            pcm = np.concatenate((pcm, np.zeros(16000, np.int16)))
        row = run_case(pcm, args.language, dual=args.dual, automatic=args.automatic)
        args.case_output.write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        {
            "prepare": prepare_models,
            "matrix": matrix,
            "network": network,
            "cli": local_cli,
            "legacy": legacy,
        }[args.mode]()


if __name__ == "__main__":
    main()
