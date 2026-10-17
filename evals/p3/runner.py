"""真实模型下的节奏输入、资源采样及事件对照。"""

import hashlib
import time
import wave
from dataclasses import asdict
from threading import Event, Thread

import numpy as np
import psutil

from speechio import (
    ASRError,
    AudioFormat,
    ModelConfig,
    RealtimeConfig,
    SessionConfig,
    Transcriber,
)
from speechio.runtime.asr import ASRRuntime


def configurations(project):
    """显式本地路径；覆盖两条路径、共享实例与独立角色。"""
    cpu = ModelConfig(model=project / ".runtime/models/whisper-tiny")
    gpu = ModelConfig(
        model=cpu.model,
        device="cuda",
    )
    native = ModelConfig(model=project / ".runtime/models/nemotron", engine="nemotron")
    return {
        "shared-whisper-cpu": (cpu, RealtimeConfig(), "whisper"),
        "shared-whisper-cuda": (gpu, RealtimeConfig(), "whisper-gpu"),
        "shared-nemotron-cpu": (native, RealtimeConfig(mode="native"), "nemotron"),
        "independent-whisper-cpu": (cpu, RealtimeConfig(model=cpu), "whisper"),
        "independent-native-cuda-final": (
            gpu,
            RealtimeConfig(mode="native", model=native),
            "whisper-gpu",
        ),
    }


def read_pcm(path):
    with wave.open(str(path)) as source:
        if (source.getframerate(), source.getnchannels(), source.getsampwidth()) != (16000, 1, 2):
            raise ValueError("评估需要 16 kHz 单声道 PCM16")
        return source.readframes(source.getnframes())


def check_configuration(name, config, realtime, baseline_name, project, baseline, long_seconds):
    """一个会话连续输入中、英文，再输入固定长度的重复语料。"""
    sources = []
    for sample_id, language in (("C01-1", "zh-CN"), ("C07-1", "en")):
        pcm = read_pcm(project / f"data/corpus/p0/{sample_id}.wav")
        sources.append((sample_id, language, pcm, baseline[(baseline_name, sample_id)]))
    # 长输入是可复现的重复音频，不用于正式语言质量评分。
    long_pcm = np.resize(
        np.frombuffer(sources[0][2], dtype="<i2"), int(long_seconds * 16000)
    ).tobytes()
    with Transcriber(config) as recognizer:
        long_expected = recognizer.transcribe(long_pcm, format=AudioFormat(), language="zh-CN").text
    report = {
        "configuration": name,
        "final_config": asdict(config),
        "realtime_config": asdict(realtime),
        "long_transform": {
            "source": "C01-1",
            "operation": "repeat_and_cut",
            "seconds": long_seconds,
        },
        "inputs": [],
        "events": [],
        "rss_samples": [],
        "capacity_samples": [],
    }
    runtime = ASRRuntime(config, realtime=realtime)
    process = psutil.Process()
    stop = Event()

    def sample_memory():
        while not stop.is_set():
            report["rss_samples"].append(
                {"at": time.monotonic(), "bytes": process.memory_info().rss}
            )
            stop.wait(0.05)

    monitor = Thread(target=sample_memory, name="P3ResourceSampler")
    runtime.prepare()
    monitor.start()
    sessions = {
        language: runtime.create_session(
            SessionConfig(realtime=True, language=language, max_audio_samples=480000)
        )
        for language in ("zh-CN", "en")
    }
    tasks = {}

    def consume(timeout=0):
        event = None
        # 每个会话单独维护消费序号，轮转消费避免一个会话屏蔽另一个。
        for current in sessions.values():
            event = current.next_event()
            if event is not None:
                break
        if event is None and timeout:
            pending = next(
                current
                for current in sessions.values()
                if any(
                    key[0] == current.session_id and not entry.get("completed")
                    for key, entry in tasks.items()
                )
            )
            event = pending.next_event(timeout=timeout)
        if event is None:
            return False
        received_at = time.monotonic()
        record = asdict(event)
        record["received_at_monotonic"] = received_at
        report["events"].append(record)
        if event.kind in ("failed", "realtime_failed", "cancelled"):
            raise AssertionError(f"真实识别失败：{event}")
        if event.kind == "final":
            entry = tasks[(event.turn.session_id, event.turn.turn_id)]
            if (
                event.result.text != entry["expected_text"]
                or event.audio_samples != entry["samples"]
            ):
                raise AssertionError(f"{name} 最终正文或范围与整段基线不同")
            entry["same_baseline_text"] = True
            entry["final_wait_seconds"] = received_at - entry["finish_started_at"]
        elif event.kind == "completed":
            tasks[(event.turn.session_id, event.turn.turn_id)]["completed"] = True
        return True

    try:
        inputs = sources * 2 + [("long-C01-repeat", "zh-CN", long_pcm, long_expected)]
        for sample_id, language, pcm, expected in inputs:
            session = sessions[language]
            unit = session.start()
            entry = {
                "sample_id": sample_id,
                "language": language,
                "unit": asdict(unit),
                "samples": len(pcm) // 2,
                "pcm_sha256": hashlib.sha256(pcm).hexdigest(),
                "expected_text": expected,
                "start_at_monotonic": time.monotonic(),
            }
            report["inputs"].append(entry)
            tasks[(unit.session_id, unit.turn_id)] = entry
            for offset in range(0, len(pcm), 6400):
                session.feed_audio(pcm[offset : offset + 6400])
                report["capacity_samples"].append(asdict(runtime.capacity))
                while consume():
                    pass
                due = entry["start_at_monotonic"] + min(offset + 6400, len(pcm)) / 32000
                time.sleep(max(0, due - time.monotonic()))
            if sample_id.startswith("long"):
                try:
                    session.feed_audio(b"\0\0")
                except ASRError as exc:
                    if exc.code != "AUDIO_CAPACITY_EXCEEDED":
                        raise
                    entry["over_capacity_rejected"] = True
                else:
                    raise AssertionError("长输入超过配置仍被接纳")
            entry["finish_started_at"] = time.monotonic()
            entry["task"] = asdict(session.end_turn())
            # 不等待最终返回，直接开始下一单元。
        while sum(e["kind"] == "completed" for e in report["events"]) < len(tasks):
            if not consume(timeout=60):
                raise TimeoutError("真实实时会话未完成")
        for entry in report["inputs"]:
            observations = [
                e
                for e in report["events"]
                if e["kind"] == "realtime" and e["unit"] == entry["unit"]
            ]
            before_end = [
                e for e in observations if e["completed_at_monotonic"] < entry["finish_started_at"]
            ]
            if not before_end:
                raise AssertionError("持续输入期间没有真实实时观察")
            versions = [e["audio_version"] for e in observations]
            if versions != sorted(versions) or (
                realtime.mode == "snapshot" and len(set(versions)) != len(versions)
            ):
                raise AssertionError("实时版本倒退或快照重复")
            if any(v > entry["samples"] for v in versions):
                raise AssertionError("实时覆盖超过有效输入")
            if realtime.mode == "native" and not any(
                e["stream_finished"] and e["audio_samples"] == entry["samples"]
                for e in observations
            ):
                raise AssertionError("原生结束范围不完整")
            entry["realtime_count"] = len(observations)
            entry["before_finish_count"] = len(before_end)
        report["session_close"] = [asdict(session.close()) for session in sessions.values()]
        for current in sessions.values():
            events = [e for e in report["events"] if e["unit"]["session_id"] == current.session_id]
            if [e["sequence"] for e in events] != list(range(1, len(events) + 1)):
                raise AssertionError("会话消费序号不连续")
            for previous, following in zip(events, events[1:]):
                if previous["kind"] == "final" and (
                    following["kind"] != "completed" or previous["unit"] != following["unit"]
                ):
                    raise AssertionError("最终与完成事件之间插入其他结果")
        if any(
            c["realtime_units"] > realtime.max_units
            or c["realtime_reserved_samples"] > realtime.max_audio_samples
            for c in report["capacity_samples"]
        ):
            raise AssertionError("实时物理计量超过配置容量")
    finally:
        report["close"] = asdict(runtime.close(timeout=60))
        stop.set()
        monitor.join()
        report["capacity_after_close"] = asdict(runtime.capacity)
    if report["close"]["status"] != "ok" or any(report["capacity_after_close"].values()):
        raise AssertionError("真实模型清理或容量释放失败")
    report["rss_min_bytes"] = min(r["bytes"] for r in report["rss_samples"])
    report["rss_peak_bytes"] = max(r["bytes"] for r in report["rss_samples"])
    report["status"] = "ok"
    return report
