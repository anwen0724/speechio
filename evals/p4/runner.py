"""公开录音、可控静音与真实模型组合；记录边界而不伪造人工评分。"""

import hashlib
import json
import time
import wave
from dataclasses import asdict

import numpy as np
import psutil

from realtime_asr import (
    AudioFormat,
    ModelConfig,
    RealtimeConfig,
    SegmentationConfig,
    SessionConfig,
    Transcriber,
    VadConfig,
)
from realtime_asr.runtime.asr import ASRRuntime


def read_audio(path):
    with wave.open(str(path)) as source:
        if (source.getframerate(), source.getnchannels(), source.getsampwidth()) != (16000, 1, 2):
            raise ValueError("语料应为 16 kHz 单声道 PCM16")
        return np.frombuffer(source.readframes(source.getnframes()), "<i2").copy()


def run(project, model, output, *, configuration=None):
    cpu = ModelConfig(model=project / ".runtime/models/whisper-tiny")
    gpu = ModelConfig(
        model=cpu.model,
        device="cuda",
    )
    native = ModelConfig(model=project / ".runtime/models/nemotron", engine="nemotron")
    configs = {
        "whisper-cpu": (cpu, RealtimeConfig()),
        "whisper-cuda": (gpu, RealtimeConfig()),
        "nemotron": (native, RealtimeConfig(mode="native")),
    }
    reports = []
    for name, (final_config, realtime) in configs.items():
        if configuration is not None and configuration != name:
            continue
        with ASRRuntime(final_config, realtime=realtime) as runtime:
            for language, sample in (("zh-CN", "C01-1"), ("en", "C07-1")):
                original = read_audio(project / f"data/corpus/p0/{sample}.wav")
                for silero_end in (False, True):
                    for variant in ("default", "trim", "backdate-quiet"):
                        voice = original
                        if variant == "backdate-quiet":
                            voice = (original.astype(np.float32) * 0.35).astype("<i2")
                        prefix, gap, tail = (
                            np.zeros(size, np.int16) for size in (20800, 22400, 22400)
                        )
                        audio = np.concatenate((prefix, voice, gap, voice, tail))
                        report = check_case(
                            runtime,
                            final_config,
                            model,
                            language,
                            sample,
                            variant,
                            silero_end,
                            audio,
                            paced=(variant == "default" and not silero_end),
                        )
                        report["configuration"] = name
                        report["source_sha256"] = hashlib.sha256(original.tobytes()).hexdigest()
                        report["transform"] = {
                            "prefix_samples": len(prefix),
                            "gap_samples": len(gap),
                            "tail_samples": len(tail),
                            "repetitions": 2,
                            "gain": 0.35 if variant == "backdate-quiet" else 1,
                        }
                        reports.append(report)
                        path = output / f"{name}-{sample}-{silero_end}-{variant}.json"
                        path.write_text(
                            json.dumps(report, ensure_ascii=False, indent=2, default=str),
                            encoding="utf-8",
                        )
                        label = f"{name}/{sample}/{silero_end}/{variant}"
                        print(f"{label}: {len(report['finals'])} segments", flush=True)
        if runtime.close().status != "ok" or any(asdict(runtime.capacity).values()):
            raise AssertionError("真实配置未释放运行时容量")
    return reports


def check_case(
    runtime, final_config, model, language, sample, variant, silero_end, audio, *, paced
):
    session = runtime.create_auto_session(
        vad=VadConfig(
            model_path=model,
            silero_end_detection=silero_end,
            webrtc_mode=0 if language == "en" else 3,
        ),
        segmentation=SegmentationConfig(
            trim_pre_roll=variant == "trim",
            backdate_duration=0.2 if variant == "backdate-quiet" else 0,
        ),
        config=SessionConfig(language=language, realtime=True),
    )
    events, finals = [], []
    report = {
        "sample_id": sample,
        "language": language,
        "variant": variant,
        "silero_end_detection": silero_end,
        "vad_info": session.vad_info,
        "paced": paced,
        "input_samples": len(audio),
        "events": events,
        "finals": finals,
        "human_annotation_checked": False,
        "webrtc_mode": 0 if language == "en" else 3,
    }
    started = time.monotonic()
    report["rss_before"] = psutil.Process().memory_info().rss

    def consume(timeout=0):
        event = session.next_event(timeout=timeout)
        if event is None:
            return False
        events.append(asdict(event))
        if event.kind in ("failed", "realtime_failed", "cancelled"):
            raise AssertionError(f"真实分句或识别失败：{event}")
        if event.kind == "final":
            boundary = event.boundary
            if boundary.end_sample - boundary.start_sample != event.audio_samples:
                raise AssertionError("最终范围与音频长度不符")
            if finals and boundary.start_sample < finals[-1]["boundary"]["end_sample"]:
                raise AssertionError("相邻录音段重叠")
            finals.append({"boundary": asdict(boundary), "text": event.result.text})
        if event.kind == "realtime" and not (
            0 <= event.stream_start_sample <= event.stream_end_sample <= len(audio)
        ):
            raise AssertionError("实时绝对范围无效")
        return True

    try:
        for start in range(0, len(audio), 512):
            session.feed_audio(audio[start : start + 512])
            while consume():
                pass
            if paced:
                # 仅真实评估建立输入节奏，不用于自动化竞态测试。
                time.sleep(0.016)
        session.end_input()
        while session.pending_segments:
            if not consume(timeout=30):
                raise AssertionError("分句结果交付超时")
        if not finals:
            raise AssertionError("真实录音没有触发录音段")
        report["realtime_observations"] = sum(event["kind"] == "realtime" for event in events)
        if paced and report["realtime_observations"] == 0:
            raise AssertionError("节奏输入没有真实实时观察")
    finally:
        report["close"] = asdict(session.close())
    # 使用另一个真实模型实例对照固定范围，避免占用共享运行时模型。
    with Transcriber(final_config) as recognizer:
        for final in finals:
            boundary = final["boundary"]
            pcm = audio[boundary["start_sample"] : boundary["end_sample"]]
            expected = recognizer.transcribe(pcm, format=AudioFormat(), language=language).text
            final["baseline_text"] = expected
            final["matches_whole_segment"] = final["text"] == expected
            if not final["matches_whole_segment"]:
                raise AssertionError("自动最终与同范围整段识别不同")
    report["elapsed_seconds"] = time.monotonic() - started
    report["rss_after"] = psutil.Process().memory_info().rss
    return report
