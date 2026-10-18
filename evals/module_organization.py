"""显式迁移前后回放：只对照稳定行为，不固定线程观察次数。"""

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from evals.p4.runner import read_audio
from speechio import (
    AudioFormat,
    ModelConfig,
    RealtimeConfig,
    SegmentationConfig,
    SessionConfig,
    Transcriber,
    VadConfig,
)
from speechio.runtime.asr import ASRRuntime


def replay(project):
    """覆盖实际三条路径、中英文及自动分句与实时联合运行。"""
    cpu = ModelConfig(engine="whisper", model_path=project / ".runtime/models/whisper-tiny")
    gpu = ModelConfig(
        engine="whisper",
        model_path=cpu.model_path,
        device="gpu",
    )
    native = ModelConfig(model_path=project / ".runtime/models/nemotron", engine="nemotron")
    reports = []
    for name, config, mode in (
        ("whisper-cpu", cpu, "snapshot"),
        ("whisper-cuda", gpu, "snapshot"),
        ("nemotron", native, "streaming"),
    ):
        with ASRRuntime(
            config, realtime=RealtimeConfig(mode=mode, initial_delay=0, interval=0.01)
        ) as runtime:
            for sample_id, language in (("C01-1", "zh-CN"), ("C07-1", "en")):
                voice = read_audio(project / f"data/corpus/p0/{sample_id}.wav")
                with Transcriber(config) as recognizer:
                    sync = recognizer.transcribe(voice, format=AudioFormat(), language=language)
                audio = np.concatenate(
                    (
                        np.zeros(20800, np.int16),
                        voice,
                        np.zeros(22400, np.int16),
                        voice,
                        np.zeros(22400, np.int16),
                    )
                )
                session = runtime.create_auto_session(
                    vad=VadConfig(
                        model_path=project / ".runtime/models/vad/silero-v6.onnx",
                        webrtc_mode=0 if language == "en" else 3,
                    ),
                    segmentation=SegmentationConfig(trim_pre_roll=True, backdate_duration=0.2),
                    config=SessionConfig(language=language, realtime=True),
                )
                report = {
                    "configuration": name,
                    "sample_id": sample_id,
                    "sync_text": sync.text,
                    "finals": [],
                    "terminal_order": [],
                    "realtime_observations": 0,
                }

                def consume(timeout=0):
                    event = session.next_event(timeout=timeout)
                    if event is None:
                        return False
                    if event.kind in ("failed", "realtime_failed", "cancelled"):
                        raise AssertionError(f"回放失败：{event}")
                    if event.kind == "realtime":
                        if (
                            not 0
                            <= event.stream_start_sample
                            <= event.stream_end_sample
                            <= len(audio)
                        ):
                            raise AssertionError("实时样本范围无效")
                        report["realtime_observations"] += 1
                    else:
                        report["terminal_order"].append(event.kind)
                    if event.kind == "final":
                        report["finals"].append(
                            {
                                "text": event.result.text,
                                "boundary": asdict(event.boundary),
                                "samples": event.audio_samples,
                            }
                        )
                    return True

                try:
                    for start in range(0, len(audio), 512):
                        session.feed_audio(audio[start : start + 512])
                        while consume():
                            pass
                        # 真实回放使用输入节奏，使结束前的快照能实际交付。
                        time.sleep(0.008)
                    session.end_input()
                    while session.pending_segments:
                        if not consume(timeout=30):
                            raise AssertionError("自动段回放超时")
                    if not report["realtime_observations"] or not report["finals"]:
                        raise AssertionError("没有实际实时观察或最终段")
                finally:
                    session.close()
                reports.append(report)
                print(f"{name}/{sample_id}: {len(report['finals'])} segments", flush=True)
        if runtime.close().status != "ok" or any(asdict(runtime.capacity).values()):
            raise AssertionError("回放后运行时没有释放容量")
    return reports


def main():
    parser = argparse.ArgumentParser(description="功能目录整理真实回放")
    parser.add_argument("phase", choices=("before", "after"))
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    folder = project / "artifacts/module-organization"
    folder.mkdir(parents=True, exist_ok=True)
    records = replay(project)
    path = folder / f"replay-{args.phase}.json"
    path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.phase == "after":
        before = json.loads((folder / "replay-before.json").read_text(encoding="utf-8"))

        def stable(rows):
            return [
                {key: value for key, value in row.items() if key != "realtime_observations"}
                for row in rows
            ]

        if stable(before) != stable(records):
            raise AssertionError("普通正文、自动边界、最终正文或终态顺序发生变化")
        print("All stable replay results match the pre-migration baseline.", flush=True)


if __name__ == "__main__":
    main()
