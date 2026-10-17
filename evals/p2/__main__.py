"""显式验证连续会话，比较保存的 P1 正文，不执行正式语料评分。"""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
import wave
from dataclasses import asdict
from pathlib import Path

from realtime_asr import ASRRuntime, RecognizerConfig, SessionConfig


def check_backend(name, project, baseline):
    """每种配置复用一个模型；两种语言各执行连续单元及重置后单元。"""
    gpu = name == "whisper-gpu"
    backend = "nemotron" if name == "nemotron" else "whisper"
    model_path = (
        project / ".runtime/models" / ("nemotron" if backend == "nemotron" else "whisper-tiny")
    )
    config = RecognizerConfig(
        model=model_path,
        engine=backend,
        device="cuda" if gpu else "cpu",
    )
    model = json.loads((model_path / "model-manifest.json").read_text(encoding="utf-8"))
    report = {
        "backend": name,
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(config).items()},
        "model_identity": {k: model[k] for k in ("model_record_id", "model_id", "revision")},
        "sessions": [],
    }
    runtime = ASRRuntime(config)
    try:
        runtime.prepare()
        for sample_id, language in (("C01-1", "zh"), ("C07-1", "en")):
            audio_path = project / "data/corpus/p0" / f"{sample_id}.wav"
            with wave.open(str(audio_path)) as source:
                pcm = source.readframes(source.getnframes())
            session = runtime.create_session(SessionConfig(language=language))
            record = {
                "session_id": session.session_id,
                "sample_id": sample_id,
                "audio_sha256": hashlib.sha256(audio_path.read_bytes()).hexdigest(),
                "tasks": [],
                "events": [],
            }
            tasks = []
            for _ in range(2):
                session.start()
                for offset in range(0, len(pcm), 1001):
                    session.feed_audio(pcm[offset : offset + 1001])
                task = session.finish()
                if session.finish() is not task:
                    raise AssertionError("重复结束创建了新任务")
                tasks.append(task)
            cancelled = session.start()
            session.feed_audio(pcm[:1001])
            if not session.cancel() or session.cancel(cancelled):
                raise AssertionError("活动取消未保持一次终结")
            for task in tasks:
                consume_pair(session, task, baseline[(name, sample_id)], record)
            event = session.get_event(timeout=30)
            if event is None or event.kind != "cancelled" or event.unit != cancelled:
                raise AssertionError("活动取消事件归属错误")
            record["events"].append(asdict(event))
            change = session.reset()
            if change.generation != 1 or change.cancelled:
                raise AssertionError("重置结果错误")
            session.start()
            session.feed_audio(pcm)
            task = session.finish()
            if task.unit.generation != 1:
                raise AssertionError("新工作未进入新代次")
            consume_pair(session, task, baseline[(name, sample_id)], record)
            sequences = [e["sequence"] for e in record["events"]]
            if sequences != list(range(1, 8)) or session.get_event() is not None:
                raise AssertionError("事件顺序或一次消费错误")
            record["reset"] = asdict(change)
            record["close"] = asdict(session.close())
            report["sessions"].append(record)
        report["capacity_after_sessions"] = asdict(runtime.capacity)
        if any(asdict(runtime.capacity).values()):
            raise AssertionError("已完成工作仍占用容量")
    finally:
        report["close"] = asdict(runtime.close(timeout=30))
    if report["close"]["status"] != "ok":
        raise AssertionError("真实运行时没有完成清理")
    report["status"] = "ok"
    return report


def consume_pair(session, task, expected_text, record):
    """只通过公开事件检查固定范围、正文和完成顺序。"""
    final, completed = session.get_event(timeout=30), session.get_event(timeout=30)
    if final is None or completed is None:
        raise AssertionError("缺少最终事件")
    if (final.kind, completed.kind) != ("final", "completed"):
        raise AssertionError("最终与完成事件顺序错误")
    for event in (final, completed):
        if event.unit != task.unit or event.task_id != task.task_id:
            raise AssertionError("任务归属错误")
        if (event.audio_samples, event.audio_version) != (task.audio_samples, task.audio_version):
            raise AssertionError("封口音频范围错误")
    if final.result.text != expected_text:
        raise AssertionError("最终正文与保存的 P1 基线不同")
    record["tasks"].append({**asdict(task), "same_p1_text": True})
    record["events"].extend((asdict(final), asdict(completed)))


def main():
    parser = argparse.ArgumentParser(description="真实 P2 会话验收")
    parser.add_argument("--project", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/p2/session-check/summary.json")
    )
    args = parser.parse_args()
    project = args.project.resolve()
    saved = json.loads(
        (project / "artifacts/p1/sdk-check/summary.json").read_text(encoding="utf-8")
    )
    baseline = {
        (backend["backend"], check["sample_id"]): check["result"]["text"]
        for backend in saved["backends"]
        for check in backend["checks"]
        if check["entry"] == "array"
    }
    report = {
        "python": sys.version,
        "platform": platform.platform(),
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "faster-whisper", "sherpa-onnx", "onnxruntime")
        },
        "p1_baseline_sha256": hashlib.sha256(
            (project / "artifacts/p1/sdk-check/summary.json").read_bytes()
        ).hexdigest(),
        "p0_formal_acceptance": "pending_human_review",
        "backends": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        for backend in ("whisper", "nemotron", "whisper-gpu"):
            report["backends"].append(check_backend(backend, project, baseline))
            print(f"{backend}: 6 个最终任务、14 个有序事件通过", flush=True)
        report["status"] = "ok"
    finally:
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
