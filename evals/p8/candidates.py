"""P8 候选使用真实模型和端口验证；输入计量仅装饰外部引擎边界。"""

import hashlib
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np
import soundfile as sf

from evals.p6.network import running_server
from evals.p8.dual import MeasuredEngine
from realtime_asr import ASRRuntime, CandidateConfig, RecognizerConfig, SessionConfig
from realtime_asr.client import RemoteClient
from realtime_asr.server import ServerConfig

OUTPUT = Path("artifacts/p8-candidates")


def config():
    return RecognizerConfig(model=Path(".runtime/models/whisper-tiny").resolve())


def sdk_case(pcm, language, model=None):
    """一个单元验证固定尾部、完整历史、继续、候选完成复用与修正。"""
    engine = MeasuredEngine(model or config())
    rows = []
    with ASRRuntime(engine=engine, candidate=CandidateConfig()) as runtime:
        session = runtime.create_session(
            SessionConfig(
                language=language, completion_mode="candidate", retain_for_correction=True
            )
        )
        unit = session.start()
        split = len(pcm) // 2
        session.feed_audio(pcm[:split])
        tail = session.request_candidate(kind="tail")
        value = session.get_event(timeout=60)
        assert value.kind == "candidate", asdict(value)
        assert engine.whole[-1]["samples"] == tail.end_sample - tail.start_sample
        rows.append(asdict(value))
        ack = session.resume(expected_audio_samples=split)
        session.feed_audio(pcm[split:])
        whole = session.request_candidate()
        value = session.get_event(timeout=60)
        assert value.kind == "candidate" and value.candidate.current
        assert whole.resume_epoch == ack.resume_epoch and whole.end_sample == len(pcm)
        expected = hashlib.sha256(
            (pcm.astype(np.float32) / 32768).astype("<f4").tobytes()
        ).hexdigest()
        assert engine.whole[-1]["digest"] == expected
        rows.append(asdict(value))
        before = len(engine.whole)
        session.finish()
        final = session.get_event(timeout=60)
        assert final.kind == "final" and final.result.text == value.candidate.model_text
        assert final.optimization["candidate_reused"] and len(engine.whole) == before
        assert session.get_event(timeout=60).kind == "completed"
        rows.append(asdict(final))
        request = session.request_correction(unit)
        correction = session.get_event(timeout=60)
        assert (
            correction.kind == "correction"
            and correction.candidate.replaces_task_id == final.task_id
        )
        assert request.end_sample == len(pcm) and engine.whole[-1]["digest"] == expected
        rows.append(asdict(correction))
        session.start()
        assert runtime.capacity.retained_audio_samples == 0
        session.cancel()
    assert engine.close_result.status == "ok"
    return {
        "events": rows,
        "actual_model_inputs": engine.whole,
        "close": asdict(engine.close_result),
    }


def network_case(pcm, filename):
    """真实服务、SDK 和文件 CLI 通过同一宿主候选角色，不在客户端配模型。"""
    rows = []
    with running_server(
        ServerConfig(recognizer=config(), candidate_config=CandidateConfig())
    ) as url:
        with RemoteClient(url, timeout=60) as client:
            whole = client.transcribe(pcm.tobytes(), language="zh")
            session = client.connect(
                mode="manual",
                language="zh",
                realtime=False,
                completion_mode="candidate",
                retain_for_correction=True,
            )
            assert session.candidate_available
            unit = session.start()["unit"]
            session.feed_audio(pcm.tobytes())
            session.request_candidate()
            while True:
                message = session.get_message(timeout=60)
                rows.append(message)
                if message.get("event", {}).get("kind") == "candidate":
                    break
            session.resume()
            session.finish()
            while True:
                message = session.get_message(timeout=60)
                rows.append(message)
                if message.get("event", {}).get("kind") == "completed":
                    break
            session.request_correction(unit)
            while True:
                message = session.get_message(timeout=60)
                rows.append(message)
                if message.get("event", {}).get("kind") == "correction":
                    break
            session.close()
            assert any(m.get("event", {}).get("kind") == "final" for m in rows)
        command = [
            sys.executable,
            "-m",
            "realtime_asr.client",
            "--url",
            url,
            "--output",
            "jsonl",
            "--language",
            "zh",
            "file",
            str(filename),
            "--mode",
            "manual",
            "--candidates",
            "--completion-mode",
            "candidate",
            "--correct-after-final",
        ]
        result = subprocess.run(
            command, capture_output=True, text=True, encoding="utf-8", timeout=90
        )
        (OUTPUT / "remote-cli.jsonl").write_text(result.stdout, encoding="utf-8")
        if result.returncode:
            raise AssertionError(result.stderr)
        events = [json.loads(line).get("event", {}) for line in result.stdout.splitlines()]
        assert any(event.get("kind") == "correction" for event in events), result.stdout
        assert sum(event.get("kind") == "final" for event in events) == 1
        browser = subprocess.run(
            [
                "node",
                "evals/p7/browser.mjs",
                url,
                str(OUTPUT.resolve()),
                str(filename),
                "candidates",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=90,
        )
        if browser.returncode:
            raise AssertionError(browser.stderr)
    return {"http": whole, "messages": rows, "remote_cli_exit": result.returncode}


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    report = {"model": asdict(config()), "cases": {}}
    for language, name in (("zh", "C01-1.wav"), ("en", "C07-1.wav")):
        pcm, rate = sf.read(Path("data/corpus/p0") / name, dtype="int16")
        assert rate == 16000 and pcm.ndim == 1
        started = perf_counter()
        report["cases"][language] = sdk_case(pcm, language)
        report["cases"][language]["elapsed_seconds"] = perf_counter() - started
    pcm, _ = sf.read("data/corpus/p0/C01-1.wav", dtype="int16")
    report["cases"]["nemotron-zh"] = sdk_case(
        pcm,
        "zh",
        RecognizerConfig(model=Path(".runtime/models/nemotron").resolve(), engine="nemotron"),
    )
    report["network"] = network_case(pcm, Path("data/corpus/p0/C01-1.wav").resolve())
    (OUTPUT / "actual.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print("真实中文、英文候选／复用／修正与网络、远程 CLI 通过")


if __name__ == "__main__":
    main()
