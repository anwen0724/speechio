"""实际本地唤醒、标准音频、网络及外围入口联合验证。"""

import json
import subprocess
import sys
import wave
from dataclasses import asdict
from pathlib import Path

import numpy as np
import soundfile as sf

from apps.client import RemoteClient
from apps.server import ServerConfig
from evals.p6.network import running_server
from realtime_asr import (
    ASRError,
    ASRRuntime,
    RecognizerConfig,
    SessionConfig,
    VadConfig,
    WakeConfig,
)
from realtime_asr.segmentation.wakeword import LocalWakeDetector

OUTPUT = Path("artifacts/p9")
MODELS = Path(".runtime/models/wake").resolve()
ASR = RecognizerConfig(model=Path(".runtime/models/whisper-tiny").resolve())
VAD = Path(".runtime/models/vad/silero-v6.onnx").resolve()


def wake_config():
    return WakeConfig(
        model_paths=(MODELS / "alexa_v0.1.onnx",), feature_model_dir=MODELS, followup_window=5
    )


def backend_case(keyword):
    detector = LocalWakeDetector(wake_config())
    try:
        hits = [
            detector.detect(keyword[i : i + 1280].astype("<i2").tobytes())
            for i in range(0, len(keyword) - 1279, 1280)
        ]
        assert "alexa_v0.1" in hits
        detector.reset()
        negative = [detector.detect(bytes(2560)) for _ in range(20)]
        assert all(item is None for item in negative)
    finally:
        detector.close()
    try:
        LocalWakeDetector(WakeConfig(backend="porcupine", keywords=("porcupine",)))
    except ASRError as exc:
        porcupine = {"code": exc.code, "positive_accepted": False}
        assert exc.code == "WAKE_ACCESS_KEY_REQUIRED"
    else:
        raise AssertionError("本验证需显式扩展带密钥的真实 Porcupine 用例")
    return {"openwakeword_hits": hits, "negative": negative, "porcupine": porcupine}


def sdk_case(pcm):
    observations, events = [], []
    with ASRRuntime(ASR) as runtime:
        session = runtime.create_auto_session(
            config=SessionConfig(language="en"), vad=VadConfig(model_path=VAD), wake=wake_config()
        )
        assert session.wake_status["state"] == "awaiting_wake"
        for offset in range(0, len(pcm), 512):
            session.feed_audio(pcm[offset : offset + 512])
            state = session.wake_status
            if not observations or state["state"] != observations[-1]["state"]:
                observations.append(state)
        session.finish_input()
        while session.pending_segments:
            event = session.get_event(timeout=60)
            assert event is not None and event.error is None
            events.append(asdict(event))
        finals = [event for event in events if event["kind"] == "final"]
        assert finals and any(item["result"]["text"].strip() for item in finals)
        assert any("satellite" in item["result"]["text"].lower() for item in finals), finals
        assert any(item["state"] == "recording" for item in observations)
        assert session.wake_status["released"]
    return {"observations": observations, "events": events, "close": asdict(runtime.close_result)}


def run_command(command, name):
    result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", timeout=120)
    (OUTPUT / name).write_text(result.stdout, encoding="utf-8")
    if result.returncode:
        raise AssertionError(result.stderr)
    return result.stdout


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    keyword, rate = sf.read(OUTPUT / "alexa_test.wav", dtype="int16")
    assert rate == 16000
    body, rate = sf.read("data/corpus/p0/C07-1.wav", dtype="int16")
    assert rate == 16000
    # 此公开样本 RMS 为约 0.0032；场景联调显式加增益，不改变生产 VAD。
    body = (body.astype(np.int32) * 8).astype(np.int16)
    pcm = np.concatenate(
        (
            np.zeros(16000, np.int16),
            keyword,
            np.zeros(2560, np.int16),
            body,
            np.zeros(16000, np.int16),
        )
    )
    filename = (OUTPUT / "combined.wav").resolve()
    sf.write(filename, pcm, 16000, subtype="PCM_16")
    report = {
        "body_gain": 8,
        "formal_quality_accepted": False,
        "backend": backend_case(keyword),
        "sdk": sdk_case(pcm),
    }
    common = ["--mode", "auto", "--wake", "--output", "jsonl", "--language", "en"]
    local = run_command(
        [
            sys.executable,
            "-m",
            "realtime_asr",
            "file",
            str(filename),
            "--model",
            str(ASR.model),
            "--vad-model",
            str(VAD),
            "--wake-models",
            str(MODELS / "alexa_v0.1.onnx"),
            "--wake-feature-dir",
            str(MODELS),
            "--wake-followup-window",
            "5",
            *common,
        ],
        "local-cli.jsonl",
    )
    assert '"record_type": "event"' in local and '"wake_status"' in local
    with running_server(
        ServerConfig(recognizer=ASR, vad_model=str(VAD), wake_config=wake_config())
    ) as url:
        with RemoteClient(url, timeout=60) as client:
            session = client.connect(mode="auto", language="en", realtime=False, wake=True)
            session.feed_audio(pcm.astype("<i2").tobytes())
            session.end_input()
            messages = []
            while not any(item.get("event", {}).get("kind") == "completed" for item in messages):
                messages.append(session.get_message(timeout=60))
            session.close()
            assert any(item["type"] == "state" for item in messages)
            report["network"] = messages
        remote = run_command(
            [
                sys.executable,
                "-m",
                "apps.client",
                "--url",
                url,
                "--output",
                "jsonl",
                "--language",
                "en",
                "file",
                str(filename),
                "--mode",
                "auto",
                "--wake",
            ],
            "remote-cli.jsonl",
        )
        assert '"state"' in remote
        run_command(
            ["node", "evals/p7/browser.mjs", url, str(OUTPUT.resolve()), str(filename), "wake"],
            "browser-log.txt",
        )
    saved = (OUTPUT / "saved.wav").resolve()
    run_command(
        [sys.executable, "examples/save_audio.py", str(ASR.model), str(filename), str(saved)],
        "save-log.txt",
    )
    with wave.open(str(saved), "rb") as recording:
        assert (
            recording.getnchannels(),
            recording.getsampwidth(),
            recording.getframerate(),
            recording.getnframes(),
        ) == (1, 2, 16000, len(pcm))
        assert recording.readframes(len(pcm)) == pcm.astype("<i2").tobytes()
    report["saved_samples"] = len(pcm)
    report["manual_device_and_hotkeys_accepted"] = False
    (OUTPUT / "actual.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print("OpenWakeWord 正反检测、实际自动识别、网络、CLI、Chrome 与标准音频保存通过")


if __name__ == "__main__":
    main()
