"""显式运行真实模型矩阵，保留 SDK 对照与关闭证据。"""

import argparse
import json
import socket
import ssl
import subprocess
import sys
from pathlib import Path
from time import monotonic, sleep

import numpy as np
import soundfile as sf

from apps.client import RemoteClient
from apps.server import ServerConfig
from evals.p6.network import running_server
from realtime_asr import (
    ASRRuntime,
    AudioFormat,
    Recognizer,
    RecognizerConfig,
    SessionConfig,
    VadConfig,
)


def gather(session):
    """正常结束后收取所有消息；失败和序号倒退使验证失败。"""
    session.close()
    messages = []
    while message := session.get_message(timeout=0):
        if message["type"] == "error" or message.get("event", {}).get("error"):
            raise AssertionError(message)
        messages.append(message)
    wires = [m["wire_sequence"] for m in messages]
    if wires != sorted(set(wires)) or messages[-1]["type"] != "closed":
        raise AssertionError("传输顺序或正常关闭不成立")
    return messages


def replay(session, pcm):
    for offset in range(0, len(pcm), 317):
        session.feed_audio(pcm[offset : offset + 317].tobytes())


def auto_baseline(config, vad, pcm, language, mode):
    with ASRRuntime(config) as runtime:
        session = runtime.create_auto_session(
            config=SessionConfig(language=language), vad=VadConfig(model_path=vad, webrtc_mode=mode)
        )
        for offset in range(0, len(pcm), 317):
            session.feed_audio(pcm[offset : offset + 317])
        session.finish_input()
        finals = []
        while session.pending_segments:
            event = session.get_event(timeout=30)
            if event is None or event.error is not None:
                raise AssertionError(event)
            if event.kind == "final":
                finals.append(
                    (event.result.text, event.boundary.start_sample, event.boundary.end_sample)
                )
        return finals


def matrix(project, folder):
    vad = project / ".runtime/models/vad/silero-v6.onnx"
    cpu = RecognizerConfig(model=project / ".runtime/models/whisper-tiny")
    cuda = RecognizerConfig(
        model=cpu.model, device="cuda", cudnn_folder=project / ".runtime/cudnn/9.16.0.29"
    )
    native = RecognizerConfig(model=project / ".runtime/models/nemotron", engine="nemotron")
    reports = []
    for name, config in (("whisper-cpu", cpu), ("whisper-cuda", cuda), ("nemotron", native)):
        server_config = ServerConfig(recognizer=config, vad_model=str(vad))
        with running_server(server_config) as url, RemoteClient(url) as client:
            with Recognizer(config) as recognizer:
                for sample_id, language, mode in (("C01-1", "zh-CN", 3), ("C07-1", "en", 0)):
                    pcm, rate = sf.read(project / f"data/corpus/p0/{sample_id}.wav", dtype="int16")
                    assert rate == 16000
                    expected = recognizer.transcribe(
                        pcm, format=AudioFormat(), language=language
                    ).text
                    http = client.transcribe(pcm.tobytes(), language=language)
                    assert http["result"]["text"] == expected
                    with client.connect(realtime=True, language=language) as session:
                        unit = session.start()["unit"]
                        replay(session, pcm)
                        session.finish()
                        messages = gather(session)
                    finals = [
                        m["event"] for m in messages if m.get("event", {}).get("kind") == "final"
                    ]
                    assert len(finals) == 1 and finals[0]["result"]["text"] == expected
                    assert finals[0]["unit"] == unit and finals[0]["audio_samples"] == len(pcm)
                    expected_auto = auto_baseline(config, vad, pcm, language, mode)
                    with client.connect(
                        mode="auto", realtime=True, language=language, webrtc_mode=mode
                    ) as session:
                        replay(session, pcm)
                        session.end_input()
                        automatic = gather(session)
                    actual_auto = [
                        (
                            m["event"]["result"]["text"],
                            m["event"]["boundary"]["start_sample"],
                            m["event"]["boundary"]["end_sample"],
                        )
                        for m in automatic
                        if m.get("event", {}).get("kind") == "final"
                    ]
                    assert expected_auto and actual_auto == expected_auto
                    reports.append(
                        {
                            "configuration": name,
                            "sample_id": sample_id,
                            "matches_sdk": True,
                            "http": http,
                            "manual": messages,
                            "automatic": automatic,
                        }
                    )
                    print(name, sample_id, "HTTP/manual/auto: passed", flush=True)
                # 声道与采样率在同一条连续输入中转换，remote CLI 另走正式入口。
                pcm, _ = sf.read(project / "data/corpus/p0/C01-1.wav", dtype="int16")
                stereo = np.column_stack((np.repeat(pcm, 3), np.repeat(pcm, 3)))
                expected = recognizer.transcribe(
                    stereo, format=AudioFormat(48000, 2), language="zh-CN"
                ).text
                assert (
                    client.transcribe(
                        stereo.tobytes(), format=AudioFormat(48000, 2), language="zh-CN"
                    )["result"]["text"]
                    == expected
                )
            with client.connect(realtime=False) as first, client.connect(realtime=False) as second:
                assert first.session_id != second.session_id
                first.start()
                first.feed_audio(b"\x01\x00" * 515)
                first.close(graceful=False)
                assert second.ping()["type"] == "pong"
                second.start()
                second.feed_audio(b"\x00\x00" * 515)
                second.finish()
                gather(second)
            deadline = monotonic() + 5
            while monotonic() < deadline:
                metrics = client.http.get("/api/v1/metrics").json()
                if metrics["sessions_open"] == metrics["sessions_reserved"] == 0:
                    break
                sleep(0.01)
            assert metrics["sessions_open"] == 0
            if name == "whisper-cpu":
                cli_checks(project, url, pcm, folder)
                browser_checks(project, url, folder)
            reports.append(
                {
                    "configuration": name,
                    "stereo_48k_matches_sdk": True,
                    "two_sessions_disconnect_isolated": True,
                    "metrics": metrics,
                }
            )
    return reports


def cli_checks(project, url, pcm, folder):
    prefix = [
        sys.executable,
        "-m",
        "realtime_asr",
        "remote",
        "--url",
        url,
        "--output",
        "jsonl",
        "--language",
        "zh-CN",
    ]
    commands = [
        ("file", [str(project / "data/corpus/p0/C01-1.wav")], None),
        ("stdin", [], pcm.tobytes()),
        ("mic", ["--mode", "manual", "--duration", "1"], None),
    ]
    records = []
    for source, args, audio in commands:
        result = subprocess.run(
            [*prefix, source, *args], input=audio, capture_output=True, timeout=60, cwd=project
        )
        (folder / f"cli-{source}.stdout").write_bytes(result.stdout)
        (folder / f"cli-{source}.stderr").write_bytes(result.stderr)
        assert result.returncode == 0, (source, result.returncode, result.stdout, result.stderr)
        messages = [json.loads(line) for line in result.stdout.splitlines()]
        assert any(
            m["type"] == "transcription" or m.get("event", {}).get("kind") == "final"
            for m in messages
        )
        records.append({"source": source, "exit_code": result.returncode, "messages": messages})
    (folder / "remote-cli.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def browser_checks(project, url, folder):
    result = subprocess.run(
        [
            "node",
            str(project / "evals/p6/browser.mjs"),
            url,
            str(project / "data/corpus/p0/C01-1.wav"),
            str(folder),
        ],
        cwd=project,
        capture_output=True,
        timeout=90,
    )
    (folder / "browser.log").write_bytes(result.stdout + result.stderr)
    assert result.returncode == 0, result.stderr.decode("utf-8")


def tls_check(project, folder):
    """局部自签名证书由客户端显式信任，不使用 verify=False。"""
    cert, key = folder / "localhost.crt", folder / "localhost.key"
    openssl_config = folder / "openssl.cnf"
    openssl_config.write_text(
        "[req]\ndistinguished_name=dn\n[dn]\nCN=localhost\n", encoding="utf-8"
    )
    command = [
        "openssl",
        "req",
        "-config",
        str(openssl_config),
        "-x509",
        "-newkey",
        "rsa:2048",
        "-nodes",
        "-days",
        "1",
        "-keyout",
        str(key),
        "-out",
        str(cert),
        "-subj",
        "/CN=localhost",
        "-addext",
        "subjectAltName=IP:127.0.0.1,DNS:localhost",
    ]
    prepared = subprocess.run(command, capture_output=True, timeout=30)
    if prepared.returncode:
        raise AssertionError(prepared.stderr.decode("utf-8", errors="replace"))
    trust = ssl.create_default_context(cafile=str(cert))
    config = ServerConfig(
        recognizer=RecognizerConfig(model=project / ".runtime/models/whisper-tiny"),
        realtime=False,
        token="eval-only-token",
        ssl_certfile=str(cert),
        ssl_keyfile=str(key),
    )
    with running_server(config, verify=trust) as url:
        with RemoteClient(url, token=config.token, verify=trust) as client:
            assert client.transcribe(b"\x00\x00" * 515)["result"]["text"] == ""
            with client.connect(realtime=False) as session:
                assert session.ping()["type"] == "pong"
                gather(session)
        with RemoteClient(url, verify=trust) as missing:
            assert missing.http.get("/api/v1/ready").status_code == 401
    return {
        "https": True,
        "wss": True,
        "certificate_verified": True,
        "auth_required": True,
        "shutdown": "ok",
    }


def upload_limits(project):
    """真实 HTTP 限制与断开，不用模型替身模拟网络接入。"""
    config = ServerConfig(
        recognizer=RecognizerConfig(model=project / ".runtime/models/whisper-tiny"),
        realtime=False,
        max_http_bytes=1024,
        body_timeout=0.1,
    )
    from urllib.parse import urlsplit

    with running_server(config) as url, RemoteClient(url) as client:
        large = client.http.post(
            "/api/v1/transcribe",
            content=bytes(1025),
            headers={"Content-Type": "application/octet-stream"},
        )
        assert large.status_code == 413
        address = urlsplit(url)
        with socket.create_connection((address.hostname, address.port), timeout=5) as transport:
            transport.sendall(
                b"POST /api/v1/transcribe HTTP/1.1\r\nHost: localhost\r\n"
                b"Content-Type: application/octet-stream\r\nContent-Length: 1024\r\n"
                b"Connection: close\r\n\r\n\x00"
            )
            response = b""
            while chunk := transport.recv(8192):
                response += chunk
            assert b"408 Request Timeout" in response and b"HTTP_BODY_TIMEOUT" in response
        with socket.create_connection((address.hostname, address.port), timeout=5) as transport:
            transport.sendall(
                b"POST /api/v1/transcribe HTTP/1.1\r\nHost: localhost\r\n"
                b"Content-Type: application/octet-stream\r\nContent-Length: 1024\r\n\r\n\x00"
            )
        deadline = monotonic() + 5
        while monotonic() < deadline:
            if client.http.get("/api/v1/metrics").json()["sessions_open"] == 0:
                break
            sleep(0.01)
        assert client.http.get("/api/v1/metrics").json()["sessions_open"] == 0
        assert client.http.get("/api/v1/ready").status_code == 200
    return {"oversize": 413, "partial_upload_timeout": 408, "upload_disconnect_released": True}


def paced_realtime(project, folder):
    """按源音频时长递交，验证 Whisper 两种设备真正交付实时观察。"""
    pcm, _ = sf.read(project / "data/corpus/p0/C01-1.wav", dtype="int16")
    baseline = json.loads((folder / "model-matrix.json").read_text(encoding="utf-8"))
    reports = []
    for device in ("cpu", "cuda"):
        config = RecognizerConfig(
            model=project / ".runtime/models/whisper-tiny",
            device=device,
            cudnn_folder=project / ".runtime/cudnn/9.16.0.29" if device == "cuda" else None,
        )
        with running_server(ServerConfig(recognizer=config)) as url, RemoteClient(url) as client:
            with client.connect(realtime=True, language="zh-CN") as session:
                session.start()
                started = monotonic()
                for offset in range(0, len(pcm), 1600):
                    block = pcm[offset : offset + 1600]
                    session.feed_audio(block.tobytes())
                    # 实际源时间决定下一块时间，不以模型返回时间当采集节奏。
                    sleep(max(0, started + (offset + len(block)) / 16000 - monotonic()))
                session.finish()
                messages = gather(session)
        previews = [m["event"] for m in messages if m.get("event", {}).get("kind") == "realtime"]
        assert previews and any(event["result"]["text"] for event in previews)
        final = next(m["event"] for m in messages if m.get("event", {}).get("kind") == "final")
        expected = next(
            row["http"]["result"]["text"]
            for row in baseline["model_matrix"]
            if row.get("configuration") == "whisper-" + device and row.get("sample_id") == "C01-1"
        )
        assert final["result"]["text"] == expected and final["audio_samples"] == len(pcm)
        reports.append(
            {
                "device": device,
                "previews": len(previews),
                "final_matches_sdk": True,
                "messages": messages,
            }
        )
        print("whisper", device, "paced realtime: passed", flush=True)
    (folder / "realtime-paced.json").write_text(
        json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return reports


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts/p6/real-check"))
    args = parser.parse_args()
    project, folder = Path.cwd(), args.output.resolve()
    folder.mkdir(parents=True, exist_ok=True)
    report = {"model_matrix": matrix(project, folder)}
    (folder / "model-matrix.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report.update(
        {
            "tls": tls_check(project, folder),
            "upload_limits": upload_limits(project),
            "paced_realtime": paced_realtime(project, folder),
            "not_verified": [
                "人工口述质量",
                "其他操作系统",
                "正式反向代理与容器组合",
                "P0 正式语料评分",
            ],
        }
    )
    (folder / "verification.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("P6 real network verification passed", flush=True)


if __name__ == "__main__":
    main()
