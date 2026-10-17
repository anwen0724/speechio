"""实际 Docker 服务：只读卷、TLS、鉴权、会话、健康与正常退出。"""

import json
import secrets
import socket
import ssl
import subprocess
import time
from pathlib import Path

import httpx
import soundfile as sf

from apps.client import RemoteClient

PACKAGES_PROBE = """
import json
import importlib.metadata as metadata
names = ('realtime-asr', 'faster-whisper', 'sherpa-onnx', 'onnxruntime', 'openwakeword')
print(json.dumps({name: metadata.version(name) for name in names}))
"""


def docker(*arguments, timeout=120):
    result = subprocess.run(
        ["docker", *arguments], capture_output=True, text=True, encoding="utf-8", timeout=timeout
    )
    if result.returncode:
        raise RuntimeError(result.stderr)
    return result.stdout.strip()


def main():
    project = Path.cwd()
    output = project / "artifacts/p10"
    tls = project / ".runtime/p10-tls"
    tls.mkdir(parents=True, exist_ok=True)
    config = tls / "openssl.cnf"
    config.write_text("[req]\ndistinguished_name=dn\n[dn]\nCN=localhost\n", encoding="utf-8")
    cert, key = tls / "server.crt", tls / "server.key"
    subprocess.run(
        [
            "openssl",
            "req",
            "-config",
            str(config),
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
        ],
        capture_output=True,
        check=True,
        timeout=30,
    )
    token = secrets.token_urlsafe(32)
    env_file = tls / "runtime.env"
    env_file.write_text("REALTIME_ASR_TOKEN=" + token + "\n", encoding="utf-8")
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    name = "realtime-asr-p10-" + secrets.token_hex(4)
    container = docker(
        "run",
        "-d",
        "--name",
        name,
        "--env-file",
        str(env_file),
        "-p",
        f"127.0.0.1:{port}:8000",
        "--mount",
        f"type=bind,src={project / '.runtime/models'},dst=/models,readonly",
        "--mount",
        f"type=bind,src={tls},dst=/run/asr-tls,readonly",
        "realtime-asr:p10",
        "--model",
        "/models/whisper-tiny",
        "--vad-model",
        "/models/vad/silero-v6.onnx",
        "--host",
        "0.0.0.0",
        "--ssl-certfile",
        "/run/asr-tls/server.crt",
        "--ssl-keyfile",
        "/run/asr-tls/server.key",
        "--candidates",
        "--wake",
        "--wake-models",
        "/models/wake/alexa_v0.1.onnx",
        "--wake-feature-dir",
        "/models/wake",
        "--wake-followup-window",
        "5",
    )
    url = f"https://127.0.0.1:{port}"
    trust = ssl.create_default_context(cafile=str(cert))
    report = {"container": container, "url": url, "tls_certificate_verified": True, "profiles": []}
    try:
        with httpx.Client(
            base_url=url,
            verify=trust,
            trust_env=False,
            timeout=5,
            headers={"Authorization": "Bearer " + token},
        ) as http:
            deadline = time.monotonic() + 90
            while True:
                try:
                    response = http.get("/api/v1/ready")
                    if response.status_code == 200:
                        break
                    if response.json().get("phase") == "failed":
                        raise AssertionError(response.json())
                except httpx.ConnectError:
                    pass
                if time.monotonic() > deadline:
                    raise TimeoutError(docker("logs", container))
                time.sleep(0.1)
            report["capabilities"] = http.get("/api/v1/capabilities").json()
        with httpx.Client(base_url=url, verify=trust, trust_env=False) as unauthorized:
            assert unauthorized.get("/api/v1/capabilities").status_code == 401
        pcm, rate = sf.read(project / "data/corpus/p0/C01-1.wav", dtype="int16")
        assert rate == 16000
        with RemoteClient(url, token=token, verify=trust, timeout=60) as client:
            whole = client.transcribe(pcm.astype("<i2").tobytes(), language="zh")
            assert whole["result"]["text"]
            sessions = [
                client.connect(
                    mode="manual", language="zh", realtime=False, completion_mode="candidate"
                )
                for _ in range(2)
            ]
            units = [session.start()["unit"] for session in sessions]
            assert units[0]["session_id"] != units[1]["session_id"]
            for session in sessions:
                for _ in range(3):
                    session.feed_audio(pcm.astype("<i2").tobytes())
                session.finish()
            for session, unit in zip(sessions, units, strict=True):
                events = []
                while not any(item.get("event", {}).get("kind") == "completed" for item in events):
                    message = session.get_message(timeout=60)
                    assert message is not None
                    if message["type"] == "event":
                        assert message["event"]["unit"] == unit
                        assert message["event"]["error"] is None
                    events.append(message)
                final = next(
                    item["event"] for item in events if item.get("event", {}).get("kind") == "final"
                )
                assert final["audio_samples"] == len(pcm) * 3
                report["profiles"].append(
                    {
                        "unit": unit,
                        "final_samples": final["audio_samples"],
                        "text": final["result"]["text"],
                    }
                )
                session.close()
            # 以实际音频时钟送入，期间调用方不消费预览；SDK 的接收仍独立运行。
            slow = client.connect(mode="manual", language="zh", realtime=True)
            slow_unit = slow.start()["unit"]
            started = time.monotonic()
            for offset in range(0, len(pcm), 1600):
                slow.feed_audio(pcm[offset : offset + 1600].astype("<i2").tobytes())
                due = started + min(offset + 1600, len(pcm)) / rate
                time.sleep(max(0, due - time.monotonic()))
            slow.finish()
            delayed = []
            while not any(item.get("event", {}).get("kind") == "completed" for item in delayed):
                message = slow.get_message(timeout=60)
                assert message is not None
                if message["type"] == "event":
                    assert message["event"]["unit"] == slow_unit
                    assert message["event"]["error"] is None
                delayed.append(message)
            slow_final = next(
                item["event"] for item in delayed if item.get("event", {}).get("kind") == "final"
            )
            assert slow_final["audio_samples"] == len(pcm)
            report["paused_application_consumer"] = {
                "input_seconds": len(pcm) / rate,
                "messages_after_pause": len(delayed),
                "final_samples": slow_final["audio_samples"],
                "socket_backpressure_injected": False,
            }
            slow.close()
            report["metrics"] = client.http.get("/api/v1/metrics").json()
        assert docker("exec", container, "python", "/app/healthcheck.py") == ""
        inspect = json.loads(docker("inspect", container))[0]
        report["runtime_user"] = inspect["Config"]["User"]
        report["readonly_mounts"] = [
            {"destination": item["Destination"], "readonly": not item["RW"]}
            for item in inspect["Mounts"]
        ]
        assert report["runtime_user"] == "asr" and all(
            item["readonly"] for item in report["readonly_mounts"]
        )
        report["image"] = inspect["Image"]
        report["python"] = docker("exec", container, "python", "--version")
        report["packages"] = json.loads(
            docker(
                "exec",
                container,
                "python",
                "-c",
                PACKAGES_PROBE,
            )
        )
        report["uv"] = docker("exec", container, "uv", "--version")
        docker("stop", "--time", "30", container)
        stopped = json.loads(docker("inspect", container))[0]
        report["exit_code"] = stopped["State"]["ExitCode"]
        report["oom_killed"] = stopped["State"]["OOMKilled"]
        assert report["exit_code"] == 0 and not report["oom_killed"]
        (output / "docker-server.log").write_text(docker("logs", container), encoding="utf-8")
        (output / "deployment.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print("Linux CPU 镜像、只读卷、TLS、鉴权、两会话长输入与正常停止通过")
    finally:
        docker("rm", "-f", container)
        env_file.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
