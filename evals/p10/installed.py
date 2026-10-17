"""新建四种用途环境，所有运行探针在项目外加载当前 wheel。"""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import httpx

from apps.client import RemoteClient
from evals.p8.installed_check import PROBE

IMPORT_PROBE = """
import json
import sys
import realtime_asr
import apps.server
import apps.client
optional = ('faster_whisper', 'sherpa_onnx', 'onnxruntime', 'pyaudio',
            'openwakeword', 'pvporcupine', 'fastapi', 'httpx')
print(json.dumps({'package': realtime_asr.__file__,
                  'optional_loaded': [name for name in optional if name in sys.modules]}))
"""

NATIVE_PROBE = """
import argparse
import json
import sys
from pathlib import Path
from dataclasses import asdict
from realtime_asr import ASRRuntime, CandidateConfig, RecognizerConfig, SessionConfig

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

p = argparse.ArgumentParser()
p.add_argument("models")
p.add_argument("audio")
p.add_argument("backend")
p.add_argument("--gpu", action="store_true")
p.add_argument("--cudnn")
p.add_argument("--cuda")
a = p.parse_args()
root = Path(a.models)
c = RecognizerConfig(
    model=root / ("whisper-tiny" if a.backend == "whisper" else "nemotron"),
    engine=a.backend,
    device="cuda" if a.gpu else "cpu",
    cudnn_folder=a.cudnn,
    cuda_directory=a.cuda,
)
import soundfile as sf

pcm, rate = sf.read(a.audio, dtype="int16")
assert rate == 16000
events = []
with ASRRuntime(c, candidate=CandidateConfig()) as r:
    s = r.create_session(
        SessionConfig(language="zh", completion_mode="candidate", retain_for_correction=True)
    )
    unit = s.start()
    s.feed_audio(pcm)
    s.request_candidate()
    e = s.get_event(timeout=60)
    assert e.kind == "candidate" and e.candidate.model_text
    events.append(asdict(e))
    s.finish()
    e = s.get_event(timeout=60)
    assert e.kind == "final" and e.optimization["candidate_reused"]
    events.append(asdict(e))
    assert s.get_event(timeout=60).kind == "completed"
    s.request_correction(unit)
    e = s.get_event(timeout=60)
    assert e.kind == "correction"
    events.append(asdict(e))
    s.close()
assert r.close_result.status == "ok"
print(
    json.dumps(
        {"backend": a.backend, "gpu": a.gpu, "events": events, "close": asdict(r.close_result)},
        ensure_ascii=False,
    )
)
"""

SERVER_PROBE = """
import json, socket, sys, threading, time, uvicorn
from pathlib import Path
from realtime_asr import RecognizerConfig, CandidateConfig, WakeConfig
from apps.server import ServerConfig, create_app

root = Path(sys.argv[1])
listener = socket.socket()
listener.bind(("127.0.0.1", 0))
listener.listen(32)
app = create_app(
    ServerConfig(
        recognizer=RecognizerConfig(model=root / "whisper-tiny"),
        vad_model=str(root / "vad/silero-v6.onnx"),
        candidate_config=CandidateConfig(),
        wake_config=WakeConfig(
            model_paths=(root / "wake/alexa_v0.1.onnx",),
            feature_model_dir=root / "wake",
            followup_window=5,
        ),
    )
)
server = uvicorn.Server(uvicorn.Config(app, log_level="warning", access_log=False))
thread = threading.Thread(target=server.run, kwargs={"sockets": [listener]})
thread.start()
print(json.dumps({"url": f"http://127.0.0.1:{listener.getsockname()[1]}"}), flush=True)
stop_file = Path(sys.argv[2])
while not stop_file.exists():
    time.sleep(0.05)
print("stop acknowledged", flush=True)
server.should_exit = True
thread.join(60)
if thread.is_alive():
    raise TimeoutError("安装包服务未退出")
listener.close()
"""


def invoke(command, *, cwd, env, timeout=180):
    result = subprocess.run(
        command, cwd=cwd, env=env, capture_output=True, text=True, encoding="utf-8", timeout=timeout
    )
    if result.returncode:
        raise AssertionError(result.stderr or result.stdout)
    return result.stdout


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outside", type=Path, required=True)
    parser.add_argument("--environments", type=Path)
    parser.add_argument("--server-only", action="store_true")
    args = parser.parse_args()
    project = Path.cwd()
    outside = args.outside.resolve()
    if outside.is_relative_to(project):
        raise ValueError("运行工作目录应在项目外")
    outside.mkdir(parents=True, exist_ok=True)
    stamp = str(time.time_ns())
    environments = (
        args.environments.resolve()
        if args.environments
        else project / ".runtime" / ("p10-installed-" + stamp)
    )
    output = project / "artifacts/p10"
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.update(
        PYTHONIOENCODING="utf-8",
        UV_CACHE_DIR=str(project / ".runtime/uv-cache"),
        UV_LINK_MODE="copy",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
    )
    wheel = project / "dist/realtime_asr-0.1.0-py3-none-any.whl"
    profiles = {
        "base": "",
        "client": "client,audio",
        "local": "whisper,nemotron,audio,vad,microphone,wake-openwakeword,wake-porcupine,models",
        "server": (
            "server,client,whisper,nemotron,audio,vad,wake-openwakeword,wake-porcupine,models"
        ),
    }
    report = {
        "new_environments": args.environments is None,
        "environment_install_retry": args.environments is not None,
        "environments": str(environments),
        "profiles": {},
    }
    if args.server_only:
        # 诊断重跑只接续同一批环境，不能借用其他安装的原生验证结果。
        checkpoint = json.loads((output / "installed-native.json").read_text(encoding="utf-8"))
        assert checkpoint["environments"] == str(environments)
        report.update(checkpoint)
    if not args.server_only:
        for name, extras in profiles.items():
            folder = environments / name
            if not folder.exists():
                invoke(
                    [
                        "uv",
                        "venv",
                        "--python",
                        str(project / ".venv/Scripts/python.exe"),
                        str(folder),
                    ],
                    cwd=project,
                    env=env,
                )
            python = folder / "Scripts/python.exe"
            requirement = str(wheel) + (f"[{extras}]" if extras else "")
            invoke(
                [
                    "uv",
                    "pip",
                    "install",
                    "--python",
                    str(python),
                    "--excludes",
                    str(project / "packaging/wake-onnx-excludes.txt"),
                    requirement,
                ],
                cwd=outside,
                env=env,
            )
            probe = invoke(
                [
                    str(python),
                    "-I",
                    "-c",
                    IMPORT_PROBE,
                ],
                cwd=outside,
                env=env,
            )
            value = json.loads(probe)
            assert Path(value["package"]).is_relative_to(folder)
            assert value["optional_loaded"] == [], value
            report["profiles"][name] = value
        (output / "installed-profiles.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        invoke(
            [str(environments / "base/Scripts/python.exe"), "-I", "-c", PROBE], cwd=outside, env=env
        )
        native = outside / "native.py"
        native.write_text(NATIVE_PROBE, encoding="utf-8")
        report["native"] = []
        for backend, gpu in [("whisper", False), ("nemotron", False), ("whisper", True)]:
            command = [
                str(environments / "local/Scripts/python.exe"),
                "-I",
                str(native),
                str(project / ".runtime/models"),
                str(project / "data/corpus/p0/C01-1.wav"),
                backend,
            ]
            if gpu:
                command += [
                    "--gpu",
                    "--cudnn",
                    str(project / ".runtime/cudnn/9.16.0.29"),
                    "--cuda",
                    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8/bin",
                ]
            report["native"].append(json.loads(invoke(command, cwd=outside, env=env)))
        (output / "installed-native.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        cli = invoke(
            [
                str(environments / "local/Scripts/python.exe"),
                "-I",
                "-m",
                "realtime_asr",
                "file",
                str(project / "data/corpus/p0/C01-1.wav"),
                "--model",
                str(project / ".runtime/models/whisper-tiny"),
                "--mode",
                "manual",
                "--candidate-only",
                "--correct-after-final",
                "--output",
                "jsonl",
            ],
            cwd=outside,
            env=env,
        )
        (output / "installed-local-cli.jsonl").write_text(cli, encoding="utf-8")
        assert '"kind": "correction"' in cli
    server_script = outside / "server.py"
    server_script.write_text(SERVER_PROBE, encoding="utf-8")
    stop_file = outside / "stop-server"
    stop_file.unlink(missing_ok=True)
    log = (output / "installed-server.log").open("wb")
    process = subprocess.Popen(
        [
            str(environments / "server/Scripts/python.exe"),
            "-I",
            str(server_script),
            str(project / ".runtime/models"),
            str(stop_file),
        ],
        cwd=outside,
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=log,
        text=True,
        encoding="utf-8",
    )
    try:
        url = json.loads(process.stdout.readline())["url"]
        deadline = time.monotonic() + 60
        with httpx.Client(base_url=url, timeout=5, trust_env=False) as http:
            while True:
                try:
                    ready = http.get("/api/v1/ready")
                    (output / "installed-readiness.json").write_text(ready.text, encoding="utf-8")
                    if ready.status_code == 200:
                        break
                    if ready.json().get("phase") == "failed":
                        raise AssertionError(ready.json())
                except httpx.ConnectError as exc:
                    (output / "installed-connect-error.txt").write_text(str(exc), encoding="utf-8")
                if time.monotonic() > deadline:
                    raise TimeoutError("新安装服务未就绪")
                time.sleep(0.05)
        with RemoteClient(url, timeout=60) as client:
            result = client.transcribe_file(project / "data/corpus/p0/C01-1.wav", language="zh")
            assert result["result"]["text"]
        remote = invoke(
            [
                str(environments / "client/Scripts/python.exe"),
                "-I",
                "-m",
                "apps.client",
                "--url",
                url,
                "--output",
                "jsonl",
                "file",
                str(project / "data/corpus/p0/C01-1.wav"),
            ],
            cwd=outside,
            env=env,
        )
        (output / "installed-remote-cli.jsonl").write_text(remote, encoding="utf-8")
        invoke(
            [
                "node",
                str(project / "evals/p7/browser.mjs"),
                url,
                str(output),
                str(project / "artifacts/p9/combined.wav"),
                "wake",
            ],
            cwd=project,
            env=env,
            timeout=120,
        )
        report["installed_server_client_browser"] = True
    finally:
        stop_file.touch()
        process.wait(60)
        log.close()
        if process.returncode:
            raise AssertionError("新安装服务异常退出")
    report["server_exit_code"] = process.returncode
    saved_script = outside / "save_audio.py"
    saved_script.write_bytes((project / "examples/save_audio.py").read_bytes())
    saved_output = outside / "saved.wav"
    saved = invoke(
        [
            str(environments / "local/Scripts/python.exe"),
            "-I",
            "-X",
            "utf8",
            str(saved_script),
            str(project / ".runtime/models/whisper-tiny"),
            str(project / "data/corpus/p0/C01-1.wav"),
            str(saved_output),
        ],
        cwd=outside,
        env=env,
    )
    import wave

    with wave.open(str(saved_output), "rb") as recording:
        assert recording.getparams()[:3] == (1, 2, 16000)
        report["saved_samples"] = recording.getnframes()
    (output / "installed-save.txt").write_text(saved, encoding="utf-8")
    (output / "installed.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("四个全新环境、项目外 CPU/GPU、CLI、客户端、服务与浏览器验证通过")


if __name__ == "__main__":
    main()
