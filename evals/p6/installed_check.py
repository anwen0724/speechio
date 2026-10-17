"""从项目外验证 wheel 的基础、客户端及服务安装边界。"""

import argparse
import json
import os
import subprocess
import time
import wave
import zipfile
from pathlib import Path

import httpx

SERVER_PROGRAM = """import json,socket,sys,threading,time
import uvicorn
from realtime_asr import RecognizerConfig
from apps.server import ServerConfig,create_app
listener=socket.socket()
listener.bind(("127.0.0.1",0))
listener.listen(32)
app=create_app(ServerConfig(recognizer=RecognizerConfig(model=sys.argv[1]),vad_model=sys.argv[2]))
server=uvicorn.Server(uvicorn.Config(app,log_level="warning",access_log=False))
thread=threading.Thread(target=server.run,kwargs={"sockets":[listener]})
thread.start()
while not server.started:
    if not thread.is_alive(): raise RuntimeError("服务未启动")
    time.sleep(.05)
print(json.dumps({"url":"http://127.0.0.1:"+str(listener.getsockname()[1])}),flush=True)
try:
    sys.stdin.readline()
finally:
    server.should_exit=True
    thread.join(60)
    listener.close()
    if thread.is_alive(): raise RuntimeError("服务未关闭")
"""

CLIENT_PROGRAM = """import json,sys
from apps.client import RemoteClient
with RemoteClient(sys.argv[1]) as client:
    with client.connect(realtime=False) as session:
        session.start()
        session.feed_audio(bytes(1030))
        session.finish()
        session.close()
        messages=[]
        while message:=session.get_message(timeout=0): messages.append(message)
        assert any(m.get("event",{}).get("kind")=="completed" for m in messages)
        assert messages[-1]["type"]=="closed"
        print(json.dumps(messages))
"""


def invoke(command, outside, env, *, input=None):
    result = subprocess.run(
        command, cwd=outside, env=env, input=input, capture_output=True, timeout=60
    )
    if result.returncode:
        raise AssertionError((command, result.returncode, result.stderr.decode("utf-8")))
    return result.stdout


def run(project, environments, outside):
    outside.mkdir(parents=True, exist_ok=True)
    env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    env["PYTHONIOENCODING"] = "utf-8"
    report = {"outside": str(outside), "installation_modes": []}
    for mode in ("base", "client", "server"):
        interpreter = environments / mode / "Scripts/python.exe"
        probe = invoke(
            [
                str(interpreter),
                "-I",
                "-c",
                "import json,sys,importlib.util,realtime_asr; "
                "import apps.client,apps.server; "
                "print(json.dumps({'package':realtime_asr.__file__,"
                "'loaded':[m for m in ('faster_whisper','sherpa_onnx','onnxruntime','pyaudio',"
                "'soundfile','fastapi','httpx') if m in sys.modules],"
                "'installed':{m:importlib.util.find_spec(m) is not None for m in "
                "('faster_whisper','sherpa_onnx','onnxruntime','pyaudio','soundfile','fastapi','httpx')}}))",
            ],
            outside,
            env,
        )
        imported = json.loads(probe)
        assert not imported["loaded"]
        assert Path(imported["package"]).is_relative_to(environments / mode)
        if mode != "server":
            assert not imported["installed"]["faster_whisper"]
            assert not imported["installed"]["onnxruntime"]
        assert not imported["installed"]["pyaudio"] and not imported["installed"]["soundfile"]
        entries = ["asr"]
        for entry in entries:
            invoke(
                [str(environments / mode / "Scripts" / (entry + ".exe")), "--help"], outside, env
            )
        report["installation_modes"].append(
            {"mode": mode, "probe": imported, "help_entries": entries}
        )

    script = outside / "run-server.py"
    script.write_text(SERVER_PROGRAM, encoding="utf-8")
    server_log = outside / "server.stderr"
    with server_log.open("wb") as errors:
        process = subprocess.Popen(
            [
                str(environments / "server/Scripts/python.exe"),
                "-I",
                str(script),
                str(project / ".runtime/models/whisper-tiny"),
                str(project / ".runtime/models/vad/silero-v6.onnx"),
            ],
            cwd=outside,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=errors,
        )
        try:
            url = json.loads(process.stdout.readline())["url"]
            with httpx.Client(base_url=url, timeout=5, trust_env=False) as http:
                deadline = time.monotonic() + 60
                while http.get("/api/v1/ready").status_code != 200:
                    if time.monotonic() > deadline:
                        raise AssertionError("独立安装服务未就绪")
                    time.sleep(0.05)
                root = http.get("/")
                assert root.status_code == 200 and "<title>Realtime ASR</title>" in root.text
                assert http.get("/audio-worklet.js").status_code == 200
                code = invoke(
                    [
                        str(environments / "client/Scripts/python.exe"),
                        "-I",
                        "-c",
                        CLIENT_PROGRAM,
                        url,
                    ],
                    outside,
                    env,
                )
                report["client_messages"] = json.loads(code)
                with wave.open(str(project / "data/corpus/p0/C01-1.wav"), "rb") as source:
                    pcm = source.readframes(source.getnframes())
                command = [
                    str(environments / "client/Scripts/asr.exe"),
                    "remote",
                    "--url",
                    url,
                    "--language",
                    "zh-CN",
                    "--output",
                    "jsonl",
                    "stdin",
                ]
                output = invoke(command, outside, env, input=pcm)
                messages = [json.loads(line) for line in output.splitlines()]
                assert any(m.get("event", {}).get("kind") == "final" for m in messages)
                report["remote_stdin"] = messages
                # 真实进程切换编码环境，输出仍应遵守公开的 UTF-8 JSONL。
                alternate = dict(env, PYTHONIOENCODING="gbk")
                encoded = invoke(command, outside, alternate, input=pcm)
                assert any(
                    json.loads(line).get("event", {}).get("kind") == "final"
                    for line in encoded.splitlines()
                )
                report["remote_jsonl_utf8"] = True
                assert http.get("/api/v1/metrics").json()["sessions_open"] == 0
        finally:
            # 控制验证进程的标准输入，请求真实 Uvicorn 正常关闭。
            process.stdin.write(b"shutdown\n")
            process.stdin.flush()
            process.stdin.close()
            process.wait(timeout=60)
            process.stdout.close()
        assert process.returncode == 0
    report["installed_server_shutdown"] = "ok"
    with zipfile.ZipFile(project / "dist/realtime_asr-0.1.0-py3-none-any.whl") as wheel:
        names = wheel.namelist()
        assert any(name.endswith("server/static/audio-worklet.js") for name in names)
        assert not any(name.startswith(("evals/", "tests/", "web/", "docs/")) for name in names)
        report["wheel_members"] = len(names)
        report["ui_packaged"] = True
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--environments", type=Path, default=Path(".runtime/p6-installed"))
    parser.add_argument("--outside", type=Path, required=True)
    args = parser.parse_args()
    report = run(Path.cwd(), args.environments.resolve(), args.outside.resolve())
    Path("artifacts/p6/installed.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("P6 independent wheel verification passed")


if __name__ == "__main__":
    main()
