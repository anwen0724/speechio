"""安装本次 wheel 到已有独立依赖环境，并从项目外验证实际包。"""

import argparse
import json
import os
import subprocess
import zipfile
from pathlib import Path

from evals.p6.installed_check import invoke, run

PROBE = """import json
import numpy as np
import realtime_asr as sdk

class Engine(sdk.BaseTranscriptionEngine):
    def __init__(self):
        self.prepared = 0
        self.closed = 0

    @property
    def capabilities(self):
        return sdk.EngineCapabilities(backend='installed-custom')

    def prepare(self):
        self.prepared += 1

    def transcribe(self, audio, language=None, *, role='final', word_timestamps=False):
        assert len(audio) == 525 and audio.dtype == np.float32
        return sdk.RecognitionResult(text='installed custom', requested_language=language)

    def close(self):
        self.closed += 1

engine = Engine()
with sdk.Recognizer(engine=engine) as recognizer:
    result = recognizer.transcribe(np.zeros(525, np.int16), format=sdk.AudioFormat())
    assert result.text == 'installed custom'
assert engine.prepared == 1 and engine.closed == 1
engine = Engine()
with sdk.ASRRuntime(engine=engine) as runtime:
    session = runtime.create_session()
    session.start()
    session.feed_audio(np.zeros(525, np.int16))
    session.finish()
    assert session.get_event(timeout=5).result.text == 'installed custom'
    assert session.get_event(timeout=5).kind == 'completed'
    session.close()
assert engine.prepared == 1 and engine.closed == 1
print(json.dumps({
    'package': sdk.__file__, 'exports': len(sdk.__all__),
    'custom_sdk':True, 'custom_runtime':True, 'closed':engine.closed,
}))
"""


def main():
    parser = argparse.ArgumentParser(description="P8 wheel 项目外检查，复用已准备的三种依赖环境")
    parser.add_argument("--environments", type=Path, default=Path(".runtime/p6-installed"))
    parser.add_argument("--outside", type=Path, required=True)
    args = parser.parse_args()
    project = Path.cwd()
    outside = args.outside.resolve()
    if outside.is_relative_to(project):
        raise ValueError("独立检查目录应在项目外")
    environments = args.environments.resolve()
    wheel_path = project / "dist/realtime_asr-0.1.0-py3-none-any.whl"
    env = os.environ.copy()
    env["UV_CACHE_DIR"] = str(project / ".runtime/uv-cache")
    env["UV_LINK_MODE"] = "copy"
    env["PYTHONIOENCODING"] = "utf-8"
    env.pop("PYTHONPATH", None)
    for mode in ("base", "client", "server"):
        interpreter = environments / mode / "Scripts/python.exe"
        subprocess.run(
            [
                "uv",
                "pip",
                "install",
                "--python",
                str(interpreter),
                "--reinstall",
                "--no-deps",
                str(wheel_path),
            ],
            env=env,
            check=True,
        )
    report = run(project, environments, outside)
    report["p8_api"] = json.loads(
        invoke(
            [
                str(environments / "base/Scripts/python.exe"),
                "-I",
                "-c",
                PROBE,
            ],
            outside,
            env,
        )
    )
    with zipfile.ZipFile(wheel_path) as wheel:
        names = [name for name in wheel.namelist() if name.startswith("realtime_asr/")]
        for name in names:
            if wheel.read(name) != (project / "src" / name).read_bytes():
                raise AssertionError(f"wheel 与实际源码／资源不一致：{name}")
        for required in ("backends/base.py", "backends/local.py", "runtime/tasks.py"):
            if f"realtime_asr/{required}" not in names:
                raise AssertionError(f"wheel 缺少 P8 模块：{required}")
    report["all_package_members_match_source"] = True
    report["dependency_profiles_reused"] = True
    report["package_reinstalled_from_current_wheel"] = True
    Path("artifacts/p8/installed.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("P8 installed wheel checks passed")


if __name__ == "__main__":
    main()
