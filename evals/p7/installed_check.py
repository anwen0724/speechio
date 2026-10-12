"""安装本次 wheel 到已有独立依赖环境，并从项目外验证实际包。"""

import argparse
import json
import os
import subprocess
import zipfile
from pathlib import Path

from evals.p6.installed_check import invoke, run

PROBE = """import json
import realtime_asr as sdk

unit = sdk.UnitRef('installed', 0, 1)
text = sdk.TextStabilizer(unit)
text.observe(sdk.TextObservation(unit, 1, '你好世界', 16000, 0))
stable = text.observe(sdk.TextObservation(unit, 2, '你好世界', 32000, 1))
assert stable.stable_text == '你好世界'
assert sdk.EndpointConfig(dynamic_silence=True).pause_for('结束。', .6) == .9
assert sdk.AcousticBoundaryDetector().feed(bytes(640)) == []
config = sdk.SessionConfig(
    realtime=True, stabilization=sdk.StabilizationConfig(), word_timestamps=True,
)
assert config.word_timestamps
print(json.dumps({
    'package': sdk.__file__, 'stable': stable.stable_text, 'exports': len(sdk.__all__),
}))
"""


def main():
    parser = argparse.ArgumentParser(description="P7 wheel 项目外检查，复用已准备的三种依赖环境")
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
    report["p7_api"] = json.loads(
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
        if not any(name.endswith("text/stabilizer.py") for name in names):
            raise AssertionError("wheel 缺少 P7 文本模块")
    report["all_package_members_match_source"] = True
    report["dependency_profiles_reused"] = True
    report["package_reinstalled_from_current_wheel"] = True
    Path("artifacts/p7/installed.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("P7 installed wheel checks passed")


if __name__ == "__main__":
    main()
