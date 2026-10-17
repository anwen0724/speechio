"""从当前源码构建 wheel，清理旧缓存并核对全部包成员。"""

import os
import shutil
import subprocess
import zipfile
from pathlib import Path


def main():
    project = Path(__file__).resolve().parents[1]
    build = project / "build"
    if build.exists():
        if build.resolve() != build or build.is_symlink() or build.is_junction():
            raise RuntimeError("build 不是项目内的普通生成目录，停止清理")
        shutil.rmtree(build)
    env = os.environ.copy()
    env["UV_CACHE_DIR"] = str(project / ".runtime/uv-cache")
    env["UV_LINK_MODE"] = "copy"
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("构建需要 uv")
    subprocess.run([uv, "build", "--wheel"], cwd=project, env=env, check=True)
    expected = {
        path.relative_to(project / "src").as_posix(): path.read_bytes()
        for path in (project / "src/realtime_asr").rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    }
    wheels = list((project / "dist").glob("realtime_asr-*.whl"))
    wheel = max(wheels, key=lambda path: path.stat().st_mtime)
    with zipfile.ZipFile(wheel) as package:
        names = {name for name in package.namelist() if name.startswith("realtime_asr/")}
        if names != set(expected):
            raise RuntimeError(
                f"wheel 成员与当前包不一致：{sorted(names.symmetric_difference(expected))}"
            )
        if any(package.read(name) != content for name, content in expected.items()):
            raise RuntimeError("wheel 内容与当前源文件不同")
    print(f"构建及 {len(expected)} 个包成员核对通过：{wheel}")


if __name__ == "__main__":
    main()
