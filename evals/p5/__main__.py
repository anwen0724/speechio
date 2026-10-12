"""显式运行 P5 本机验证，不自动播放音频或伪造口述验收。"""

import json
from pathlib import Path

from evals.p5.runner import run


def main():
    project = Path(__file__).resolve().parents[2]
    folder = project / "artifacts/p5/real-check"
    folder.mkdir(parents=True, exist_ok=True)
    reports = run(project, folder)
    (folder / "summary.json").write_text(
        json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
