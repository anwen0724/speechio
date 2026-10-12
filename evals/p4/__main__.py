"""准备本地检测模型或执行真实分句验证。"""

import argparse
import hashlib
import json
from importlib.metadata import distribution
from pathlib import Path

from evals.p4.runner import run


def main():
    parser = argparse.ArgumentParser(description="P4 真实检测与分句验证")
    parser.add_argument("--prepare-vad", action="store_true")
    parser.add_argument("--configuration", choices=("whisper-cpu", "whisper-cuda", "nemotron"))
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[2]
    output = project / "artifacts/p4/segmentation-check"
    output.mkdir(parents=True, exist_ok=True)
    model = project / ".runtime/models/vad/silero-v6.onnx"
    if args.prepare_vad:
        source = distribution("faster-whisper").locate_file(
            "faster_whisper/assets/silero_vad_v6.onnx"
        )
        data = source.read_bytes()
        expected = "4cbf549b8326f60f80f2536d9eefeb450a9abe83365a098031c89719f1be17d2"
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError("准备的 Silero 图与已检查版本不同")
        model.parent.mkdir(parents=True, exist_ok=True)
        if model.exists() and model.read_bytes() != data:
            raise ValueError("现有检测模型身份不同，不能覆盖")
        model.write_bytes(data)
        identity = {
            "model": "silero-v6",
            "sha256": expected,
            "source": "faster-whisper 1.2.1 distributed asset",
            "upstream": "https://github.com/snakers4/silero-vad",
            "license": "MIT",
        }
        model.with_suffix(".json").write_text(json.dumps(identity, indent=2), encoding="utf-8")
        print(f"Prepared: {model}")
        return
    results = run(project, model, output, configuration=args.configuration)
    path = output / f"{args.configuration or 'all'}.json"
    path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(f"Verified {len(results)} cases: {path}")


if __name__ == "__main__":
    main()
