"""独立程序示例：只通过公开 SDK 识别本地文件。"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from realtime_asr import Recognizer, RecognizerConfig


def main():
    """模型与录音由使用者提供，不进行下载或目录猜测。"""
    parser = argparse.ArgumentParser(description="通过 SDK 转写本地录音")
    parser.add_argument("audio", type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--backend", choices=("whisper", "nemotron"), default="whisper")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--cudnn", type=Path)
    parser.add_argument("--language")
    args = parser.parse_args()
    config = RecognizerConfig(
        model=args.model, engine=args.backend, device=args.device, cudnn_folder=args.cudnn
    )
    with Recognizer(config) as recognizer:
        result = recognizer.transcribe_file(args.audio, language=args.language)
    print(
        json.dumps(
            {"result": asdict(result), "close": asdict(recognizer.close_result)},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
