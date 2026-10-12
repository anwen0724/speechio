"""只使用公开 SDK：持续接收多个标准 WAV 单元并消费最终事件。"""

import argparse
import json
import wave
from dataclasses import asdict
from pathlib import Path

from realtime_asr import ASRRuntime, RecognizerConfig, SessionConfig


def main():
    parser = argparse.ArgumentParser(description="标准 WAV 连续会话示例")
    parser.add_argument("audio", nargs="+", type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--backend", choices=("whisper", "nemotron"), default="whisper")
    parser.add_argument("--language", default=None)
    args = parser.parse_args()
    with ASRRuntime(RecognizerConfig(model_path=args.model, backend=args.backend)) as runtime:
        session = runtime.create_session(SessionConfig(language=args.language))
        for path in args.audio:
            with wave.open(str(path)) as source:
                if (source.getframerate(), source.getnchannels(), source.getsampwidth()) != (
                    16000,
                    1,
                    2,
                ):
                    raise ValueError("此示例需要 16 kHz 单声道 PCM16 WAV")
                session.start()
                while chunk := source.readframes(2048):
                    session.feed_audio(chunk)
                session.finish()
        for _ in args.audio:
            while True:
                event = session.get_event(timeout=30)
                if event is None:
                    raise TimeoutError("等待识别事件超时")
                print(json.dumps(asdict(event), ensure_ascii=False))
                if event.kind in ("completed", "cancelled", "failed"):
                    break
        session.close()
    print(json.dumps(asdict(runtime.close_result), ensure_ascii=False))


if __name__ == "__main__":
    main()
