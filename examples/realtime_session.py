"""公开实时 SDK 示例：按原音频节奏送入标准 WAV，打印实时和最终事件。"""

import argparse
import json
import time
import wave
from dataclasses import asdict
from pathlib import Path

from realtime_asr import ASRRuntime, RealtimeConfig, RecognizerConfig, SessionConfig


def main():
    parser = argparse.ArgumentParser(description="实时会话：标准 PCM16 WAV 节奏输入")
    parser.add_argument("audio", type=Path, nargs="+")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--backend", choices=("whisper", "nemotron"), default="whisper")
    parser.add_argument("--language")
    args = parser.parse_args()
    model = RecognizerConfig(model_path=args.model, backend=args.backend)
    mode = model.capabilities.realtime_mode
    with ASRRuntime(
        model,
        realtime=RealtimeConfig(mode=mode),
    ) as runtime:
        session = runtime.create_session(SessionConfig(realtime=True, language=args.language))
        completed = 0

        def consume(timeout=0):
            nonlocal completed
            event = session.get_event(timeout=timeout)
            if event is None:
                return False
            print(json.dumps(asdict(event), ensure_ascii=False), flush=True)
            if event.kind in ("completed", "failed", "cancelled"):
                completed += 1
            return True

        for path in args.audio:
            with wave.open(str(path)) as source:
                if (source.getframerate(), source.getnchannels(), source.getsampwidth()) != (
                    16000,
                    1,
                    2,
                ):
                    raise ValueError("此示例需要 16 kHz 单声道 PCM16 WAV")
                session.start()
                started, samples = time.monotonic(), 0
                while chunk := source.readframes(3200):
                    session.feed_audio(chunk)
                    samples += len(chunk) // 2
                    while consume():
                        pass
                    time.sleep(max(0, started + samples / 16000 - time.monotonic()))
                session.finish()
        while completed < len(args.audio):
            if not consume(timeout=60):
                raise TimeoutError("等待识别事件超时")
        session.close()
    print(json.dumps(asdict(runtime.close_result), ensure_ascii=False))


if __name__ == "__main__":
    main()
