"""使用标准 PCM 文件连续分句；麦克风和语料评估属于其他入口。"""

import argparse
import json
import wave
from dataclasses import asdict

from realtime_asr import (
    ModelConfig,
    RealtimeConfig,
    SessionConfig,
    VadConfig,
)
from realtime_asr.bundled import default_vad_model
from realtime_asr.runtime.asr import ASRRuntime


def main():
    parser = argparse.ArgumentParser(description="连续 PCM 文件自动分句")
    parser.add_argument("audio")
    parser.add_argument("--model", required=True)
    parser.add_argument("--vad-model", default=None)
    parser.add_argument("--backend", choices=("whisper", "nemotron"), default="whisper")
    parser.add_argument("--language", default=None)
    parser.add_argument("--realtime", action="store_true")
    args = parser.parse_args()
    model = ModelConfig(model=args.model, engine=args.backend)
    realtime = None
    if args.realtime:
        realtime = RealtimeConfig(mode=model.capabilities.realtime_mode)
    with ASRRuntime(model, realtime=realtime) as runtime:
        session = runtime.create_auto_session(
            vad=VadConfig(model_path=args.vad_model or default_vad_model()),
            config=SessionConfig(language=args.language, realtime=args.realtime),
        )

        def consume(timeout=0):
            event = session.next_event(timeout=timeout)
            if event is not None:
                print(json.dumps(asdict(event), ensure_ascii=False))
            return event

        with wave.open(args.audio) as source:
            if (source.getframerate(), source.getnchannels(), source.getsampwidth()) != (
                16000,
                1,
                2,
            ):
                raise ValueError("本示例接收 16 kHz 单声道 PCM16 WAV")
            while chunk := source.readframes(512):
                session.feed_audio(chunk)
                while consume() is not None:
                    pass
        session.end_input()
        while session.pending_segments:
            if consume(timeout=30) is None:
                raise TimeoutError("录音段结果尚未完成")
        session.close()


if __name__ == "__main__":
    main()
