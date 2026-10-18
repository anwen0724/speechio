"""通过公开快照保存本轮标准录音；保存和识别分别报告结果。"""

import argparse
import json
import wave
from contextlib import closing
from dataclasses import asdict
from pathlib import Path

from speechio import ModelConfig, SessionConfig
from speechio.audio.files import audio_file_chunks
from speechio.runtime.asr import ASRRuntime


def main():
    parser = argparse.ArgumentParser(description="解码音频，在结束识别前保存当前标准录音")
    parser.add_argument("model_path", type=Path)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--backend", choices=("whisper", "nemotron"), default="whisper")
    args = parser.parse_args()
    with ASRRuntime(ModelConfig(model_path=args.model_path, engine=args.backend)) as runtime:
        with closing(audio_file_chunks(args.input)) as blocks:
            format, first = next(blocks)
            session = runtime.create_session(SessionConfig(format=format))
            session.start()
            session.feed_audio(first)
            for _, block in blocks:
                session.feed_audio(block)
            snapshot = session.current_audio()
            with wave.open(str(args.output), "wb") as recording:
                recording.setparams((1, 2, snapshot.sample_rate, 0, "NONE", "not compressed"))
                recording.writeframes(snapshot.pcm)
            print(
                json.dumps(
                    {
                        "saved": str(args.output),
                        "unit": asdict(snapshot.turn),
                        "start_sample": snapshot.start_sample,
                        "end_sample": snapshot.end_sample,
                    },
                    ensure_ascii=False,
                )
            )
            session.end_turn()
            while True:
                event = session.next_event(timeout=60)
                if event is None:
                    raise TimeoutError("等待最终结果超时")
                if event.error:
                    raise RuntimeError(event.error.code)
                if event.kind == "final":
                    print(event.result.text)
                if event.kind == "completed":
                    break
            session.close()


if __name__ == "__main__":
    main()
