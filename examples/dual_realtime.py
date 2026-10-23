"""使用已经准备的两个模型，通过公开 SDK 消费原文、合并和最终结果。"""

import argparse
from contextlib import closing

from speechio import (
    LocalModelConfig,
    RealtimeConfig,
    RealtimeMergeConfig,
    SessionConfig,
    STTError,
)
from speechio.audio.files import audio_file_chunks
from speechio.stt.options import TranscriptionOptions
from speechio.stt.runtime.asr import STTRuntime


def show(event):
    """模型错误保留分类；合并文本只作为预览，不替代最终正文。"""
    if event.error is not None:
        raise STTError(event.error.code, event.error.operation, event.error.message)
    print(
        event.kind,
        event.realtime_role,
        event.result.text if event.result else "",
        event.merge.text if event.merge else "",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file")
    parser.add_argument("--accurate-model", required=True)
    parser.add_argument("--ultrafast-model", required=True)
    parser.add_argument("--language")
    args = parser.parse_args()
    accurate = LocalModelConfig(model_path=args.accurate_model, adapter="nemotron")
    fast = LocalModelConfig(model_path=args.ultrafast_model, adapter="nemotron")
    with STTRuntime(
        accurate,
        realtime=RealtimeConfig(mode="streaming"),
        ultrafast=RealtimeConfig(mode="streaming", config=fast),
    ) as runtime:
        with closing(audio_file_chunks(args.file)) as chunks:
            format, first = next(chunks)
            session = runtime.create_session(
                SessionConfig(
                    format=format,
                    options=TranscriptionOptions(language=args.language),
                    realtime=True,
                    merge=RealtimeMergeConfig(),
                )
            )
            try:
                session.start()
                session.feed_audio(first)
                for _, audio in chunks:
                    session.feed_audio(audio)
                    while event := session.next_event():
                        show(event)
                session.end_turn()
                while True:
                    event = session.next_event(timeout=60)
                    if event is None:
                        raise TimeoutError("识别没有在等待期限内完成")
                    show(event)
                    if event.kind == "completed":
                        break
            finally:
                session.close()


if __name__ == "__main__":
    main()
