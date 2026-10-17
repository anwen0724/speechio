"""完整输入仍在同一单元；候选、最终和修正分别消费。"""

import argparse
from contextlib import closing
from pathlib import Path

from realtime_asr import ASRRuntime, CandidateConfig, RecognizerConfig, SessionConfig
from realtime_asr.core.audio.files import audio_file_chunks


def wait_for(session, kind):
    """示例遇到实际失败即退出，不把模型失败当作继续等待。"""
    while True:
        event = session.get_event(timeout=60)
        if event is None:
            raise TimeoutError(f"等待 {kind} 超时")
        if event.error:
            raise RuntimeError(f"{event.error.code}: {event.error.message}")
        if event.kind == kind:
            return event


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("audio", type=Path)
    args = parser.parse_args()
    with ASRRuntime(RecognizerConfig(model=args.model), candidate=CandidateConfig()) as runtime:
        with closing(audio_file_chunks(args.audio)) as blocks:
            format, first = next(blocks)
            session = runtime.create_session(
                SessionConfig(
                    format=format, completion_mode="candidate", retain_for_correction=True
                )
            )
            unit = session.start()
            session.feed_audio(first)
            session.request_candidate(hint=True)
            session.resume()
            for _, block in blocks:
                session.feed_audio(block)
            session.request_candidate()
            event = wait_for(session, "candidate")
            print("整轮候选：", event.candidate.text)
            session.finish()
            final = wait_for(session, "final")
            print(final.result.text)
            wait_for(session, "completed")
            session.request_correction(unit)
            event = session.get_event(timeout=60)
            print(event.kind, event.candidate.text)
            session.close()


if __name__ == "__main__":
    main()
