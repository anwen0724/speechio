"""控制台 Space 开始／结束录音，Esc 取消当前录音并退出。"""

import argparse
import os
from contextlib import contextmanager
from pathlib import Path
from time import sleep

from realtime_asr import (
    AudioFormat,
    MicrophoneConfig,
    MicrophoneInput,
    ModelConfig,
    SessionConfig,
)
from realtime_asr.runtime.asr import ASRRuntime


@contextmanager
def terminal_keys():
    """键盘属于示例入口；POSIX 退出时恢复终端设置。"""
    if os.name == "nt":
        import msvcrt

        yield lambda: msvcrt.getwch() if msvcrt.kbhit() else None
    else:
        import select
        import sys
        import termios
        import tty

        old = termios.tcgetattr(sys.stdin)
        try:
            tty.setcbreak(sys.stdin.fileno())
            yield lambda: sys.stdin.read(1) if select.select([sys.stdin], [], [], 0)[0] else None
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--device", type=int)
    args = parser.parse_args()
    microphone = None
    with (
        ASRRuntime(ModelConfig(model=args.model)) as runtime,
        terminal_keys() as read_key,
    ):
        session = runtime.create_session(SessionConfig(format=AudioFormat()))
        print("Space 开始／结束；Esc 取消当前录音并退出。")
        try:
            while True:
                key = read_key()
                if key == "\x1b":
                    session.cancel_turn()
                    break
                if key == " ":
                    if microphone is None:
                        microphone = MicrophoneInput(
                            MicrophoneConfig(device_index=args.device, sample_rate=16000)
                        )
                        microphone.prepare()
                        session.start()
                        microphone.start()
                        print("录音中")
                    else:
                        closed = microphone.close()
                        if closed.status != "ok":
                            raise RuntimeError(f"采集释放失败：{closed.status}")
                        while block := microphone.read_audio():
                            session.feed_audio(block)
                        microphone = None
                        session.end_turn()
                        print("已结束，等待识别")
                if microphone is not None:
                    block = microphone.read_audio(timeout=0.05)
                    if block:
                        session.feed_audio(block)
                else:
                    sleep(0.02)
                while event := session.next_event():
                    if event.error:
                        raise RuntimeError(event.error.code)
                    if event.kind == "final":
                        print(event.result.text)
        finally:
            cleanup = microphone.close() if microphone is not None else None
            session.close()
            if cleanup is not None and cleanup.status != "ok":
                raise RuntimeError(f"采集释放失败：{cleanup.status}")


if __name__ == "__main__":
    main()
