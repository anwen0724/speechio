"""远程命令行；文件用 HTTP，连续输入用独立的 WebSocket。"""

import argparse
import json
import os
import sys
from threading import Event, Thread
from time import monotonic

from realtime_asr import ASRError, AudioFormat, MicrophoneConfig, MicrophoneInput
from realtime_asr.cli.args import _positive_int, _positive_seconds, _seconds
from realtime_asr.cli.input import stdin_blocks
from realtime_asr.client import RemoteClient


class RemoteOutput:
    """保留完整协议 JSONL；文本模式只将最终正文写到 stdout。"""

    def __init__(self, mode, stdout, stderr):
        self.mode, self.stdout, self.stderr = mode, stdout, stderr

    def write(self, message):
        if self.mode == "jsonl":
            self.stdout.write(json.dumps(message, ensure_ascii=False) + "\n")
            self.stdout.flush()
            return
        event = message.get("event", {})
        if message["type"] == "transcription":
            self.stdout.write(message["result"]["text"] + "\n")
            self.stdout.flush()
        elif event.get("kind") == "final":
            self.stdout.write(event["result"]["text"] + "\n")
            self.stdout.flush()
        elif event.get("kind") == "realtime":
            stable = event.get("stabilization")
            text = stable["display_text"] if stable else event["result"]["text"]
            self.stderr.write("[realtime] " + text + "\n")
            self.stderr.flush()
        elif message["type"] == "error":
            self.stderr.write(f"[{message['code']}] {message['message']}\n")
            self.stderr.flush()


class MessageConsumer:
    """输入管道空闲时仍独立输出结果，失败交回输入拥有者。"""

    def __init__(self, session, output):
        self.session, self.output = session, output
        self.stop = Event()
        self.error = None
        self.thread = Thread(target=self._run, name="ASRRemoteCLIOutput", daemon=True)

    def _run(self):
        try:
            while not self.stop.is_set():
                message = self.session.get_message(timeout=0.05)
                if message is None:
                    continue
                self.output.write(message)
                event = message.get("event", {})
                if event.get("kind") in ("failed", "realtime_failed"):
                    error = event["error"]
                    raise ASRError(error["code"], error["operation"], error["message"])
                if message["type"] == "error":
                    raise ASRError(message["code"], message["operation"], message["message"])
                if message["type"] == "closed":
                    return
        except Exception as exc:
            self.error = exc

    def check(self):
        if self.error is not None:
            raise self.error
        self.session.check()

    def close(self, *, abort=False):
        if abort:
            self.stop.set()
        self.thread.join(5)
        if self.thread.is_alive():
            self.stop.set()
            self.thread.join(0.1)
            raise ASRError("CLOSE_TIMEOUT", "close_output", "远程输出线程尚未退出")
        self.check()


def parser():
    command = argparse.ArgumentParser(prog="realtime-asr remote", description="接入远程 ASR 服务")
    command.add_argument("--url", required=True)
    command.add_argument("--token-env", default="REALTIME_ASR_TOKEN")
    command.add_argument("--language")
    command.add_argument("--output", choices=("text", "jsonl"), default="text")
    command.add_argument("--word-timestamps", action="store_true")
    command.add_argument("--timeout", type=_positive_seconds, default=30)
    commands = command.add_subparsers(dest="input", required=True)
    file = commands.add_parser("file")
    file.add_argument("paths", nargs="+")
    for name in ("stdin", "mic"):
        source = commands.add_parser(name)
        source.add_argument(
            "--mode", choices=("auto", "manual"), default="auto" if name == "mic" else "manual"
        )
        source.add_argument("--realtime", action=argparse.BooleanOptionalAction, default=False)
        source.add_argument("--stable-text", action="store_true")
        source.add_argument("--dynamic-silence", action="store_true")
        source.add_argument("--punctuation-split", action="store_true")
        source.add_argument("--early-transcription-silence", type=_seconds, default=0)
        source.add_argument(
            "--sample-rate", type=_positive_int, default=None if name == "mic" else 16000
        )
        source.add_argument("--channels", type=_positive_int, default=1)
        if name == "stdin":
            source.add_argument("--sample-type", choices=("int16", "float32"), default="int16")
        else:
            source.add_argument("--input-device", type=int)
            source.add_argument("--duration", type=_positive_seconds)
    return command


def streaming(client, args, output, stdin):
    """先停止自有采集，再排空网络；失败取消不会伪造正常 EOF。"""
    microphone = session = consumer = None
    success = False
    code = 0
    try:
        if args.input == "mic":
            microphone = MicrophoneInput(
                MicrophoneConfig(
                    device_index=args.input_device,
                    sample_rate=args.sample_rate,
                    channels=args.channels,
                )
            )
            format = microphone.prepare()
        else:
            format = AudioFormat(args.sample_rate, args.channels, args.sample_type)
        text_options = {
            name: getattr(args, name)
            for name in (
                "stable_text",
                "word_timestamps",
                "dynamic_silence",
                "punctuation_split",
                "early_transcription_silence",
            )
            if getattr(args, name)
        }
        session = client.connect(
            mode=args.mode,
            format=format,
            language=args.language,
            realtime=args.realtime,
            **text_options,
        )
        if args.mode == "manual":
            session.start()
        consumer = MessageConsumer(session, output)
        consumer.thread.start()
        try:
            if microphone is not None:
                microphone.start()
                started = monotonic()
                while args.duration is None or monotonic() - started < args.duration:
                    consumer.check()
                    if audio := microphone.read_audio(timeout=0.05):
                        session.feed_audio(audio)
            else:
                for audio in stdin_blocks(getattr(stdin, "buffer", stdin), format, consumer.check):
                    session.feed_audio(audio)
        except KeyboardInterrupt:
            code = 130
        if microphone is not None:
            result = microphone.close(timeout=5)
            if result.status != "ok":
                raise ASRError("CLOSE_TIMEOUT", "microphone", "采集资源尚未释放")
            while audio := microphone.read_audio():
                session.feed_audio(audio)
        session.end_input() if args.mode == "auto" else session.finish()
        session.close()
        consumer.close()
        success = True
        return code
    finally:
        if microphone is not None:
            microphone.close(timeout=5)
        if session is not None and not success:
            session.close(graceful=False)
        if consumer is not None and not success:
            consumer.close(abort=True)


def main(argv=None, *, stdin=None, stdout=None, stderr=None):
    """标准流借用调用方对象；不实例化本地识别器。"""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    args = parser().parse_args(argv)
    output = RemoteOutput(args.output, stdout or sys.stdout, stderr or sys.stderr)
    try:
        with RemoteClient(
            args.url, token=os.environ.get(args.token_env), timeout=args.timeout
        ) as client:
            if args.input == "file":
                for path in args.paths:
                    output.write(
                        client.transcribe_file(
                            path, language=args.language, word_timestamps=args.word_timestamps
                        )
                    )
                return 0
            return streaming(client, args, output, stdin or sys.stdin)
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        error = (
            exc
            if isinstance(exc, ASRError)
            else ASRError("INPUT_OUTPUT_FAILED", "remote", str(exc))
        )
        output.write(
            {
                "protocol_version": 1,
                "type": "error",
                "code": error.code,
                "operation": error.operation,
                "message": error.message,
            }
        )
        return 4 if error.code.endswith("TIMEOUT") else 1


if __name__ == "__main__":
    raise SystemExit(main())
