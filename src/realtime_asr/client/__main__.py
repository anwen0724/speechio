"""远程命令行；基本文件使用 HTTP，候选与连续音频使用 WebSocket。"""

import argparse
import json
import os
import sys
from contextlib import closing
from itertools import chain
from threading import Event, Thread
from time import monotonic

from realtime_asr import ASRError, AudioFormat, MicrophoneConfig, MicrophoneInput
from realtime_asr.cli.args import (
    _positive_int,
    _positive_seconds,
    _seconds,
    add_candidate_request_args,
)
from realtime_asr.cli.candidates import CandidateTicker
from realtime_asr.cli.input import stdin_blocks
from realtime_asr.client import RemoteClient
from realtime_asr.core.audio.files import audio_file_chunks


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
            merge = event.get("merge")
            if merge is not None:
                text = merge["text"]
            elif stable is not None:
                text = stable["display_text"]
            else:
                text = event["result"]["text"]
            self.stderr.write("[realtime] " + text + "\n")
            self.stderr.flush()
        elif event.get("candidate") is not None:
            value = event["candidate"]
            self.stderr.write(f"[{event['kind']}:{value['status']}] {value['text']}\n")
            if event.get("error"):
                self.stderr.write(f"[{event['error']['code']}] {event['error']['message']}\n")
            self.stderr.flush()
        elif message["type"] == "status":
            self.stderr.write(f"[{message['status']}] {message.get('reason', '')}\n")
            self.stderr.flush()
        elif message["type"] == "error":
            self.stderr.write(f"[{message['code']}] {message['message']}\n")
            self.stderr.flush()


class MessageConsumer:
    """输入管道空闲时仍独立输出结果，失败交回输入拥有者。"""

    def __init__(self, session, output, *, correct_after_final=False):
        self.session, self.output = session, output
        self.stop = Event()
        self.error = None
        self.correct_after_final = correct_after_final
        self.correction_pending = False
        self.terminal_seen = False
        self.results_done = Event()
        self.thread = Thread(target=self._run, name="ASRRemoteCLIOutput", daemon=True)

    def _run(self):
        try:
            while not self.stop.is_set():
                message = self.session.get_message(timeout=0.05)
                if message is None:
                    continue
                self.output.write(message)
                event = message.get("event", {})
                if event.get("kind") == "final" and self.correct_after_final:
                    self.correction_pending = True
                    self.session.request_correction(event["unit"])
                if event.get("kind") in ("correction", "correction_failed"):
                    self.correction_pending = False
                if event.get("kind") == "completed":
                    self.terminal_seen = True
                if self.terminal_seen and not self.correction_pending:
                    self.results_done.set()
                if event.get("kind") in ("failed", "realtime_failed", "correction_failed"):
                    error = event["error"]
                    raise ASRError(error["code"], error["operation"], error["message"])
                if message["type"] == "error":
                    raise ASRError(message["code"], message["operation"], message["message"])
                if message["type"] == "closed":
                    return
        except Exception as exc:
            self.error = exc
            self.results_done.set()

    def wait_results(self, timeout):
        """手动修正完成后才关闭连接，避免 completed 抢先结束消费者。"""
        deadline = monotonic() + timeout
        while not self.results_done.wait(0.05):
            self.check()
            if monotonic() >= deadline:
                raise ASRError("DRAIN_TIMEOUT", "correct", "等待远程修正结果超时")
        self.check()

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


def parser(*, diagnostic=False):
    command = argparse.ArgumentParser(prog="asr remote", description="接入远程 ASR 服务")
    command.add_argument(
        "--config", default=argparse.SUPPRESS, help="指定 config.toml，默认读取当前目录"
    )
    command.add_argument("--url", required=True)
    command.add_argument("--token-env", default="REALTIME_ASR_TOKEN")
    command.add_argument("--language")
    command.add_argument("--output", choices=("text", "jsonl"), default="text")
    command.add_argument("--word-timestamps", action=argparse.BooleanOptionalAction, default=False)
    command.add_argument("--timeout", type=_positive_seconds, default=30)
    commands = command.add_subparsers(dest="input", required=True)
    file = commands.add_parser("file")
    file.add_argument("paths", nargs="*" if diagnostic else "+")
    for name in ("file", "stdin", "mic"):
        source = file if name == "file" else commands.add_parser(name)
        source.add_argument("--candidates", action=argparse.BooleanOptionalAction, default=False)
        add_candidate_request_args(source)
        source.add_argument("--wake", action=argparse.BooleanOptionalAction, default=False)
        source.add_argument(
            "--mode", choices=("auto", "manual"), default="auto" if name == "mic" else "manual"
        )
        source.add_argument("--realtime", action=argparse.BooleanOptionalAction, default=False)
        source.add_argument("--stable-text", action=argparse.BooleanOptionalAction, default=False)
        source.add_argument("--dual-realtime", action=argparse.BooleanOptionalAction, default=False)
        source.add_argument(
            "--dynamic-silence", action=argparse.BooleanOptionalAction, default=False
        )
        source.add_argument(
            "--punctuation-split", action=argparse.BooleanOptionalAction, default=False
        )
        source.add_argument("--early-transcription-silence", type=_seconds, default=0)
        if name != "file":
            source.add_argument(
                "--sample-rate", type=_positive_int, default=None if name == "mic" else 16000
            )
            source.add_argument("--channels", type=_positive_int, default=1)
        if name == "stdin":
            source.add_argument("--sample-type", choices=("int16", "float32"), default="int16")
        elif name == "mic":
            source.add_argument("--input-device", type=int)
            source.add_argument("--duration", type=_positive_seconds)
    return command


def streaming(client, args, output, stdin, *, file_input=None):
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
        elif file_input is not None:
            format, file_blocks = file_input
        else:
            format = AudioFormat(args.sample_rate, args.channels, args.sample_type)
        text_options = {
            name: getattr(args, name)
            for name in (
                "stable_text",
                "dual_realtime",
                "word_timestamps",
                "dynamic_silence",
                "punctuation_split",
                "early_transcription_silence",
                "completion_mode",
                "retain_for_correction",
            )
            if getattr(args, name)
        }
        if args.wake:
            text_options["wake"] = True
        session = client.connect(
            mode=args.mode,
            format=format,
            language=args.language,
            realtime=args.realtime,
            **text_options,
        )
        if args.candidates and not session.candidate_available:
            raise ASRError("CANDIDATE_NOT_CONFIGURED", "candidate", "远端没有候选角色")
        ticker = CandidateTicker(args, format, session.request_candidate)

        def accept(audio):
            session.feed_audio(audio)
            if skipped := ticker.accept(audio):
                output.write(
                    {
                        "protocol_version": 1,
                        "type": "status",
                        "status": "candidate_skipped",
                        "reason": skipped,
                    }
                )

        if args.mode == "manual":
            session.start()
        consumer = MessageConsumer(session, output, correct_after_final=args.correct_after_final)
        consumer.thread.start()
        try:
            if microphone is not None:
                microphone.start()
                started = monotonic()
                while args.duration is None or monotonic() - started < args.duration:
                    consumer.check()
                    if audio := microphone.read_audio(timeout=0.05):
                        accept(audio)
            elif file_input is not None:
                dtype = "<i2" if format.sample_type == "int16" else "<f4"
                for block in file_blocks:
                    consumer.check()
                    accept(block.astype(dtype, copy=False).tobytes())
            else:
                for audio in stdin_blocks(getattr(stdin, "buffer", stdin), format, consumer.check):
                    accept(audio)
        except KeyboardInterrupt:
            code = 130
        if microphone is not None:
            result = microphone.close(timeout=5)
            if result.status != "ok":
                raise ASRError("CLOSE_TIMEOUT", "microphone", "采集资源尚未释放")
            while audio := microphone.read_audio():
                accept(audio)
        session.end_input() if args.mode == "auto" else session.finish()
        if args.correct_after_final:
            consumer.wait_results(args.timeout)
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


def parse_options(argv=None, *, diagnostic=False):
    """远程配置只影响协议参数，不构造本地模型。"""
    from realtime_asr.cli.config import configure, remote_source, split_config

    selected = list(sys.argv[1:] if argv is None else argv)
    _, cleaned = split_config(selected)
    source, _ = remote_source(cleaned)
    command = parser(diagnostic=diagnostic)
    try:
        invocation = configure(selected, "remote-" + str(source))
    except ASRError as exc:
        command.error(exc.message)
    args = command.parse_args(invocation.argv)
    args.launch_config = invocation
    args.completion_mode = args.completion_mode or "final"
    if args.wake and args.mode != "auto":
        command.error("唤醒只支持自动输入")
    if args.correct_after_final:
        args.retain_for_correction = True
        if args.mode != "manual":
            command.error("结束后修正只支持手动输入")
    if args.dual_realtime and not args.realtime:
        command.error("双实时要求 --realtime")
    return args


def main(argv=None, *, stdin=None, stdout=None, stderr=None):
    """标准流借用调用方对象；不实例化本地识别器。"""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    args = parse_options(argv)
    output = RemoteOutput(args.output, stdout or sys.stdout, stderr or sys.stderr)
    try:
        with RemoteClient(
            args.url, token=os.environ.get(args.token_env), timeout=args.timeout
        ) as client:
            if args.input == "file":
                for path in args.paths:
                    if (
                        args.candidates
                        or args.completion_mode == "candidate"
                        or args.retain_for_correction
                        or args.wake
                    ):
                        with closing(audio_file_chunks(path)) as blocks:
                            format, first = next(blocks)
                            source = format, chain((first,), (audio for _, audio in blocks))
                            code = streaming(client, args, output, stdin, file_input=source)
                        if code:
                            return code
                        continue
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
