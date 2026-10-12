"""CLI 生命周期协调：输入与事件独立推进，识别规则由公开 SDK 承担。"""

import os
import sys
from contextlib import closing
from dataclasses import asdict
from itertools import chain
from threading import Event, Thread
from time import monotonic

from realtime_asr import (
    ASRError,
    ASRRuntime,
    AudioFormat,
    EndpointConfig,
    MicrophoneConfig,
    MicrophoneInput,
    RealtimeMergeConfig,
    RecognizerConfig,
    SegmentationConfig,
    SessionConfig,
    StabilizationConfig,
    VadConfig,
    list_input_devices,
)
from realtime_asr.audio.files import audio_file_chunks
from realtime_asr.cli.args import candidate_role, parse_args, realtime_roles, wake_role
from realtime_asr.cli.candidates import CandidateTicker
from realtime_asr.cli.input import stdin_blocks
from realtime_asr.cli.output import Output


class _EventConsumer:
    """stdin 等待下一块时仍消费结果，保存输出失败供输入线程处理。"""

    def __init__(self, session, output, index, source, *, automatic, correct_after_final=False):
        self.session = session
        self.output = output
        self.index = index
        self.source = source
        self.automatic = automatic
        self.correct_after_final = correct_after_final
        self.correction_pending = False
        self.finished = Event()
        self.stopping = Event()
        self.done = Event()
        self.error = None
        self.terminals = 0
        self.thread = Thread(target=self._run, name="ASRCLIEvents", daemon=True)

    def _run(self):
        try:
            while not self.stopping.is_set():
                event = self.session.get_event(timeout=0.05)
                if event is not None:
                    self.output.event(event, input_index=self.index, source=self.source)
                    if event.kind == "final" and self.correct_after_final:
                        self.correction_pending = True
                        self.session.request_correction(event.unit)
                    if event.kind in ("correction", "correction_failed"):
                        self.correction_pending = False
                    if event.kind in ("failed", "realtime_failed", "correction_failed"):
                        raise ASRError(event.error.code, event.error.operation, event.error.message)
                    if event.kind in ("completed", "cancelled"):
                        self.terminals += 1
                if self.finished.is_set():
                    empty = (
                        self.session.pending_segments == 0
                        if self.automatic
                        else self.terminals >= 1
                    )
                    if empty and not self.correction_pending:
                        break
        except Exception as exc:
            self.error = exc
        finally:
            self.done.set()

    def check(self):
        if self.error is not None:
            raise self.error

    def drain(self, timeout):
        self.finished.set()
        deadline = monotonic() + timeout
        # 短等待让主线程处理再次 Ctrl+C，不用长时间锁等待延后取消。
        while not self.done.is_set():
            remaining = deadline - monotonic()
            if remaining <= 0:
                raise ASRError("DRAIN_TIMEOUT", "drain", "等待最终结果超时")
            self.done.wait(min(0.05, remaining))
        self.check()

    def close(self, timeout):
        self.stopping.set()
        if self.thread.ident is None:
            return
        self.thread.join(timeout)
        if self.thread.is_alive():
            raise ASRError("CLOSE_TIMEOUT", "close_events", "事件消费仍未退出")


def _session(runtime, args, format):
    """输入格式交给会话，不在 CLI 复制音频转换或切分逻辑。"""
    samples = int(args.max_audio_seconds * 16000)
    config = SessionConfig(
        format=format,
        language=args.language,
        max_audio_samples=samples,
        realtime=args.realtime,
        stabilization=StabilizationConfig() if args.stable_text else None,
        word_timestamps=args.word_timestamps,
        merge=RealtimeMergeConfig() if args.dual_realtime else None,
        completion_mode=args.completion_mode,
        retain_for_correction=args.retain_for_correction,
    )
    if args.mode == "auto":
        return runtime.create_auto_session(
            wake=wake_role(args),
            config=config,
            vad=VadConfig(
                model_path=args.vad_model,
                webrtc_mode=args.webrtc_mode,
                silero_end_detection=args.silero_end_detection,
            ),
            segmentation=SegmentationConfig(
                trim_pre_roll=args.trim_pre_roll,
                backdate_duration=args.backdate_seconds,
                max_buffer_samples=max(1920000, samples + 17024),
            ),
            endpoint=EndpointConfig(
                early_transcription_silence=args.early_transcription_silence,
                dynamic_silence=args.dynamic_silence,
                punctuation_split=args.punctuation_split,
            ),
        )
    session = runtime.create_session(config)
    session.start()
    return session


def _run_input(runtime, args, output, index, source, format, blocks, *, microphone=None):
    """明确正常封口和失败取消，并对自己创建的消费线程负责。"""
    session = _session(runtime, args, format)
    consumer = _EventConsumer(
        session,
        output,
        index,
        source,
        automatic=args.mode == "auto",
        correct_after_final=args.correct_after_final,
    )
    ticker = CandidateTicker(args, format, session.request_candidate)
    last_wake_state = None

    def accept(block):
        nonlocal last_wake_state
        session.feed_audio(block)
        if args.wake:
            wake = session.wake_status
            stamp = (wake["state"], wake.get("detections"), wake.get("timeouts"))
            if stamp != last_wake_state:
                output.record(
                    "status",
                    status="input_state",
                    input_state=session.input_state,
                    wake_status=wake,
                    input_index=index,
                    source=source,
                )
                last_wake_state = stamp
        if skipped := ticker.accept(block):
            output.record("status", status="candidate_skipped", reason=skipped, source=source)

    reason, code = "eof", 0
    cleanup_error = None
    try:
        consumer.thread.start()
        fields = {
            "status": "ready",
            "input_index": index,
            "source": source,
            "format": asdict(format),
        }
        if microphone is not None:
            fields["device"] = asdict(microphone.device)
            microphone.start()
        output.record("status", **fields)
        try:
            if microphone is not None:
                started = monotonic()
                while args.duration is None or monotonic() - started < args.duration:
                    consumer.check()
                    pcm = microphone.read_audio(timeout=0.05)
                    if pcm:
                        accept(pcm)
                reason = "duration"
            else:
                if args.command == "stdin":
                    blocks = stdin_blocks(blocks, format, consumer.check)
                for block in blocks:
                    consumer.check()
                    accept(block)
        except KeyboardInterrupt:
            reason, code = "interrupt", 130
        # 先停止设备，再接纳已经成功入队的尾部；不释放使用中的模型。
        if microphone is not None:
            result = microphone.close(timeout=args.close_timeout)
            if result.status != "ok":
                raise ASRError(
                    "CLOSE_TIMEOUT" if result.status == "pending" else "MICROPHONE_CLEANUP_FAILED",
                    "close_microphone",
                    result.status,
                )
            while pcm := microphone.read_audio():
                consumer.check()
                session.feed_audio(pcm)
        if args.mode == "auto":
            session.finish_input()
        else:
            session.finish()
        consumer.drain(args.drain_timeout)
    except KeyboardInterrupt:
        reason, code = "interrupt", 130
    except Exception:
        reason = "error"
        raise
    finally:
        change = session.close()
        try:
            consumer.close(args.close_timeout)
        except ASRError as exc:
            cleanup_error = exc
        if reason == "interrupt":
            for event in change.cancelled:
                output.event(event, input_index=index, source=source)
        if reason != "error":
            if cleanup_error is not None:
                raise cleanup_error
    return code, reason


def main(argv=None, *, stdin=None, stdout=None, stderr=None):
    """安装入口与模块入口共用同一流程；标准流由进程拥有。"""
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] in ("models", "gpu-runtime"):
        from realtime_asr.cli.resources import main as resources

        return resources(argv[1:], gpu=argv[0] == "gpu-runtime", stdout=stdout)
    if argv and argv[0] == "serve":
        from realtime_asr.server.__main__ import main as serve

        return serve(argv[1:])
    if argv and argv[0] == "remote":
        from realtime_asr.client.__main__ import main as remote

        return remote(argv[1:], stdin=stdin, stdout=stdout, stderr=stderr)
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    args = parse_args(argv)
    stdin = sys.stdin if stdin is None else stdin
    output = Output(
        args.output,
        sys.stdout if stdout is None else stdout,
        sys.stderr if stderr is None else stderr,
    )
    try:
        return _execute(args, stdin, output)
    except BrokenPipeError:
        if output.stdout is sys.stdout:
            # 保留标准流对象；避免解释器退出时再次刷新坏管道而改写退出码。
            null_descriptor = os.open(os.devnull, os.O_WRONLY)
            try:
                os.dup2(null_descriptor, sys.stdout.fileno())
            finally:
                os.close(null_descriptor)
        return 1


def _execute(args, stdin, output):
    """创建并关闭本次调用的资源；输出错误交由入口统一处理。"""
    runtime = microphone = None
    code, index, source = 0, 0, args.command
    reason = "eof"
    try:
        if args.command == "devices":
            output.record(
                "devices",
                channels=args.channels,
                devices=[asdict(d) for d in list_input_devices(channels=args.channels)],
            )
            return 0
        config = RecognizerConfig(
            model_path=args.model,
            backend=args.backend,
            device=args.device,
            cudnn_folder=args.cudnn_folder,
            cuda_directory=args.cuda_directory,
        )
        realtime, ultrafast = realtime_roles(args, config)
        candidate = candidate_role(args, config)
        runtime = ASRRuntime(
            None if args.candidate_only else config,
            realtime=realtime,
            ultrafast=ultrafast,
            candidate=candidate,
            final_enabled=not args.candidate_only,
        )
        output.record("status", status="preparing", input_index=0, source=source)
        runtime.prepare()
        if args.command == "mic":
            microphone = MicrophoneInput(
                MicrophoneConfig(
                    device_index=args.input_device,
                    sample_rate=args.sample_rate,
                    channels=args.channels,
                    chunk_frames=args.chunk_frames,
                    queue_seconds=args.queue_seconds,
                )
            )
            format = microphone.prepare()
            code, reason = _run_input(
                runtime, args, output, 0, source, format, (), microphone=microphone
            )
        elif args.command == "stdin":
            format = AudioFormat(args.sample_rate, args.channels, args.sample_type)
            binary = getattr(stdin, "buffer", stdin)
            code, reason = _run_input(runtime, args, output, 0, source, format, binary)
        else:
            for index, source in enumerate(args.paths):
                with closing(audio_file_chunks(source)) as chunks:
                    format, first = next(chunks)
                    blocks = chain((first,), (chunk for _, chunk in chunks))
                    code, reason = _run_input(runtime, args, output, index, source, format, blocks)
                if code:
                    break
                if index < len(args.paths) - 1:
                    output.record(
                        "status",
                        status="stopped",
                        input_index=index,
                        source=source,
                        reason=reason,
                        exit_code=code,
                    )
    except KeyboardInterrupt:
        code = 130
        reason = "interrupt"
    except BrokenPipeError:
        raise
    except ASRError as exc:
        reason = "error"
        code = (
            4
            if exc.code in ("DRAIN_TIMEOUT", "CLOSE_TIMEOUT", "MICROPHONE_CLEANUP_FAILED")
            else (2 if exc.code == "INVALID_CONFIGURATION" else 1)
        )
        output.record(
            "error",
            code=exc.code,
            operation=exc.operation,
            message=exc.message,
            input_index=index,
            source=source,
        )
    except Exception as exc:
        code = 1
        reason = "error"
        output.record(
            "error",
            code="INPUT_OUTPUT_FAILED",
            operation="cli",
            message=str(exc),
            input_index=index,
            source=source,
        )
    finally:
        failed_cleanup = []
        for owner in (microphone, runtime):
            if owner is not None:
                result = owner.close(timeout=args.close_timeout)
                if result.status != "ok":
                    failed_cleanup.append(asdict(result))
        if failed_cleanup and not output.failed:
            code = code or 4
            if code != 130:
                reason = "error"
            output.record(
                "error",
                code="CLEANUP_FAILED",
                operation="close",
                message="资源尚未成功释放",
                input_index=index,
                source=source,
                cleanup=failed_cleanup,
            )
        if (args.command != "devices" or code) and not output.failed:
            output.record(
                "status",
                status="stopped",
                input_index=index,
                source=source,
                reason=reason,
                exit_code=code,
            )
    return code
