"""面向使用者的音频、文件地址和实时转写入口。"""

from os import PathLike
from pathlib import Path
from sys import exc_info
from time import monotonic

from speechio.adapters.contracts import validate_engine
from speechio.audio.formats import AudioFormat
from speechio.audio.microphone import MicrophoneConfig, MicrophoneInput
from speechio.bundled import default_vad_model
from speechio.errors import ASRError
from speechio.stt.runtime.asr import ASRRuntime
from speechio.stt.runtime.realtime import RealtimeConfig
from speechio.stt.segmentation.gate import SegmentationConfig
from speechio.stt.segmentation.vad import VadConfig
from speechio.stt.session.session import SessionConfig
from speechio.stt.text.endpoint import EndpointConfig
from speechio.stt.transcriber import Transcriber, resolve_model_options


def transcribe_audio(
    audio,
    *,
    config=None,
    model_path=None,
    language=None,
    word_timestamps=False,
    format=None,
    max_audio_seconds=None,
    **options,
):
    """转写一段已有音频；路径读文件，字节或数组按声明格式转换。"""
    transcriber = Transcriber(
        config=config, model_path=model_path, max_audio_seconds=max_audio_seconds, **options
    )
    try:
        # 快捷调用沿用底层的输入校验顺序，不通过 __enter__ 提前准备模型。
        if isinstance(audio, (str, PathLike)):
            return transcriber.transcribe_file(
                audio, language=language, word_timestamps=word_timestamps
            )
        source = format or AudioFormat()
        if isinstance(audio, (bytes, bytearray, memoryview)):
            return transcriber.transcribe(
                audio, format=source, language=language, word_timestamps=word_timestamps
            )
        try:
            import numpy as np
        except ImportError:  # pragma: no cover - 基础依赖固定包含 NumPy
            np = None
        if np is not None and isinstance(audio, np.ndarray):
            return transcriber.transcribe(
                audio, format=source, language=language, word_timestamps=word_timestamps
            )
        return transcriber.transcribe_chunks(
            audio, format=source, language=language, word_timestamps=word_timestamps
        )
    finally:
        transcriber.__exit__(*exc_info())


def transcribe_url(
    url,
    *,
    config=None,
    engine=None,
    language=None,
    word_timestamps=False,
    max_audio_seconds=None,
):
    """转写服务可访问的文件地址；不在本机下载，调用结束后释放引擎。"""
    transcriber = Transcriber(config=config, engine=engine, max_audio_seconds=max_audio_seconds)
    try:
        return transcriber.transcribe_url(url, language=language, word_timestamps=word_timestamps)
    finally:
        transcriber.__exit__(*exc_info())


class StreamTranscriber:
    """持续转写麦克风；按段交付最终结果，预览走回调。"""

    def __init__(
        self,
        *,
        config=None,
        model_path=None,
        vad_model=None,
        language=None,
        on_partial=None,
        input_device=None,
        realtime=True,
        completion_mode="auto",
        drain_timeout=10.0,
        **options,
    ):
        if on_partial is not None and not callable(on_partial):
            raise ASRError("INVALID_CONFIGURATION", "configure", "on_partial 应为可调用对象")
        self._model_path = model_path
        self._config = config
        self._options = options
        self._completion_mode = completion_mode
        # 默认使用随包分发的 Silero v6；显式路径优先，启动前就检查文件。
        self._vad_model = Path(vad_model) if vad_model is not None else default_vad_model()
        if not self._vad_model.is_file():
            raise ASRError(
                "INVALID_CONFIGURATION", "configure", f"检测模型不存在：{self._vad_model}"
            )
        self._language = language
        self._on_partial = on_partial
        self._input_device = input_device
        self._realtime = realtime
        self._drain_timeout = drain_timeout
        self._runtime = None
        self._session = None
        self._microphone = None
        self._ending = False

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            self.close()
        except BaseException as cleanup_error:
            if exc is None:
                raise
            # 清理失败不能覆盖触发退出的推理错误或使用者异常。
            exc.add_note(f"关闭 StreamTranscriber 时发生异常：{cleanup_error!r}")
            for note in getattr(cleanup_error, "__notes__", ()):
                exc.add_note(note)
        return False

    @property
    def state(self):
        return "idle" if self._session is None else "listening"

    def start(self):
        """加载模型、打开麦克风并建立自动分句会话。"""
        if self._session is not None:
            return self
        config, engine = resolve_model_options(
            self._config, model_path=self._model_path, **self._options
        )
        capabilities = config.capabilities if engine is None else validate_engine(engine)
        if self._realtime and capabilities.realtime_mode is None:
            raise ASRError(
                "BACKEND_INCOMPATIBLE",
                "configure",
                "所选模型不支持实时预览；请选流式模型，或设置 realtime=False 仅获取分段最终结果",
            )
        runtime = ASRRuntime(
            config,
            engine=engine,
            realtime=RealtimeConfig(mode=capabilities.realtime_mode) if self._realtime else None,
        )
        microphone = None
        try:
            runtime.prepare()
            microphone = MicrophoneInput(MicrophoneConfig(device_index=self._input_device))
            format = microphone.prepare()
            session = runtime.create_auto_session(
                config=SessionConfig(
                    format=format,
                    language=self._language,
                    realtime=self._realtime,
                    completion_mode=self._completion_mode,
                ),
                vad=VadConfig(model_path=self._vad_model),
                segmentation=SegmentationConfig(),
                endpoint=EndpointConfig(),
            )
            microphone.start()
        except BaseException:
            if microphone is not None:
                microphone.close(timeout=2)
            runtime.close()
            raise
        self._runtime, self._session, self._microphone = runtime, session, microphone
        return self

    def end_input(self):
        """停止采集，送完采集队列后结束输入；继续迭代可取得剩余结果。"""
        if self._ending or self._session is None:
            return
        if self._microphone is not None:
            self._stop_microphone(self._microphone)
            # 采集器关闭不会清空队列，必须先交付其中的成功采集块。
            while (pcm := self._microphone.read_audio(timeout=0)) is not None:
                self._session.feed_audio(pcm)
        self._session.end_input()
        self._ending = True

    def _stop_microphone(self, microphone):
        """限时等待读取退出，避免将暂时空队列误认为最后一块已到达。"""
        deadline = monotonic() + self._drain_timeout
        result = microphone.close(timeout=min(2, max(0, self._drain_timeout)))
        while result.status == "pending":
            remaining = deadline - monotonic()
            if remaining <= 0:
                raise ASRError("CLOSE_TIMEOUT", "close_microphone", "等待采集结束超时")
            result = microphone.close(timeout=remaining)
        if result.status != "ok":
            error = ASRError("MICROPHONE_CLEANUP_FAILED", "close_microphone", "采集器清理失败")
            error.cleanup_errors = result.errors
            raise error

    def close(self):
        """清理采集器、会话和运行时；重复调用无副作用。"""
        session, microphone, runtime = self._session, self._microphone, self._runtime
        errors = []
        try:
            if session is not None:
                self.end_input()
                self._drain(session)
            elif microphone is not None:
                self._stop_microphone(microphone)
        except BaseException as exc:
            errors.append(exc)
        finally:
            self._session = self._microphone = self._runtime = None
            self._ending = False

        # 每个资源独立清理；排空或前一个资源失败不能跳过后续释放。
        if session is not None:
            try:
                session.close()
            except BaseException as exc:
                errors.append(exc)
        if runtime is not None:
            try:
                result = runtime.close()
                if result.status != "ok":
                    code = "CLOSE_TIMEOUT" if result.status == "pending" else "CLEANUP_FAILED"
                    error = ASRError(code, "close", "运行时关闭未成功")
                    error.cleanup_errors = result.errors
                    raise error
            except BaseException as exc:
                errors.append(exc)
        if errors:
            first, *remaining = errors
            for error in remaining:
                first.add_note(f"后续清理异常：{error!r}")
            raise first

    def _drain(self, session):
        """关闭前取走已完成的最终结果，避免直接丢弃。"""
        deadline = monotonic() + self._drain_timeout
        while monotonic() < deadline:
            event = session.next_event(timeout=0.05)
            if event is not None:
                self._handle(event)
                continue
            if session.pending_segments == 0:
                return

    def __iter__(self):
        self.start()
        try:
            while True:
                if not self._ending:
                    pcm = self._microphone.read_audio(timeout=0.05)
                    if pcm:
                        self._session.feed_audio(pcm)
                event = self._session.next_event(timeout=0.05 if self._ending else 0)
                if event is None:
                    if self._ending and self._session.pending_segments == 0:
                        return
                    continue
                transcript = self._handle(event)
                if transcript is not None:
                    yield transcript
        finally:
            self.__exit__(*exc_info())

    def _handle(self, event):
        """过滤事件：最终正文交付，预览走回调，失败转为异常。"""
        if event.kind == "final" and event.result is not None:
            return event.result
        if event.kind == "realtime":
            if self._on_partial is not None:
                if event.merge is not None:
                    text = event.merge.text
                elif event.stabilization is not None:
                    text = event.stabilization.display_text
                elif event.result is not None:
                    text = event.result.text
                else:
                    text = ""
                self._on_partial(text)
            return None
        if event.kind in ("failed", "realtime_failed", "correction_failed"):
            error = event.error
            raise ASRError(error.code, error.operation, error.message)
        return None


def transcribe_stream(
    *, config=None, model_path=None, vad_model=None, language=None, on_partial=None, **options
):
    """持续转写麦克风；每次产出一段的最终结果，默认使用内置检测模型。"""
    with StreamTranscriber(
        config=config,
        model_path=model_path,
        vad_model=vad_model,
        language=language,
        on_partial=on_partial,
        **options,
    ) as transcriber:
        yield from transcriber
