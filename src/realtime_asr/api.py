"""面向使用者的两个入口：转写一段音频，或持续转写实时音频。"""

from os import PathLike
from pathlib import Path

from realtime_asr.audio.conversion import AudioFormat
from realtime_asr.audio.microphone import MicrophoneConfig, MicrophoneInput
from realtime_asr.bundled import default_vad_model
from realtime_asr.errors import ASRError
from realtime_asr.runtime.asr import ASRRuntime
from realtime_asr.runtime.realtime import RealtimeConfig
from realtime_asr.runtime.transcriber import ModelConfig, Transcriber
from realtime_asr.segmentation.gate import SegmentationConfig
from realtime_asr.segmentation.vad import VadConfig
from realtime_asr.session.session import SessionConfig
from realtime_asr.text.endpoint import EndpointConfig


def transcribe_audio(
    audio,
    *,
    model="tiny",
    language=None,
    word_timestamps=False,
    format=None,
    **options,
):
    """转写一段已有音频；路径读文件，字节或数组按声明格式转换。"""
    with Transcriber(model=model, **options) as transcriber:
        if isinstance(audio, (str, PathLike)):
            return transcriber.transcribe_file(
                audio, language=language, word_timestamps=word_timestamps
            )
        source = format or AudioFormat()
        if isinstance(audio, (bytes, bytearray, memoryview)):
            return transcriber.transcribe(
                bytes(audio), format=source, language=language, word_timestamps=word_timestamps
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


class StreamTranscriber:
    """持续转写麦克风；按段交付最终结果，预览走回调。"""

    def __init__(
        self,
        *,
        model="tiny",
        vad_model=None,
        language=None,
        on_partial=None,
        input_device=None,
        realtime=True,
        drain_timeout=10.0,
        **options,
    ):
        if on_partial is not None and not callable(on_partial):
            raise ASRError("INVALID_CONFIGURATION", "configure", "on_partial 应为可调用对象")
        self._model = model
        self._options = options
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
        self.close()
        return False

    @property
    def state(self):
        return "idle" if self._session is None else "listening"

    def start(self):
        """加载模型、打开麦克风并建立自动分句会话。"""
        if self._session is not None:
            return self
        config = ModelConfig(model=self._model, **self._options)
        runtime = ASRRuntime(
            config,
            realtime=RealtimeConfig(mode=config.capabilities.realtime_mode)
            if self._realtime
            else None,
        )
        microphone = None
        try:
            runtime.prepare()
            microphone = MicrophoneInput(MicrophoneConfig(device_index=self._input_device))
            format = microphone.prepare()
            session = runtime.create_auto_session(
                config=SessionConfig(
                    format=format, language=self._language, realtime=self._realtime
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
        """停止采集并结束输入；已固定的段落仍会交付。"""
        if self._ending or self._session is None:
            return
        self._ending = True
        if self._microphone is not None:
            self._microphone.close(timeout=2)
        self._session.end_input()

    def close(self):
        """清理采集器、会话和运行时；重复调用无副作用。"""
        session, microphone, runtime = self._session, self._microphone, self._runtime
        self._session = self._microphone = self._runtime = None
        self._ending = False
        if session is not None:
            try:
                try:
                    if microphone is not None:
                        microphone.close(timeout=2)
                    session.end_input()
                except ASRError:
                    pass
                self._drain(session)
            finally:
                session.close()
        elif microphone is not None:
            microphone.close(timeout=2)
        if runtime is not None:
            runtime.close()

    def _drain(self, session):
        """关闭前取走已完成的最终结果，避免直接丢弃。"""
        from time import monotonic

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
            self.close()

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


def transcribe_stream(*, model="tiny", vad_model=None, language=None, on_partial=None, **options):
    """持续转写麦克风；每次产出一段的最终结果，默认使用内置检测模型。"""
    with StreamTranscriber(
        model=model,
        vad_model=vad_model,
        language=language,
        on_partial=on_partial,
        **options,
    ) as transcriber:
        yield from transcriber
