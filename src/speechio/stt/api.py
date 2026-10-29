"""面向使用者的音频、文件地址和实时转写入口。"""

from os import PathLike
from pathlib import Path
from sys import exc_info
from time import monotonic

from speechio._async import cancellable, run_operation
from speechio.adapters.contracts import validate_engine
from speechio.audio.microphone import MicrophoneConfig, MicrophoneInput
from speechio.bundled import default_vad_model
from speechio.errors import STTError
from speechio.models.factory import engine_capabilities
from speechio.stt.options import resolve_options
from speechio.stt.runtime.asr import STTRuntime
from speechio.stt.runtime.realtime import RealtimeConfig
from speechio.stt.segmentation.gate import SegmentationConfig
from speechio.stt.segmentation.vad import VadConfig
from speechio.stt.session.session import SessionConfig
from speechio.stt.text.endpoint import SpeechEndConfig
from speechio.stt.transcriber import Transcriber


@cancellable(STTError)
def transcribe_audio(
    audio,
    *,
    config,
    format=None,
    options=None,
    max_audio_seconds=None,
    task_timeout_seconds=3600.0,
    cancellation=None,
):
    """识别一次文件或裸音频；裸音频必须声明格式。"""
    transcriber = Transcriber(config, max_audio_seconds=max_audio_seconds)
    try:
        if isinstance(audio, (str, PathLike)):
            return transcriber.transcribe_file(
                audio, options=options, task_timeout_seconds=task_timeout_seconds
            )
        if format is None:
            raise STTError("INVALID_CONFIGURATION", "transcribe", "裸音频需要显式提供 AudioFormat")
        import numpy as np

        if isinstance(audio, (bytes, bytearray, memoryview, np.ndarray)):
            return transcriber.transcribe(audio, format=format, options=options)
        return transcriber.transcribe_chunks(audio, format=format, options=options)
    finally:
        transcriber.__exit__(*exc_info())


@cancellable(STTError)
def transcribe_url(
    url,
    *,
    config,
    options=None,
    max_audio_seconds=None,
    task_timeout_seconds=3600.0,
    cancellation=None,
):
    """识别服务可访问的文件地址，不在本机下载。"""
    transcriber = Transcriber(config, max_audio_seconds=max_audio_seconds)
    try:
        return transcriber.transcribe_url(
            url, options=options, task_timeout_seconds=task_timeout_seconds
        )
    finally:
        transcriber.__exit__(*exc_info())


async def transcribe_audio_async(
    audio,
    *,
    config,
    format=None,
    options=None,
    max_audio_seconds=None,
    task_timeout_seconds=3600.0,
    cancellation=None,
):
    """一次性异步识别文件、裸音频或有限同步块序列。"""
    return await run_operation(
        transcribe_audio,
        audio,
        config=config,
        format=format,
        options=options,
        max_audio_seconds=max_audio_seconds,
        task_timeout_seconds=task_timeout_seconds,
        cancellation=cancellation,
    )


async def transcribe_url_async(
    url,
    *,
    config,
    options=None,
    max_audio_seconds=None,
    task_timeout_seconds=3600.0,
    cancellation=None,
):
    """异步识别远程文件地址；取消停止本地等待，不承诺撤回云端任务。"""
    return await run_operation(
        transcribe_url,
        url,
        config=config,
        options=options,
        max_audio_seconds=max_audio_seconds,
        task_timeout_seconds=task_timeout_seconds,
        cancellation=cancellation,
    )


class MicrophoneTranscriber:
    """持续转写麦克风；按段交付最终结果，预览走回调。"""

    def __init__(
        self,
        config,
        *,
        options=None,
        microphone=None,
        vad=None,
        segmentation=None,
        endpoint=None,
        on_partial_text=None,
        realtime=True,
        completion_mode="auto",
        drain_timeout_seconds=10.0,
    ):
        self._initialize(
            config,
            None,
            options,
            microphone,
            vad,
            segmentation,
            endpoint,
            on_partial_text,
            realtime,
            completion_mode,
            drain_timeout_seconds,
        )

    @classmethod
    def from_adapter(
        cls,
        adapter,
        *,
        options=None,
        microphone=None,
        vad=None,
        segmentation=None,
        endpoint=None,
        on_partial_text=None,
        realtime=True,
        completion_mode="auto",
        drain_timeout_seconds=10.0,
    ):
        """使用自定义识别适配器采集麦克风。"""
        validate_engine(adapter)
        instance = cls.__new__(cls)
        instance._initialize(
            None,
            adapter,
            options,
            microphone,
            vad,
            segmentation,
            endpoint,
            on_partial_text,
            realtime,
            completion_mode,
            drain_timeout_seconds,
        )
        return instance

    def _initialize(
        self,
        config,
        adapter,
        options,
        microphone,
        vad,
        segmentation,
        endpoint,
        on_partial_text,
        realtime,
        completion_mode,
        drain_timeout_seconds,
    ):
        from speechio.stt.session.session import _validate_timeout

        if on_partial_text is not None and not callable(on_partial_text):
            raise STTError("INVALID_CONFIGURATION", "configure", "on_partial_text 应为函数")
        if completion_mode not in ("auto", "final"):
            raise STTError("INVALID_CONFIGURATION", "configure", "麦克风入口只支持 auto / final")
        _validate_timeout(drain_timeout_seconds)
        self._config, self._adapter = config, adapter
        self._options = resolve_options(options)
        self._microphone_config = microphone if microphone is not None else MicrophoneConfig()
        self._vad = vad if vad is not None else VadConfig(model_path=default_vad_model())
        if not isinstance(self._vad, VadConfig) or not Path(self._vad.model_path).is_file():
            raise STTError("INVALID_CONFIGURATION", "configure", "VAD 模型路径不存在")
        self._segmentation = segmentation if segmentation is not None else SegmentationConfig()
        self._endpoint = endpoint if endpoint is not None else SpeechEndConfig()
        self._on_partial = on_partial_text
        self._realtime = realtime
        self._completion_mode = completion_mode
        self._drain_timeout = drain_timeout_seconds
        self._runtime = self._session = self._microphone = None
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
            exc.add_note(f"关闭 MicrophoneTranscriber 时发生异常：{cleanup_error!r}")
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
        capabilities = (
            engine_capabilities(self._config)
            if self._adapter is None
            else validate_engine(self._adapter)
        )
        if self._realtime and capabilities.realtime_mode is None:
            raise STTError(
                "BACKEND_INCOMPATIBLE",
                "configure",
                "所选模型不支持实时预览；请选流式模型，或设置 realtime=False 仅获取分段最终结果",
            )
        realtime = RealtimeConfig(mode=capabilities.realtime_mode) if self._realtime else None
        runtime = (
            STTRuntime(self._config, realtime=realtime)
            if self._adapter is None
            else STTRuntime.from_adapter(self._adapter, realtime=realtime)
        )
        microphone = None
        try:
            runtime.prepare()
            microphone = MicrophoneInput(self._microphone_config)
            format = microphone.prepare()
            session = runtime.create_auto_session(
                config=SessionConfig(
                    format=format,
                    options=self._options,
                    realtime=self._realtime,
                    completion_mode=self._completion_mode,
                ),
                vad=self._vad,
                segmentation=self._segmentation,
                endpoint=self._endpoint,
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
                raise STTError("CLOSE_TIMEOUT", "close_microphone", "等待采集结束超时")
            result = microphone.close(timeout=remaining)
        if result.status != "ok":
            error = STTError("MICROPHONE_CLEANUP_FAILED", "close_microphone", "采集器清理失败")
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
                    error = STTError(code, "close", "运行时关闭未成功")
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
            raise STTError(error.code, error.operation, error.message)
        return None


def transcribe_microphone(
    *,
    config,
    options=None,
    microphone=None,
    vad=None,
    segmentation=None,
    endpoint=None,
    on_partial_text=None,
    realtime=True,
    completion_mode="auto",
    drain_timeout_seconds=10.0,
):
    """快捷采集麦克风，每次产出一个音频段的最终结果。"""
    with MicrophoneTranscriber(
        config,
        options=options,
        microphone=microphone,
        vad=vad,
        segmentation=segmentation,
        endpoint=endpoint,
        on_partial_text=on_partial_text,
        realtime=realtime,
        completion_mode=completion_mode,
        drain_timeout_seconds=drain_timeout_seconds,
    ) as transcriber:
        yield from transcriber
