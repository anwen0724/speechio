"""设备采集拥有 PortAudio 资源；音频转换与识别由调用方组织。"""

import sys
from collections import deque
from dataclasses import dataclass, replace
from math import isfinite
from threading import Condition, Event, Thread

from realtime_asr.audio.conversion import AudioFormat
from realtime_asr.errors import ASRError
from realtime_asr.results import CloseError, CloseResult

SAMPLE_RATES = (8000, 9600, 11025, 12000, 16000, 22050, 24000, 32000, 44100, 48000)


@dataclass(frozen=True)
class InputDevice:
    """PortAudio 实际输入设备与按给定声道探测的格式。"""

    index: int
    name: str
    host_api: str
    max_input_channels: int
    default_sample_rate: int
    is_default: bool
    supported_sample_rates: tuple[int, ...]


@dataclass(frozen=True, kw_only=True)
class MicrophoneConfig:
    """明确选择设备与源音频预算，不声明已重采样后的格式。"""

    device_index: int | None = None
    sample_rate: int | None = None
    channels: int = 1
    chunk_frames: int = 1024
    queue_seconds: float = 5

    def __post_init__(self):
        values = (self.channels, self.chunk_frames)
        if self.sample_rate is not None:
            values += (self.sample_rate,)
        if (
            any(type(value) is not int or value <= 0 for value in values)
            or (
                self.device_index is not None
                and (type(self.device_index) is not int or self.device_index < 0)
            )
            or (
                not isinstance(self.queue_seconds, (int, float))
                or isinstance(self.queue_seconds, bool)
                or not isfinite(self.queue_seconds)
                or self.queue_seconds <= 0
            )
        ):
            raise ASRError("INVALID_CONFIGURATION", "configure_microphone", "麦克风配置无效")


def _portaudio():
    """只在显式设备操作时加载可选依赖。"""
    try:
        import pyaudio
    except ImportError as exc:
        raise ASRError(
            "OPTIONAL_DEPENDENCY_MISSING", "open_microphone", "请安装 realtime-asr[microphone]"
        ) from exc
    return pyaudio


def _device(interface, index, channels, default_index, pcm_type):
    """查询实际设备，不用设备名称替代索引，也不切换无效指定设备。"""
    try:
        info = interface.get_device_info_by_index(index)
    except Exception as exc:
        raise ASRError("INPUT_DEVICE_INVALID", "select_device", str(exc)) from exc
    if info["maxInputChannels"] < channels:
        raise ASRError("INPUT_DEVICE_INVALID", "select_device", "设备没有所需输入声道")
    rates = []
    for rate in SAMPLE_RATES:
        try:
            if interface.is_format_supported(
                rate, input_device=index, input_channels=channels, input_format=pcm_type
            ):
                rates.append(rate)
        except ValueError:
            pass
    return InputDevice(
        index,
        info["name"],
        interface.get_host_api_info_by_index(info["hostApi"])["name"],
        int(info["maxInputChannels"]),
        int(info["defaultSampleRate"]),
        index == default_index,
        tuple(rates),
    )


def list_input_devices(*, channels=1):
    """枚举拥有输入声道的设备；操作结束后回收自己的 PortAudio 实例。"""
    if type(channels) is not int or channels <= 0:
        raise ASRError("INVALID_CONFIGURATION", "list_devices", "声道应为正整数")
    library = _portaudio()
    interface = None
    try:
        interface = library.PyAudio()
        try:
            default_index = int(interface.get_default_input_device_info()["index"])
        except OSError:
            default_index = None
        devices = []
        for index in range(interface.get_device_count()):
            info = interface.get_device_info_by_index(index)
            if info["maxInputChannels"] > 0:
                # 声道不适合的输入设备仍可见，支持率为空。
                query_channels = min(channels, int(info["maxInputChannels"]))
                device = _device(interface, index, query_channels, default_index, library.paInt16)
                if query_channels != channels:
                    device = replace(device, supported_sample_rates=())
                devices.append(device)
        return tuple(devices)
    except ASRError:
        raise
    except Exception as exc:
        raise ASRError("DEVICE_ENUMERATION_FAILED", "list_devices", str(exc)) from exc
    finally:
        if interface is not None:
            primary = sys.exception()
            try:
                interface.terminate()
            except Exception as exc:
                cleanup = CloseError("MICROPHONE_CLEANUP_FAILED", "terminate_portaudio", str(exc))
                if primary is not None:
                    primary.cleanup_errors = (cleanup,)
                else:
                    raise ASRError(cleanup.code, cleanup.operation, cleanup.message) from exc


class MicrophoneInput:
    """独立设备输入拥有者，不持有 ASR、VAD 或会话对象。"""

    def __init__(self, config=None):
        self.config = config if config is not None else MicrophoneConfig()
        if not isinstance(self.config, MicrophoneConfig):
            raise ASRError("INVALID_CONFIGURATION", "configure_microphone", "需要 MicrophoneConfig")
        self.state = "new"
        self.format = None
        self.device = None
        self._interface = None
        self._stream = None
        self._close_result = None
        self._condition = Condition()
        self._stop = Event()
        self._worker = None
        self._queue = deque()
        self.received_frames = 0
        self.buffered_frames = 0
        self.error = None

    def prepare(self):
        """实际打开选中设备，优先 16 kHz，显式格式不自动降级。"""
        if self.state in ("ready", "running"):
            return self.format
        if self.state != "new":
            raise ASRError("MICROPHONE_CLOSED", "prepare_microphone", "采集器不能重新准备")
        library = _portaudio()
        try:
            self._interface = library.PyAudio()
            default_index = None
            try:
                default_index = int(self._interface.get_default_input_device_info()["index"])
            except OSError:
                if self.config.device_index is None:
                    raise ASRError("INPUT_DEVICE_INVALID", "select_device", "没有系统默认输入设备")
            index = (
                self.config.device_index if self.config.device_index is not None else default_index
            )
            self.device = _device(
                self._interface, index, self.config.channels, default_index, library.paInt16
            )
            candidates = (
                [self.config.sample_rate]
                if self.config.sample_rate
                else list(
                    dict.fromkeys([16000, self.device.default_sample_rate, *reversed(SAMPLE_RATES)])
                )
            )
            rate = None
            for candidate in candidates:
                try:
                    if self._interface.is_format_supported(
                        candidate,
                        input_device=index,
                        input_channels=self.config.channels,
                        input_format=library.paInt16,
                    ):
                        rate = candidate
                        break
                except ValueError:
                    continue
            if rate is None:
                raise ASRError("DEVICE_FORMAT_UNSUPPORTED", "open_microphone", "设备不支持请求格式")
            format = AudioFormat(rate, self.config.channels, "int16")
            if int(rate * self.config.queue_seconds) < self.config.chunk_frames:
                raise ASRError("INVALID_CONFIGURATION", "open_microphone", "缓冲不足一个采集块")
            self._stream = self._interface.open(
                format=library.paInt16,
                rate=rate,
                channels=self.config.channels,
                input=True,
                input_device_index=index,
                frames_per_buffer=self.config.chunk_frames,
                start=False,
            )
            self.format = format
            self.state = "ready"
            return self.format
        except BaseException as exc:
            cleanup = self._release()
            if cleanup.errors:
                exc.cleanup_errors = cleanup.errors
            if isinstance(exc, (ASRError, KeyboardInterrupt, SystemExit)):
                raise
            raise ASRError("MICROPHONE_OPEN_FAILED", "open_microphone", str(exc)) from exc

    def _release(self):
        """每项资源分别清理，前一项失败仍继续执行后续项。"""
        errors = []
        with self._condition:
            stream, self._stream = self._stream, None
            interface, self._interface = self._interface, None
        operations = []
        if stream is not None:
            operations.extend((("stop_stream", stream.stop_stream), ("close_stream", stream.close)))
        if interface is not None:
            operations.append(("terminate_portaudio", interface.terminate))
        for name, operation in operations:
            try:
                operation()
            except Exception as exc:
                errors.append(CloseError("MICROPHONE_CLEANUP_FAILED", name, str(exc)))
        with self._condition:
            self.state = "closed"
            self._close_result = CloseResult(
                status="failed" if errors else "ok",
                cleanup_method="portaudio_close",
                errors=tuple(errors),
            )
            self._condition.notify_all()
            return self._close_result

    def close(self, *, timeout=2):
        """停止接收并限时等待；读调用实际退出前不销毁句柄。"""
        _timeout(timeout)
        with self._condition:
            if self._close_result is not None:
                self.state = "closed"
                return self._close_result
            self._stop.set()
            self.state = "stopping"
            worker = self._worker
            if worker is None:
                if self._interface is None:
                    return self._release()
                worker = self._worker = Thread(
                    target=self._release, name="ASRMicrophoneInput", daemon=True
                )
                worker.start()
        worker.join(timeout)
        with self._condition:
            if worker.is_alive():
                return CloseResult(status="pending", cleanup_method="portaudio_close")
            self.state = "closed"
            return self._close_result

    def start(self):
        """独立读取源块，识别和转换不会进入采集线程。"""
        with self._condition:
            if self.state == "running":
                return self.format
            if self.state == "new":
                self.prepare()
            if self.state != "ready":
                raise ASRError("MICROPHONE_CLOSED", "start_microphone", "采集器不可启动")
            try:
                self._stream.start_stream()
                self.state = "running"
                self._worker = Thread(target=self._capture, name="ASRMicrophoneInput", daemon=True)
                self._worker.start()
            except BaseException as exc:
                self._worker = None
                cleanup = self._release()
                if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                    exc.cleanup_errors = cleanup.errors
                    raise
                error = ASRError("MICROPHONE_OPEN_FAILED", "start_microphone", str(exc))
                error.cleanup_errors = cleanup.errors
                raise error from exc
            return self.format

    def _capture(self):
        """只接收成功块；溢出和满缓冲使本次输入失败，不能丢旧数据。"""
        library = _portaudio()
        limit = int(self.format.sample_rate * self.config.queue_seconds)
        try:
            while not self._stop.is_set():
                try:
                    pcm = self._stream.read(self.config.chunk_frames, exception_on_overflow=True)
                except OSError as exc:
                    code = (
                        "MICROPHONE_OVERFLOW"
                        if exc.errno == library.paInputOverflowed
                        else "MICROPHONE_READ_FAILED"
                    )
                    raise ASRError(code, "read_microphone", str(exc)) from exc
                frame_bytes = self.format.channels * 2
                if not pcm or len(pcm) % frame_bytes:
                    raise ASRError("MICROPHONE_READ_FAILED", "read_microphone", "设备返回无效源帧")
                frames = len(pcm) // frame_bytes
                with self._condition:
                    if self.buffered_frames + frames > limit:
                        raise ASRError(
                            "CAPTURE_CAPACITY_EXCEEDED", "read_microphone", "采集缓冲已满，输入停止"
                        )
                    self._queue.append(bytes(pcm))
                    self.buffered_frames += frames
                    self.received_frames += frames
                    self._condition.notify_all()
        except Exception as exc:
            with self._condition:
                self.error = ASRError(
                    exc.code if isinstance(exc, ASRError) else "MICROPHONE_READ_FAILED",
                    "read_microphone",
                    str(exc),
                )
                self.state = "failed"
                self._condition.notify_all()
        finally:
            self._release()
            with self._condition:
                if self.error is not None:
                    self.state = "failed"
                self._condition.notify_all()

    def read_audio(self, *, timeout=0):
        """故障前成功入队的块先交付，之后明确交付故障，不伪装 EOF。"""
        _timeout(timeout, allow_none=True)
        with self._condition:
            self._condition.wait_for(
                lambda: self._queue or self.error is not None or self._close_result is not None,
                timeout=timeout,
            )
            if self._queue:
                pcm = self._queue.popleft()
                self.buffered_frames -= len(pcm) // (self.format.channels * 2)
                return pcm
            if self.error is not None:
                raise ASRError(self.error.code, self.error.operation, self.error.message)
            return None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        result = self.close()
        if exc_type is None and result.status != "ok":
            raise ASRError("MICROPHONE_CLEANUP_FAILED", "close_microphone", result.status)
        if exc is not None and result.errors:
            exc.cleanup_errors = result.errors
        return False


def _timeout(value, *, allow_none=False):
    """在等待接口边界校验秒数，不向驱动传入无效超时。"""
    if value is None and allow_none:
        return
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not isfinite(value)
        or value < 0
    ):
        raise ASRError("INVALID_CONFIGURATION", "wait_microphone", "等待秒数应为有限非负值")
