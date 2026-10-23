"""标准音频边界与连续转换；保留 P0 固定格式入口。"""

from dataclasses import dataclass
from math import isfinite

import numpy as np

from speechio.audio.formats import AudioFormat, _positive_integer
from speechio.errors import STTError

SAMPLE_RATE = 16000


def validate_max_audio_seconds(value):
    """时长上限由调用方选择；None 不限，正整数和有限正小数均可。"""
    if value is not None and (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or value <= 0
        or (isinstance(value, float) and not isfinite(value))
    ):
        raise STTError(
            "INVALID_CONFIGURATION", "configure", "max_audio_seconds 应为有限正数或 None"
        )


@dataclass(frozen=True)
class AudioFrame:
    """携带有效 PCM16 和标准采样位置，不包含补零或源字节偏移。"""

    pcm: bytes
    start_sample: int
    end_sample: int


class AudioConverter:
    """一段连续音频的转换拥有者，隐藏源余量、滤波历史和输出凑帧。"""

    def __init__(self, format: AudioFormat, *, frame_samples: int = 512, max_audio_seconds=None):
        if not isinstance(format, AudioFormat) or not _positive_integer(frame_samples):
            raise STTError("INVALID_AUDIO_FORMAT", "convert_audio", "格式或标准块长无效")
        self._format = format
        validate_max_audio_seconds(max_audio_seconds)
        self._max_audio_seconds = max_audio_seconds
        self._source_bytes = 0
        self._frame_samples = frame_samples
        self._state = "open"
        self._pending = bytearray()
        self._position = 0
        self._source_remainder = b""
        self._representation = None
        self._resampler = None

    @property
    def format(self) -> AudioFormat:
        return self._format

    @property
    def frame_samples(self) -> int:
        return self._frame_samples

    @property
    def state(self) -> str:
        return self._state

    def feed(self, chunk) -> list[AudioFrame]:
        """接受一个源块，只交付凑满的标准块。"""
        self._require_open()
        source_bytes = 0
        if self._max_audio_seconds is not None:
            scalar_bytes = 2 if self.format.sample_type == "int16" else 4
            if isinstance(chunk, (bytes, bytearray, memoryview)):
                source_bytes = chunk.nbytes if isinstance(chunk, memoryview) else len(chunk)
            elif isinstance(chunk, np.ndarray):
                source_bytes = chunk.size * scalar_bytes
            # 在复制、转换本块前拒绝超限输入；源字节余量也跨块累计。
            seconds = (self._source_bytes + source_bytes) / (
                self.format.sample_rate * self.format.channels * scalar_bytes
            )
            if seconds > self._max_audio_seconds:
                raise STTError("AUDIO_CAPACITY_EXCEEDED", "convert_audio", "音频超过配置的时长上限")
        array, representation, remainder, nonempty = self._parse_chunk(chunk)
        if not nonempty:
            return []
        # 当前块完全校验后才接纳；拒绝不会消费此前跨块余量。
        self._source_remainder = remainder
        self._source_bytes += source_bytes
        self._representation = representation
        mono = array.astype(np.float32)
        if self.format.channels > 1:
            mono = mono.mean(axis=1)
        else:
            mono = mono.reshape(-1)
        if mono.size:
            self._append_pcm(self._resample(mono))
        return self._take_frames()

    def drain_ready(self) -> list[AudioFrame]:
        """交付已经生成的标准短块，不结束输入或刷新重采样滤波器。"""
        self._require_open()
        return self._take_frames(tail=True)

    def finish(self) -> list[AudioFrame]:
        """正常结束输入，交付滤波输出和有效短尾块。"""
        if self.state == "finished":
            return []
        self._require_open()
        if self._source_remainder:
            self._state = "failed"
            raise STTError("AUDIO_INPUT_TRUNCATED", "convert_audio", "尾部缺少完整源采样帧")
        if self._resampler is not None:
            self._append_pcm(self._resample(np.empty(0, np.float32), last=True))
        self._state = "finished"
        return self._take_frames(tail=True)

    def close(self) -> None:
        """丢弃未交付数据；关闭不隐含正常结束。"""
        self._pending.clear()
        self._source_remainder = b""
        self._resampler = None
        self._state = "closed"

    def _resample(self, mono, *, last=False):
        """每段只建立一份 HQ 滤波状态，结束时恰好刷新一次。"""
        if self.format.sample_rate == SAMPLE_RATE:
            return mono
        try:
            if self._resampler is None:
                try:
                    import soxr
                except (ImportError, OSError) as exc:
                    raise STTError(
                        "DEPENDENCY_MISSING", "convert_audio", "请安装 audio 依赖"
                    ) from exc
                self._resampler = soxr.ResampleStream(
                    self.format.sample_rate, SAMPLE_RATE, 1, dtype="float32", quality="HQ"
                )
            return self._resampler.resample_chunk(mono, last=last)
        except STTError:
            raise
        except Exception as exc:
            self._state = "failed"
            raise STTError("AUDIO_CONVERSION_FAILED", "convert_audio", str(exc)) from exc

    def _require_open(self):
        codes = {
            "finished": "CONVERTER_FINISHED",
            "failed": "AUDIO_CONVERSION_FAILED",
            "closed": "CONVERTER_CLOSED",
        }
        if self.state != "open":
            raise STTError(codes[self.state], "convert_audio", "转换器不再接受输入")

    def _parse_chunk(self, chunk):
        """校验外部表示并解析完整源帧；只返回候选余量，不改写状态。"""
        dtype = "<i2" if self.format.sample_type == "int16" else "<f4"
        if isinstance(chunk, (bytes, bytearray, memoryview)):
            if isinstance(chunk, memoryview) and (
                chunk.ndim != 1 or chunk.itemsize != 1 or not chunk.c_contiguous
            ):
                raise STTError("INVALID_AUDIO", "convert_audio", "memoryview 应为连续字节")
            representation = "bytes"
            nonempty = len(chunk) > 0
            raw = self._source_remainder + bytes(chunk)
            frame_bytes = self.format.channels * np.dtype(dtype).itemsize
            count = len(raw) // frame_bytes * frame_bytes
            array = np.frombuffer(raw[:count], dtype=dtype).reshape(-1, self.format.channels)
            remainder = raw[count:]
        elif isinstance(chunk, np.ndarray):
            representation = "array"
            valid_shape = (chunk.ndim == 1 and self.format.channels == 1) or (
                chunk.ndim == 2 and chunk.shape[1] == self.format.channels
            )
            valid_dtype = (
                chunk.dtype.kind == "i" and chunk.dtype.itemsize == 2
                if self.format.sample_type == "int16"
                else chunk.dtype.kind == "f" and chunk.dtype.itemsize in (4, 8)
            )
            if not valid_shape or not valid_dtype:
                raise STTError("INVALID_AUDIO", "convert_audio", "数组形状或类型不符合格式")
            array = chunk.astype(np.float32) if self.format.sample_type == "float32" else chunk
            remainder = b""
            nonempty = chunk.size > 0
        else:
            raise STTError("INVALID_AUDIO", "convert_audio", "输入应为字节或 NumPy 数组")
        if nonempty and self._representation not in (None, representation):
            raise STTError("INVALID_AUDIO", "convert_audio", "同一音频不能混用字节和数组")
        if not np.isfinite(array).all():
            raise STTError("INVALID_AUDIO", "convert_audio", "输入含非有限采样")
        return array, representation, remainder, nonempty

    def _append_pcm(self, mono):
        """在混声和滤波之后量化；不执行峰值归一化。"""
        if self.format.sample_type == "float32":
            quantized = np.clip(mono, -1, 1) * 32767
        else:
            quantized = np.clip(mono, -32768, 32767)
        self._pending.extend(quantized.astype("<i2").tobytes())

    def _take_frames(self, *, tail=False) -> list[AudioFrame]:
        """凑帧只改变交付边界；位置按实际 PCM 长度推进。"""
        frames = []
        block_bytes = self.frame_samples * 2
        consumed = 0
        while len(self._pending) - consumed >= block_bytes:
            end = consumed + block_bytes
            frames.append(self._frame(bytes(self._pending[consumed:end])))
            consumed = end
        if tail and consumed < len(self._pending):
            frames.append(self._frame(bytes(self._pending[consumed:])))
            consumed = len(self._pending)
        del self._pending[:consumed]
        return frames

    def _frame(self, pcm: bytes) -> AudioFrame:
        end = self._position + len(pcm) // 2
        frame = AudioFrame(pcm, self._position, end)
        self._position = end
        return frame


def as_audio(audio, *, allow_empty=False, allow_column=False):
    """将合法单声道输入转换为连续 float32。"""
    if audio is None:
        raise STTError("INVALID_AUDIO", "audio", "音频不能为 None")
    try:
        array = np.asarray(audio, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise STTError("INVALID_AUDIO", "audio", "音频应为数值数组") from exc
    if allow_column and array.ndim == 2 and array.shape[1] == 1:
        array = array.reshape(-1)
    if array.ndim != 1 or (array.size == 0 and not allow_empty):
        raise STTError("INVALID_AUDIO", "audio", "音频应为非空单声道数组")
    if not np.isfinite(array).all():
        raise STTError("INVALID_AUDIO", "audio", "音频不能含 NaN 或无穷值")
    return np.ascontiguousarray(array)


def validate_sample_rate(sample_rate):
    """只接受整数 16000；省略表示固定采样率，不隐式转换字符串或小数。"""
    if sample_rate is not None and (
        isinstance(sample_rate, bool)
        or not isinstance(sample_rate, (int, np.integer))
        or sample_rate != SAMPLE_RATE
    ):
        raise STTError("INVALID_SAMPLE_RATE", "accept_audio", "采样率应为整数 16000")
