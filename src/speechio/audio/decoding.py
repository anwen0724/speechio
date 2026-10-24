"""文件解码实现：SoundFile 保留原读取行为，PyAV 补充压缩音频格式。"""

import os
import struct
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np

from speechio.audio.formats import AudioFormat
from speechio.errors import STTError


@dataclass(frozen=True)
class AudioFileInfo:
    """源音轨的格式与声明时长；时长未知时为 None，不代表零秒。"""

    format: AudioFormat
    duration_seconds: float | None


class SoundFileSource:
    """保持 PCM16 的整数幅度，其他编码读取为浮点样本。"""

    def __init__(self, source):
        self.source = source
        self.info = AudioFileInfo(
            AudioFormat(
                source.samplerate,
                source.channels,
                "int16" if source.subtype == "PCM_16" else "float32",
            ),
            source.frames / source.samplerate,
        )

    def chunks(self, size):
        """有界读取源样本，不读取整个文件到内存。"""
        while True:
            chunk = self.source.read(size, dtype=self.info.format.sample_type, always_2d=True)
            if not chunk.size:
                return
            yield chunk


class PyAVSource:
    """只解码第一条音轨；输出交错浮点样本，重采样仍由 AudioConverter 负责。"""

    def __init__(self, container):
        self.container = container
        if not container.streams.audio:
            raise STTError("AUDIO_TRACK_MISSING", "read_file", "文件中没有音轨")
        self.stream = container.streams.audio[0]
        context = self.stream.codec_context
        duration = self.stream.duration
        self.info = AudioFileInfo(
            AudioFormat(context.sample_rate, context.channels, "float32"),
            float(duration * self.stream.time_base) if duration is not None else None,
        )

    def chunks(self, size):
        """合并解码帧到请求块长；正常结束时交付不足一块的尾部。"""
        import av

        format = self.info.format
        converter = av.AudioResampler(
            format="flt", layout=self.stream.codec_context.layout, rate=format.sample_rate
        )
        pending = np.empty((0, format.channels), np.float32)

        def frames():
            for frame in self.container.decode(self.stream):
                yield from converter.resample(frame)
            yield from converter.resample(None)

        for frame in frames():
            samples = frame.to_ndarray().reshape(-1, format.channels)
            pending = np.concatenate((pending, samples))
            offset = 0
            while len(pending) - offset >= size:
                yield pending[offset : offset + size]
                offset += size
            pending = pending[offset:]
        if len(pending):
            yield pending


@contextmanager
def open_audio_source(path):
    """只在打开时选择解码器；读取中途失败不回退，不访问远程地址。"""
    if not isinstance(path, (str, os.PathLike)):
        raise STTError("INVALID_AUDIO", "read_file", "输入应为本地文件路径")
    try:
        import soundfile as sf

        _check_pcm_wav_length(path)
        try:
            source = sf.SoundFile(os.fspath(path))
        except sf.LibsndfileError as error:
            # SF_ERR_UNRECOGNISED_FORMAT / SF_ERR_UNSUPPORTED_ENCODING。
            # 文件不存在、权限错误与已识别文件的损坏不触发备用解码。
            if error.code not in (1, 4):
                raise
        else:
            with source:
                yield SoundFileSource(source)
            return

        import av

        # 文件对象确保输入只来自本地文件；容器也禁止引用外部网络资源。
        with open(path, "rb") as file:
            with av.open(file, options={"protocol_whitelist": "file,pipe"}) as container:
                yield PyAVSource(container)
    except STTError:
        raise
    except ImportError as error:
        raise STTError(
            "DEPENDENCY_MISSING", "read_file", "缺少音频解码依赖，请重新安装 speechio"
        ) from error
    except Exception as error:
        raise STTError(
            "AUDIO_READ_FAILED", "read_file", f"无法读取或解码音频文件：{error}"
        ) from error


def _check_pcm_wav_length(path):
    """检查 RIFF PCM 数据块，防止解码器将截断文件当作正常短音频。"""
    with open(path, "rb") as file:
        header = file.read(12)
        if len(header) != 12 or header[:4] != b"RIFF" or header[8:] != b"WAVE":
            return
        file_size = os.fstat(file.fileno()).st_size
        riff_end = 8 + struct.unpack_from("<I", header, 4)[0]
        alignment = None
        data_blocks = []
        offset = 12
        while offset + 8 <= min(file_size, riff_end):
            file.seek(offset)
            kind, size = struct.unpack("<4sI", file.read(8))
            start = offset + 8
            if kind == b"fmt ":
                fmt = file.read(min(size, 40))
                if len(fmt) >= 16:
                    tag = struct.unpack_from("<H", fmt)[0]
                    pcm_guid = b"\x01\x00\x00\x00\x00\x00\x10\x00\x80\x00\x00\xaa\x00\x38\x9b\x71"
                    if tag == 1 or (tag == 65534 and len(fmt) >= 40 and fmt[24:40] == pcm_guid):
                        alignment = struct.unpack_from("<H", fmt, 12)[0]
            elif kind == b"data":
                data_blocks.append((start, size))
            # 数据只跳过不读取；未知信息块及奇数字节填充均保留。
            offset = start + size + size % 2
        if alignment is not None and any(
            start + size > min(file_size, riff_end) or not alignment or size % alignment
            for start, size in data_blocks
        ):
            raise STTError("AUDIO_READ_FAILED", "read_file", "PCM WAV 音频数据已截断或采样帧不完整")
