"""本地文件解码边界，解码结果沿用连续音频转换规则。"""

import wave
from contextlib import closing
from pathlib import Path

import numpy as np

from speechio.audio.decoding import AudioFileInfo, open_audio_source
from speechio.errors import ASRError
from speechio.stt.input import SAMPLE_RATE, AudioConverter, validate_max_audio_seconds


def read_audio_info(path) -> AudioFileInfo:
    """读取第一条音轨的信息并关闭文件；不加载模型或完整解码音频。"""
    with open_audio_source(path) as source:
        return source.info


def read_audio_file(path, *, max_audio_seconds=None) -> bytes:
    """读取完整标准 PCM；解码失败不返回已读取的前缀。"""
    converter = None
    try:
        with closing(
            audio_file_chunks(path, chunk_frames=8192, max_audio_seconds=max_audio_seconds)
        ) as chunks:
            output = bytearray()
            for format, chunk in chunks:
                if converter is None:
                    converter = AudioConverter(format)
                for frame in converter.feed(chunk):
                    output.extend(frame.pcm)
            for frame in converter.finish():
                output.extend(frame.pcm)
            return bytes(output)
    finally:
        if converter is not None:
            converter.close()


def read_wav(path):
    """读取并校验标准 PCM16 WAV。"""
    try:
        with wave.open(str(Path(path)), "rb") as wav:
            if wav.getframerate() != SAMPLE_RATE:
                raise ASRError("INVALID_SAMPLE_RATE", "read_wav", "P0 只接受 16000 Hz")
            if wav.getnchannels() != 1 or wav.getsampwidth() != 2 or wav.getcomptype() != "NONE":
                raise ASRError("INVALID_AUDIO", "read_wav", "P0 只接受单声道 PCM16 WAV")
            count = wav.getnframes()
            raw = wav.readframes(count)
            if count == 0 or len(raw) != count * 2:
                raise ASRError("INVALID_AUDIO", "read_wav", "WAV 为空或帧已截断")
    except (OSError, EOFError, wave.Error) as exc:
        raise ASRError("INVALID_AUDIO", "read_wav", str(exc)) from exc
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0


def audio_file_chunks(path, *, chunk_frames=1024, max_audio_seconds=None):
    """按源格式交付有界音频块；检查实际时长，关闭生成器即释放文件。"""
    validate_max_audio_seconds(max_audio_seconds)
    if isinstance(chunk_frames, bool) or not isinstance(chunk_frames, int) or chunk_frames <= 0:
        raise ASRError("INVALID_CONFIGURATION", "read_file", "chunk_frames 应为正整数")
    with open_audio_source(path) as source:
        info = source.info
        if (
            max_audio_seconds is not None
            and info.duration_seconds is not None
            and info.duration_seconds > max_audio_seconds
        ):
            raise ASRError("AUDIO_CAPACITY_EXCEEDED", "read_file", "音频超过配置的时长上限")
        samples = 0
        for chunk in source.chunks(chunk_frames):
            samples += len(chunk)
            if (
                max_audio_seconds is not None
                and samples / info.format.sample_rate > max_audio_seconds
            ):
                raise ASRError("AUDIO_CAPACITY_EXCEEDED", "read_file", "音频超过配置的时长上限")
            yield info.format, chunk
        if not samples:
            # 空文件也交付格式，让上层保持原有空音频处理行为。
            yield info.format, np.empty((0, info.format.channels), dtype=info.format.sample_type)
