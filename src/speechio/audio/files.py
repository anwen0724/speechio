"""本地文件解码边界，解码结果沿用连续音频转换规则。"""

import os
import wave
from pathlib import Path

import numpy as np

from speechio.audio.conversion import SAMPLE_RATE, AudioConverter, AudioFormat
from speechio.errors import ASRError


def read_audio_file(path) -> bytes:
    """读取文件头格式，返回完整标准 PCM；异常不返回成功前缀。"""
    if not isinstance(path, (str, os.PathLike)):
        raise ASRError("INVALID_AUDIO", "read_file", "输入应为本地文件路径")
    try:
        import soundfile as sf
    except (ImportError, OSError) as exc:
        raise ASRError("DEPENDENCY_MISSING", "read_file", "请安装 audio 依赖") from exc
    converter = None
    try:
        with sf.SoundFile(os.fspath(path)) as source:
            sample_type = "int16" if source.subtype == "PCM_16" else "float32"
            converter = AudioConverter(AudioFormat(source.samplerate, source.channels, sample_type))
            output = bytearray()
            while True:
                chunk = source.read(8192, dtype=sample_type, always_2d=True)
                if not chunk.size:
                    break
                for frame in converter.feed(chunk):
                    output.extend(frame.pcm)
            for frame in converter.finish():
                output.extend(frame.pcm)
            return bytes(output)
    except ASRError:
        raise
    except Exception as exc:
        raise ASRError("AUDIO_READ_FAILED", "read_file", str(exc)) from exc
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


def audio_file_chunks(path, *, chunk_frames=1024):
    """按文件头交付格式与有限源块；结束或关闭生成器即释放文件。"""
    if not isinstance(path, (str, os.PathLike)):
        raise ASRError("INVALID_AUDIO", "read_file", "输入应为本地文件路径")
    try:
        import soundfile as sf
    except (ImportError, OSError) as exc:
        raise ASRError("DEPENDENCY_MISSING", "read_file", "请安装 audio 依赖") from exc
    try:
        with sf.SoundFile(os.fspath(path)) as source:
            sample_type = "int16" if source.subtype == "PCM_16" else "float32"
            format = AudioFormat(source.samplerate, source.channels, sample_type)
            # 空文件仍交付格式，让会话决定没有有效样本的失败。
            chunk = source.read(chunk_frames, dtype=sample_type, always_2d=True)
            yield format, chunk
            while chunk.size:
                chunk = source.read(chunk_frames, dtype=sample_type, always_2d=True)
                if chunk.size:
                    yield format, chunk
    except ASRError:
        raise
    except Exception as exc:
        raise ASRError("AUDIO_READ_FAILED", "read_file", str(exc)) from exc
