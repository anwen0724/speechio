"""P0 标准音频边界；不在识别入口隐式重采样或归一化。"""

import wave
from pathlib import Path

import numpy as np

from realtime_asr.core.errors import ASRError

SAMPLE_RATE = 16000


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


def as_audio(audio, *, allow_empty=False, allow_column=False):
    """将合法单声道输入转换为连续 float32。"""
    if audio is None:
        raise ASRError("INVALID_AUDIO", "audio", "音频不能为 None")
    try:
        array = np.asarray(audio, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ASRError("INVALID_AUDIO", "audio", "音频应为数值数组") from exc
    if allow_column and array.ndim == 2 and array.shape[1] == 1:
        array = array.reshape(-1)
    if array.ndim != 1 or (array.size == 0 and not allow_empty):
        raise ASRError("INVALID_AUDIO", "audio", "音频应为非空单声道数组")
    if not np.isfinite(array).all():
        raise ASRError("INVALID_AUDIO", "audio", "音频不能含 NaN 或无穷值")
    return np.ascontiguousarray(array)


def validate_sample_rate(sample_rate):
    """只接受整数 16000；省略表示固定采样率，不隐式转换字符串或小数。"""
    if sample_rate is not None and (
        isinstance(sample_rate, bool)
        or not isinstance(sample_rate, (int, np.integer))
        or sample_rate != SAMPLE_RATE
    ):
        raise ASRError("INVALID_SAMPLE_RATE", "accept_audio", "采样率应为整数 16000")
