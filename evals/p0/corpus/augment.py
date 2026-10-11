"""显式离线增广，返回转换参数和新的标注草稿。"""

import wave
from pathlib import Path

import numpy as np


def write_pcm16(path, values):
    """保存标准 WAV；量化前拒绝削波，不按峰值暗中改变录音。"""
    values = np.asarray(values)
    if not np.isfinite(values).all() or np.max(np.abs(values), initial=0) > 1:
        raise ValueError("音频非有限或超出 PCM16 范围")
    pcm = np.rint(values * 32768).clip(-32768, 32767).astype("<i2")
    with wave.open(str(Path(path)), "wb") as output:
        output.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        output.writeframes(pcm.tobytes())


def draft_intervals(audio):
    """能量仅生成待人工核实的边界草稿，不作语音检测验收依据。"""
    active = np.flatnonzero(np.abs(audio) > 0.01)
    return [[int(active[0]), int(active[-1] + 1)]] if active.size else []


def quiet_opening(audio, intervals):
    """首次标注语音起点后的 8000 样本幅度乘 0.1。"""
    start = intervals[0][0]
    end = min(start + 8000, audio.size)
    values = audio.copy()
    values[start:end] *= 0.1
    return values, intervals, {"gain_factor": 0.1, "gain_range": [start, end]}


def pause_between(audio, intervals):
    """两段母录音正文之间插入 4800 个零样本。"""
    start, end = intervals[0][0], intervals[-1][1]
    speech = audio[start:end]
    values = np.concatenate((speech, np.zeros(4800, np.float32), speech))
    ranges = [[0, speech.size], [speech.size + 4800, values.size]]
    return values, ranges, {"pause_samples": 4800, "parent_range": [start, end]}


def speed_up(audio, intervals):
    """按 1.25 倍速度压缩，区间按新采样位置重新形成草稿。"""
    from scipy.signal import resample_poly

    values = resample_poly(audio, 4, 5).astype(np.float32)
    ranges = [[round(start / 1.25), round(end / 1.25)] for start, end in intervals]
    return values, ranges, {"speed_ratio": 1.25, "resample_up": 4, "resample_down": 5}


def add_noise(audio, intervals, noise, *, noise_sample_id, seed):
    """有效语音区间的 RMS 决定 10 dB 混合；全段等比例避免削波。"""
    if noise.size < audio.size:
        raise ValueError("噪声长度不足，不隐含循环")
    noise = noise[: audio.size]
    mask = np.zeros(audio.size, dtype=bool)
    for start, end in intervals:
        mask[start:end] = True
    speech_rms = float(np.sqrt(np.mean(np.square(audio[mask].astype(np.float64)))))
    noise_rms = float(np.sqrt(np.mean(np.square(noise[mask].astype(np.float64)))))
    if noise_rms == 0:
        raise ValueError("噪声 RMS 为零")
    noise_scale = speech_rms / (noise_rms * 10 ** (10 / 20))
    mixed = audio + noise * noise_scale
    mix_scale = min(1.0, (32767 / 32768) / float(np.max(np.abs(mixed))))
    mixed = (mixed * mix_scale).astype(np.float32)
    return (
        mixed,
        intervals,
        {
            "snr_db": 10.0,
            "noise_sample_id": noise_sample_id,
            "noise_range": [0, audio.size],
            "noise_scale": noise_scale,
            "mix_scale": mix_scale,
            "seed": seed,
        },
    )


def tail_variant(audio, intervals, remainder):
    """调整首尾静音，尾部包含语音的条件仍需核实草稿区间。"""
    end = intervals[-1][1]
    values = audio[:end]
    padding = (remainder - values.size) % 512
    values = np.concatenate((np.zeros(padding, np.float32), values))
    ranges = [[start + padding, stop + padding] for start, stop in intervals]
    return (
        values,
        ranges,
        {"leading_silence_samples": padding, "original_end_sample": end, "remainder": remainder},
    )
