"""10 ms 能量包络中的发声谷检测；输出信号，不切音频或启动推理。"""

from collections import deque
from dataclasses import dataclass
from math import ceil, log10, sqrt

import numpy as np

from realtime_asr.core.errors import ASRError


@dataclass(frozen=True)
class AcousticBoundary:
    """边界位置相对本检测流，延迟来自已经处理的样本数。"""

    boundary_sample: int
    reason: str
    score: float
    energy_db: float
    noise_floor_db: float
    drop_db: float
    valley_depth_db: float
    latency_ms: float


@dataclass(frozen=True)
class _EnergyFrame:
    center: int
    rms: float
    energy: float
    noise: float
    speech: bool
    vowel: bool
    voicing: float


class AcousticBoundaryDetector:
    """保留跨调用余量与 900 ms 历史，以 30 ms 前瞻确认发声结束。"""

    def __init__(self, sensitivity=0.6):
        if not isinstance(sensitivity, (int, float)) or not 0 <= sensitivity <= 1:
            raise ASRError("INVALID_CONFIGURATION", "configure", "声学灵敏度应在 0 至 1 内")
        self.speech_margin = 10.5 - 4.5 * sensitivity
        self.vowel_margin = 14 - 5 * sensitivity
        self.drop = 5.8 - 3.2 * sensitivity
        self.valley_depth = 3.2 - 1.8 * sensitivity
        self.recovery = 2.2 - 1.1 * sensitivity
        self.min_voicing = 0.62 - 0.12 * sensitivity
        self.max_crossings = 0.20 + 0.05 * sensitivity
        self._pending = b""
        self._history = deque(maxlen=97)
        self._processed = 0
        self._noise = -70.0
        self._last_boundary = -2560

    def feed(self, pcm):
        """接纳相邻 PCM16；短帧保留到下一调用，不补零。"""
        if not isinstance(pcm, bytes) or len(pcm) % 2:
            raise ASRError("INVALID_AUDIO", "acoustic", "声学输入应为完整 PCM16 字节")
        data = self._pending + pcm
        complete = len(data) // 320 * 320
        self._pending = data[complete:]
        events = []
        for offset in range(0, complete, 320):
            samples = (
                np.frombuffer(data[offset : offset + 320], dtype="<i2").astype(np.float32) / 32768
            )
            self._history.append(self._analyze(samples))
            event = self._boundary()
            if event is not None:
                events.append(event)
        return events

    def _analyze(self, samples):
        """更新缓慢变化的噪声底，再用周期相关性区分持续发声与噪声。"""
        rms = float(np.sqrt(np.mean(samples * samples)) + 1e-12)
        energy = 20 * log10(rms + 1e-12)
        if self._processed < 1600:
            self._noise = min(self._noise, energy)
        elif energy < self._noise + 3:
            self._noise = 0.92 * self._noise + 0.08 * energy
        else:
            self._noise = 0.997 * self._noise + 0.003 * min(energy, self._noise + 18)
        if self._processed >= 1600:
            self._noise = max(-100, min(-20, self._noise))
        centered = samples - float(np.mean(samples))
        voicing = 0.0
        if rms >= 0.004 and float(np.dot(centered, centered)) > 1e-9:
            for lag in range(45, 158):
                left, right = centered[:-lag], centered[lag:]
                denominator = sqrt(float(np.dot(left, left)) * float(np.dot(right, right)))
                if denominator > 1e-9:
                    voicing = max(voicing, float(np.dot(left, right)) / denominator)
        crossing = (
            float(np.count_nonzero(np.signbit(samples[1:]) != np.signbit(samples[:-1]))) / 159
        )
        speech = rms >= 0.004 and energy >= self._noise + self.speech_margin
        vowel = (
            speech
            and energy >= self._noise + self.vowel_margin
            and (voicing >= self.min_voicing and crossing <= self.max_crossings)
        )
        frame = _EnergyFrame(self._processed + 80, rms, energy, self._noise, speech, vowel, voicing)
        self._processed += 160
        return frame

    def _boundary(self):
        """低谷之前需要持续发声，之后需要恢复或停顿证据。"""
        frames = list(self._history)
        position = len(frames) - 4
        if position < 7:
            return None
        candidate = frames[position]
        if candidate.center - self._last_boundary < 2560:
            return None
        before = frames[max(0, position - 18) : position]
        after = frames[position + 1 : position + 4]
        if sum(frame.speech for frame in before[-7:]) < ceil(7 * 0.65):
            return None
        vowels = [frame for frame in before[-8:] if frame.vowel]
        if len(vowels) < 4:
            return None
        peak_before = max(frame.energy for frame in before)
        peak_after = max(frame.energy for frame in after)
        avg_before = sum(frame.energy for frame in before[-5:]) / len(before[-5:])
        avg_after = sum(frame.energy for frame in after) / 3
        drop = peak_before - candidate.energy
        valley = min(avg_before, peak_after) - candidate.energy
        vowel_drop = max(frame.energy for frame in vowels) - candidate.energy
        local_minimum = all(
            candidate.energy <= frame.energy + 0.8 for frame in before[-2:] + after[:2]
        )
        vowel_ended = (
            not candidate.vowel
            and not any(frame.vowel for frame in after)
            and (vowel_drop >= max(1.5, self.drop * 0.55))
        )
        vowel_valley = local_minimum and vowel_drop >= self.drop and valley >= self.valley_depth
        if not (vowel_ended or vowel_valley):
            return None

        def component(value, threshold):
            return max(0, min(1.25, value / threshold))

        score = component(vowel_drop, self.drop) + component(valley, self.valley_depth)
        score += 0.8 * component(max(frame.voicing for frame in vowels), self.min_voicing)
        after_speech = any(frame.speech for frame in after)
        if after_speech and not any(frame.vowel for frame in after):
            score += 0.7 * component(peak_after - candidate.energy, self.recovery)
        elif avg_after < candidate.noise + self.speech_margin + 2:
            score += 0.45
        if candidate.rms < 0.005:
            score += 0.2
        if score < 2:
            return None
        self._last_boundary = candidate.center
        return AcousticBoundary(
            candidate.center,
            "vowel-ended" if after_speech else "vowel-to-pause",
            score,
            candidate.energy,
            candidate.noise,
            drop,
            valley,
            (self._processed - candidate.center) / 16,
        )
