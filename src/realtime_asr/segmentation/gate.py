"""按标准样本划分录音段；不调用识别模型，不拥有任务或结果。"""

from collections import deque
from dataclasses import dataclass, field
from math import ceil, isfinite

from realtime_asr.audio.conversion import AudioFrame
from realtime_asr.errors import ASRError


@dataclass(frozen=True, kw_only=True)
class SegmentationConfig:
    """分句时间均为秒；检测状态机只使用换算后的标准样本数。"""

    silence_confirmation: float = 0.16
    post_speech_silence: float = 0.6
    min_recording_duration: float = 0.5
    min_gap: float = 0
    pre_roll_duration: float = 1
    trim_pre_roll: bool = False
    backdate_duration: float = 0
    max_buffer_samples: int = 1920000

    def __post_init__(self):
        seconds = (
            self.silence_confirmation,
            self.post_speech_silence,
            self.min_recording_duration,
            self.min_gap,
            self.pre_roll_duration,
            self.backdate_duration,
        )
        if any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not isfinite(value)
            or value < 0
            for value in seconds
        ) or (
            type(self.trim_pre_roll) is not bool
            or self.min_gap > self.pre_roll_duration
            or self.backdate_duration > self.post_speech_silence
            or type(self.max_buffer_samples) is not int
            or self.max_buffer_samples <= 0
        ):
            raise ASRError("INVALID_CONFIGURATION", "configure", "自动分句配置无效")


@dataclass(frozen=True)
class SegmentBoundary:
    """固定录音段在连续标准音频中的半开范围与封口依据。"""

    start_sample: int
    end_sample: int
    trigger_sample: int
    reason: str
    pre_roll_reason: str


@dataclass
class RecordingSegment:
    """分句与会话之间的待递交音频；递交位置由会话维护。"""

    start_sample: int
    pre_roll_reason: str
    pcm: bytearray = field(default_factory=bytearray)
    boundary: SegmentBoundary | None = None
    unit: object = None
    delivered_samples: int = 0


class SegmentationGate:
    """保留前置历史并确定切点，调用方接管生成段的任务归属。"""

    def __init__(self, config: SegmentationConfig, *, max_segment_samples: int):
        self.config = config
        self.max_segment_samples = max_segment_samples
        self.active = None
        self.position = 0
        self._history = deque()
        self._recent = deque()
        self._last_end = 0
        self._next_start_sample = 0
        self._silence_start = None
        self._waiting_onset = False
        self._continuing = False
        self._gap_protection = False

    @property
    def history_samples(self):
        return sum(frame.end_sample - frame.start_sample for frame, _ in self._history)

    def feed(self, frame: AudioFrame, activity) -> list[RecordingSegment]:
        """接纳一次正向检测，只返回本次新创建的段。"""
        self.position = frame.end_sample
        started = []
        if self.active is None:
            self._waiting_onset |= activity.onset or self._continuing
            if self._waiting_onset and frame.end_sample < self._next_start_sample:
                self._gap_protection = True
            if self._waiting_onset and (
                self._continuing or frame.end_sample >= self._next_start_sample
            ):
                history, trim_reason = self._select_history()
                start = history[0][0].start_sample if history else frame.start_sample
                segment = RecordingSegment(start, trim_reason)
                for old_frame, _ in history:
                    segment.pcm.extend(old_frame.pcm)
                self.active = segment
                self._history.clear()
                self._waiting_onset = self._continuing = False
                self._gap_protection = False
                started.append(segment)
            else:
                self._remember(frame, activity)
                return started
        remaining = self.max_segment_samples - len(self.active.pcm) // 2
        frame_samples = frame.end_sample - frame.start_sample
        if frame_samples >= remaining:
            self.active.pcm.extend(frame.pcm[: remaining * 2])
            self.position = frame.start_sample + remaining
            self.seal(reason="capacity")
            self._continuing = True
            if frame_samples > remaining:
                remainder = AudioFrame(frame.pcm[remaining * 2 :], self.position, frame.end_sample)
                started.extend(self.feed(remainder, activity))
            return started
        self.active.pcm.extend(frame.pcm)
        self._recent.append((frame, activity))
        keep_after = frame.end_sample - int(self.config.backdate_duration * 16000)
        while self._recent and self._recent[0][0].end_sample <= keep_after:
            self._recent.popleft()
        if activity.speech:
            self._silence_start = None
        elif self._silence_start is None:
            self._silence_start = frame.start_sample
        silence_samples = (
            0 if self._silence_start is None else frame.end_sample - self._silence_start
        )
        required_silence = int(self.config.silence_confirmation * 16000) + int(
            self.config.post_speech_silence * 16000
        )
        if (
            not activity.speech
            and silence_samples >= required_silence
            and len(self.active.pcm) // 2 >= int(self.config.min_recording_duration * 16000)
        ):
            self.seal(reason="silence")
        return started

    def _select_history(self):
        """只裁已有证据支持的稳定静音，轻声首部向前纳入保护。"""
        history = list(self._history)
        if not self.config.trim_pre_roll:
            return history, "disabled"
        if self._gap_protection:
            return history, "gap_protection"
        if not history or self.history_samples <= 9600:
            return history, "insufficient_audio"
        onset = None
        run_start = None
        gap = 0
        for index, (frame, activity) in enumerate(history):
            if activity.webrtc or activity.silero:
                if run_start is None or gap > 1280:
                    run_start = index
                onset = run_start
                gap = 0
            elif run_start is not None:
                gap += frame.end_sample - frame.start_sample
        if onset is None:
            # 当前开口帧还未入历史，它本身就是明确的开口位置。
            onset = len(history)
        energy = sorted(activity.rms for _, activity in history[:onset])
        if not energy:
            return history, "no_silence_prefix"
        floor_count = max(1, ceil(len(energy) * 0.2))
        threshold = sum(energy[:floor_count]) / floor_count * 2.5 + 25

        def stable(index):
            activity = history[index][1]
            return not (activity.webrtc or activity.silero) and activity.rms <= threshold

        effective_onset = onset
        index = onset - 1
        while index >= 0 and not stable(index):
            effective_onset = index
            index -= 1
        silence = 0
        while index >= 0 and stable(index):
            frame = history[index][0]
            silence += frame.end_sample - frame.start_sample
            index -= 1
        if silence < 3200:
            return history, "insufficient_stable_silence"
        effective_start = (
            history[effective_onset][0].start_sample
            if effective_onset < len(history)
            else history[-1][0].end_sample
        )
        candidate = min(effective_start - 2560, history[-1][0].end_sample - 9600)
        chosen = 0
        for index, (frame, _) in enumerate(history):
            if frame.start_sample <= candidate:
                chosen = index
        if chosen == 0:
            return history, "no_removable_prefix"
        return history[chosen:], "trimmed"

    def _remember(self, frame, activity):
        self._history.append((frame, activity))
        earliest = max(
            self._last_end, frame.end_sample - int(self.config.pre_roll_duration * 16000)
        )
        while self._history and self._history[0][0].end_sample <= earliest:
            self._history.popleft()
        if self._history and self._history[0][0].start_sample < earliest:
            old, metadata = self._history[0]
            self._history[0] = (
                AudioFrame(old.pcm[(earliest - old.start_sample) * 2 :], earliest, old.end_sample),
                metadata,
            )

    def seal(self, *, reason="manual"):
        """显式结束不要求最短段，也不执行静音回溯。"""
        if self.active is None:
            return None
        segment = self.active
        cut = self.position
        if reason == "silence":
            cut = max(
                self.position - int(self.config.backdate_duration * 16000),
                segment.start_sample + int(self.config.min_recording_duration * 16000),
            )
            self._last_end = cut
            for frame, metadata in self._recent:
                if frame.end_sample > cut:
                    start = max(cut, frame.start_sample)
                    self._remember(
                        AudioFrame(
                            frame.pcm[(start - frame.start_sample) * 2 :], start, frame.end_sample
                        ),
                        metadata,
                    )
            del segment.pcm[(cut - segment.start_sample) * 2 :]
        segment.boundary = SegmentBoundary(
            segment.start_sample, cut, self.position, reason, segment.pre_roll_reason
        )
        self.active = None
        self._last_end = cut
        self._next_start_sample = cut + int(self.config.min_gap * 16000)
        self._silence_start = None
        self._recent.clear()
        return segment

    def cancel_active(self):
        """取消当前录音及监听历史，不复用被取消的采样。"""
        self.active = None
        self._history.clear()
        self._recent.clear()
        self._silence_start = None
        self._waiting_onset = self._continuing = False
        self._gap_protection = False
        self._last_end = self.position
        self._next_start_sample = self.position
