"""自动输入拥有连续转换和待递交录音段，复用现有会话任务与事件。"""

from collections import deque
from dataclasses import dataclass, field
from threading import RLock
from time import monotonic

import numpy as np

from realtime_asr.audio.conversion import AudioConverter, AudioFormat
from realtime_asr.errors import ASRError
from realtime_asr.segmentation.gate import SegmentationGate
from realtime_asr.segmentation.vad import VoiceActivityDetector
from realtime_asr.session.session import Session, _validate_timeout


@dataclass
class _AutomaticInput:
    """一个输入代次；控制操作替换它，使正在检测的旧输入自然失效。"""

    converter: AudioConverter
    gate: SegmentationGate
    detector: VoiceActivityDetector
    segments: deque = field(default_factory=deque)
    state: str = "listening"
    cancelled_units: set = field(default_factory=set)
    cancel_current: bool = False


class AutomaticSession(Session):
    """宿主只连续送音频；封口和后台任务使用普通会话的同一实现。"""

    def __init__(self, runtime, config, vad, segmentation):
        super().__init__(runtime, config)
        self._input_lock = RLock()
        self._vad_config = vad
        self._segmentation = segmentation
        minimum = int(segmentation.min_recording_duration * 16000)
        pre_roll = int(segmentation.pre_roll_duration * 16000)
        if (
            config.max_audio_samples < max(minimum, pre_roll, 1)
            or config.max_audio_samples > runtime._limits.max_task_audio_samples
            or segmentation.max_buffer_samples < config.max_audio_samples + pre_roll + 1024
        ):
            raise ASRError("INVALID_CONFIGURATION", "configure", "自动段长与缓冲预算不一致")
        self._input = self._new_input()
        self._vad_info = self._input.detector.info

    def _new_input(self):
        return _AutomaticInput(
            AudioConverter(self.config.format),
            SegmentationGate(self._segmentation, max_segment_samples=self.config.max_audio_samples),
            VoiceActivityDetector(self._vad_config),
        )

    @property
    def input_state(self):
        with self._runtime._condition:
            if self._closed:
                return "closed"
            return self._input.state

    @property
    def vad_info(self):
        return dict(self._vad_info)

    @property
    def pending_segments(self):
        with self._input_lock, self._runtime._condition:
            if self._input is None:
                return 0
            unassigned = sum(segment.unit is None for segment in self._input.segments)
            return len(self._units) + unassigned

    def start(self):
        raise ASRError("AUTOMATIC_INPUT", "start", "自动会话直接连续输入音频")

    def _require_input(self):
        self._require_open()
        state = self._input
        if state.state in ("finished", "failed"):
            raise ASRError("INPUT_ENDED", "feed_audio", "输入已经结束或失败，请重置后重新输入")
        return state

    def _buffer_samples(self, state):
        return state.gate.history_samples + sum(len(segment.pcm) // 2 for segment in state.segments)

    def _check_chunk_capacity(self, state, chunk):
        """接纳前估计整个源块上界，拒绝后转换器状态不变。"""
        scalar_bytes = 2 if self.config.format.sample_type == "int16" else 4
        if isinstance(chunk, (bytes, bytearray, memoryview)):
            scalars = len(chunk) / scalar_bytes
        elif isinstance(chunk, np.ndarray):
            scalars = chunk.size
        else:
            raise ASRError("INVALID_AUDIO", "feed_audio", "输入应为字节或 NumPy 数组")
        upper = int(scalars / self.config.format.channels * 16000 / self.config.format.sample_rate)
        if self._buffer_samples(state) + upper + 1024 > self._segmentation.max_buffer_samples:
            raise ASRError("AUDIO_CAPACITY_EXCEEDED", "feed_audio", "自动输入缓冲不足，本块未接纳")

    def feed_audio(self, chunk):
        """本会话串行转换和检测；不持有全局锁，也不等待 ASR 模型。"""
        with self._input_lock:
            with self._runtime._condition:
                state = self._require_input()
                self._apply_cancellations(state)
                self._pump(state)
                self._check_chunk_capacity(state, chunk)
            try:
                frames = state.converter.feed(chunk)
                self._detect_frames(state, frames)
            except ASRError as exc:
                if exc.code not in ("INVALID_AUDIO", "INPUT_CANCELLED", "SESSION_CLOSED"):
                    self._fail_input(state, exc)
                raise
            with self._runtime._condition:
                self._check_generation(state)
                self._pump(state)
            return sum(frame.end_sample - frame.start_sample for frame in frames)

    def _check_generation(self, state):
        self._require_open()
        if self._input is not state:
            raise ASRError("INPUT_CANCELLED", "feed_audio", "检测所属输入已失效")

    def _detect_frames(self, state, frames):
        for frame in frames:
            activity = state.detector.detect(
                frame.pcm, frame.start_sample, recording=state.gate.active is not None
            )
            with self._runtime._condition:
                self._check_generation(state)
                if state.cancel_current:
                    # 转换已经接纳本块；取消掉本块尚未登记的段，保留转换时钟。
                    state.gate.position = frames[-1].end_sample
                    self._apply_cancellations(state)
                    raise ASRError("INPUT_CANCELLED", "feed_audio", "当前输入在检测期间被取消")
                state.segments.extend(state.gate.feed(frame, activity))
                self._pump(state)
        if state.state not in ("failed", "finished"):
            state.state = "recording" if state.gate.active is not None else "listening"

    def _pump(self, state):
        """按段推进，容量不足保留原段；固定任务不再占自动缓冲。"""
        self._apply_cancellations(state)
        while state.segments:
            segment = state.segments[0]
            try:
                if segment.unit is None:
                    segment.unit = self._begin_unit(AudioFormat())
                    self._active.stream_start_sample = segment.start_sample
                unit = self._units[segment.unit.unit_id]
                samples = len(segment.pcm) // 2
                if segment.boundary is None:
                    samples = max(0, samples - int(self._segmentation.backdate_duration * 16000))
                pcm = bytes(segment.pcm[segment.delivered_samples * 2 : samples * 2])
                self._append_standard(unit, pcm)
                segment.delivered_samples = samples
                if segment.boundary is None:
                    return
                unit.boundary = segment.boundary
                Session.finish(self)
                state.segments.popleft()
            except ASRError as exc:
                if exc.code in (
                    "TASK_CAPACITY_EXCEEDED",
                    "RESULT_CAPACITY_EXCEEDED",
                    "REALTIME_CAPACITY_EXCEEDED",
                ):
                    return
                raise

    def _apply_cancellations(self, state):
        """只在输入拥有者内改动分句状态，控制线程只登记失效目标。"""
        if state.cancel_current:
            state.gate.cancel_active()
            state.detector.reset()
            state.segments = deque(
                segment for segment in state.segments if segment.boundary is not None
            )
            state.cancel_current = False
            state.state = "listening"
        if state.cancelled_units:
            state.segments = deque(
                segment for segment in state.segments if segment.unit not in state.cancelled_units
            )
            state.cancelled_units.clear()

    def cancel(self, target=None):
        """立即失效任务；检测线程在下一边界清除自己的候选与缓存。"""
        with self._runtime._condition:
            self._require_open()
            state = self._input
            current = state.gate.active
            ref = target if target is not None else (current.unit if current else None)
            changed = Session.cancel(self, ref) if ref is not None else False
            if current is not None and (target is None or current.unit == target):
                state.cancel_current = True
                changed = True
            if changed and ref is not None:
                state.cancelled_units.add(ref)
            self._runtime._condition.notify_all()
            return changed

    def finish_input(self):
        """只刷新一次源尾部；终止监听但不取消已固定任务。"""
        with self._input_lock:
            with self._runtime._condition:
                self._require_open()
                state = self._input
                if state.state == "failed":
                    raise ASRError("INPUT_ENDED", "finish_input", "输入已失败，请重置")
                self._apply_cancellations(state)
            if state.state != "finished":
                try:
                    self._detect_frames(state, state.converter.finish())
                    state.gate.seal(reason="input_finished")
                    state.state = "finished"
                except ASRError as exc:
                    self._fail_input(state, exc)
                    raise
            with self._runtime._condition:
                self._check_generation(state)
                self._pump(state)

    def finish(self):
        """手动封口当前自动段，继续监听；容量失败可原样重试。"""
        with self._input_lock, self._runtime._condition:
            state = self._require_input()
            self._apply_cancellations(state)
            segment = state.gate.seal()
            if segment is None:
                if state.segments and state.segments[0].boundary is not None:
                    segment = state.segments[0]
                else:
                    raise ASRError("NO_ACTIVE_UNIT", "finish", "没有活动录音段")
            self._pump(state)
            if segment in state.segments:
                raise ASRError("TASK_CAPACITY_EXCEEDED", "finish", "封口段已保留，等待容量后重试")
            state.state = "listening"
            return self._last_task

    def _wait_token(self):
        """只比较会影响递交或消费的公开流程状态，防止等待前漏通知。"""
        realtime = self._runtime._realtime
        return (
            self._runtime._tasks,
            len(realtime.states) if realtime else 0,
            self._next_event() is not None,
            id(self._input),
            self._closed,
        )

    def get_event(self, *, timeout=0.0):
        """消费和容量唤醒都推进待递交段，等待不阻塞持续输入。"""
        _validate_timeout(timeout, allow_none=True)
        deadline = None if timeout is None else monotonic() + timeout
        while True:
            with self._input_lock, self._runtime._condition:
                if self._closed:
                    return None
                state = self._input
                self._pump(state)
                event = Session.get_event(self)
                if event is not None:
                    self._pump(state)
                    return event
                token = self._wait_token()
            remaining = None if deadline is None else max(0, deadline - monotonic())
            if remaining == 0:
                return None
            with self._runtime._condition:
                if token == self._wait_token():
                    self._runtime._condition.wait(remaining)
                if self._input is not state:
                    return None

    def _fail_input(self, state, error):
        """检测错误只失败尚未固定的当前输入，不改写已提交段。"""
        with self._runtime._condition:
            if self._input is not state or self._closed:
                return
            if self._active is not None and self._active.task is None:
                self._terminal(self._active, "failed", error=error)
            state.gate.cancel_active()
            state.segments.clear()
            state.converter.close()
            state.state = "failed"

    def close(self):
        """控制不等待检测锁；正在执行的检测保留自己的本地引用。"""
        with self._runtime._condition:
            change = Session.close(self)
            self._input = None
            return change

    def reset(self):
        """锁外重新准备检测；旧调用不能向新代次登记结果。"""
        fresh = self._new_input()
        with self._runtime._condition:
            change = Session.reset(self)
            self._input = fresh
            self._vad_info = fresh.detector.info
            return change
