"""自动输入拥有连续转换和待递交录音段，复用现有会话任务与事件。"""

from collections import deque
from dataclasses import dataclass, field, replace
from hashlib import sha256
from threading import RLock
from time import monotonic
from uuid import uuid4

import numpy as np

from speechio.audio.conversion import AudioConverter, AudioFormat, AudioFrame
from speechio.errors import ASRError
from speechio.results import Transcript
from speechio.runtime.tasks import InferenceRequest
from speechio.segmentation.gate import SegmentationGate
from speechio.segmentation.vad import (
    VoiceActivity,
    VoiceDetector,
    create_detector,
    local_detector_factory,
)
from speechio.segmentation.wakeword import WakeConfig, WakeGate
from speechio.session.session import FinalTask, Session, _AudioUnit, _validate_timeout
from speechio.text.endpoint import EndpointConfig, SentenceBoundaryPolicy, SentenceHint


@dataclass
class _AutomaticInput:
    """一个输入代次；控制操作替换它，使正在检测的旧输入自然失效。"""

    converter: AudioConverter
    gate: SegmentationGate
    detector: VoiceDetector
    wake: WakeGate | None = None
    segments: deque = field(default_factory=deque)
    state: str = "listening"
    cancelled_units: set = field(default_factory=set)
    cancel_current: bool = False
    early_requested: bool = False


@dataclass(frozen=True)
class _SentenceCut:
    """有词时间证据的待提交切点；当前单元另行提供归属。"""

    hint: SentenceHint
    sample: int
    covered_samples: int


@dataclass(frozen=True)
class EarlyFinalResult:
    """固定提前输入的结果与摘要，仅由自动会话决定复用。"""

    audio_samples: int
    audio_digest: bytes
    result: Transcript
    timing: tuple[float, float, float]

    def matches_audio(self, pcm):
        return self.audio_samples == len(pcm) // 2 and self.audio_digest == sha256(pcm).digest()


@dataclass(frozen=True)
class _AuxiliaryRequest:
    """请求用途和锚点由自动会话保存，不进入通用执行队列。"""

    request: InferenceRequest
    purpose: str
    hint: SentenceHint | None
    epoch: int


@dataclass
class _AutomaticUnit(_AudioUnit):
    """自动语音策略的单元状态；普通会话与运行时不操作这些字段。"""

    optimization: dict = field(default_factory=dict)
    auxiliary: _AuxiliaryRequest | None = None
    endpoint_epoch: int = 0
    early_result: EarlyFinalResult | None = None
    pending_split: _SentenceCut | None = None
    sentence_policy: SentenceBoundaryPolicy | None = None
    wake_context: dict | None = None


class LiveSession(Session):
    """宿主只连续送音频；封口和后台任务使用普通会话的同一实现。"""

    def __init__(
        self,
        runtime,
        config,
        vad,
        segmentation,
        endpoint=None,
        *,
        detector_factory=None,
        wake=None,
        wake_detector_factory=None,
    ):
        super().__init__(runtime, config)
        self._endpoint = endpoint if endpoint is not None else EndpointConfig()
        if not isinstance(self._endpoint, EndpointConfig):
            raise ASRError("INVALID_CONFIGURATION", "configure", "需要 EndpointConfig")
        shortest = (
            min(
                self._endpoint.sentence_pause,
                self._endpoint.unknown_pause,
                self._endpoint.continuation_pause,
            )
            if self._endpoint.dynamic_silence
            else segmentation.post_speech_silence
        )
        if segmentation.backdate_duration > shortest or (
            self._endpoint.early_transcription_silence > 0
            and self._endpoint.early_transcription_silence >= shortest
        ):
            raise ASRError("INVALID_CONFIGURATION", "configure", "回溯或提前阈值超过结束等待")
        if self._endpoint.dynamic_silence and not config.realtime:
            raise ASRError("INVALID_CONFIGURATION", "configure", "动态等待需要实时观察")
        if self._endpoint.punctuation_split and (
            not config.realtime
            or config.stabilization is None
            or not runtime.capabilities.word_timestamps
        ):
            raise ASRError("BACKEND_INCOMPATIBLE", "configure", "句末切段需要稳定观察和词时间能力")
        if self._endpoint.punctuation_split and self._completion_source == "streaming":
            raise ASRError(
                "BACKEND_INCOMPATIBLE",
                "configure",
                "标点回溯切段会改变已发送的音频边界，需要 completion_mode='final' 重识别",
            )
        self._optimization_totals = {}
        self._closed_wake = None
        self._input_lock = RLock()
        self._detector_factory = detector_factory or local_detector_factory(vad)
        self._segmentation = segmentation
        if (wake is not None and not isinstance(wake, WakeConfig)) or (
            wake_detector_factory is not None
            and (wake is None or not callable(wake_detector_factory))
        ):
            raise ASRError("INVALID_CONFIGURATION", "configure", "唤醒配置或工厂无效")
        self._wake_config, self._wake_factory = wake, wake_detector_factory
        minimum = int(segmentation.min_recording_duration * 16000)
        pre_roll = int(segmentation.pre_roll_duration * 16000)
        if (
            config.max_audio_samples < max(minimum, pre_roll, 1)
            or config.max_audio_samples > runtime.limits.max_task_audio_samples
            or segmentation.max_buffer_samples < config.max_audio_samples + pre_roll + 1024
        ):
            raise ASRError("INVALID_CONFIGURATION", "configure", "自动段长与缓冲预算不一致")
        self._input = self._new_input()
        self._vad_info = self._input.detector.info

    def _new_input(self):
        state = _AutomaticInput(
            AudioConverter(self.config.format),
            SegmentationGate(self._segmentation, max_segment_samples=self.config.max_audio_samples),
            create_detector(self._detector_factory),
        )
        if self._wake_config is not None:
            state.wake = WakeGate(
                self._wake_config,
                int(self._segmentation.pre_roll_duration * 16000),
                self._wake_factory,
                on_release=self._runtime.release_wake_resource,
            )
            self._runtime.own_wake_resource(self.session_id, state.wake)
        return state

    @property
    def wake_status(self):
        """返回当前输入代次的门控观察，不改写已封口单元。"""
        with self._runtime.coordination:
            state = self._input
            if state is None:
                return {
                    **(self._closed_wake.status if self._closed_wake else {"enabled": False}),
                    "state": "closed",
                    "released": self._closed_wake.released if self._closed_wake else True,
                }
            if state.wake is None:
                return {"enabled": False, "state": "disabled", "released": True}
            return dict(state.wake.status)

    @property
    def input_state(self):
        with self._runtime.coordination:
            if self._closed:
                return "closed"
            if self._runtime_error is not None:
                return "failed"
            return self._input.state

    @property
    def vad_info(self):
        return dict(self._vad_info)

    @property
    def optimization_status(self):
        """读取本代次诊断副本；关闭后仍可检查最后的释放前结论。"""
        with self._runtime.coordination:
            return dict(self._optimization_totals)

    def _count_optimization(self, unit, name):
        unit.optimization[name] = unit.optimization.get(name, 0) + 1
        self._optimization_totals[name] = self._optimization_totals.get(name, 0) + 1

    @property
    def pending_segments(self):
        with self._input_lock, self._runtime.coordination:
            if self._input is None:
                return len(self._units)
            unassigned = sum(segment.turn is None for segment in self._input.segments)
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

    def resume(self, *, expected_audio_samples=None):
        """自动输入的录音段由门控负责，不把继续误作重新开始分句。"""
        raise ASRError("MODE_UNAVAILABLE", "resume", "同轮继续只适用于手动输入")

    def current_audio(self):
        """先递交已形成的确定音频，不把未确定回溯尾部归给当前单元。"""
        with self._input_lock, self._runtime.coordination:
            state = self._require_input()
            self._pump(state)
            return super().current_audio()

    def feed_audio(self, chunk):
        """本会话串行转换和检测；不持有全局锁，也不等待 ASR 模型。"""
        with self._input_lock:
            with self._runtime.coordination:
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
            with self._runtime.coordination:
                self._check_generation(state)
                self._pump(state)
            return sum(frame.end_sample - frame.start_sample for frame in frames)

    def _check_generation(self, state):
        self._require_open()
        if self._input is not state:
            raise ASRError("INPUT_CANCELLED", "feed_audio", "检测所属输入已失效")

    def _detect_frames(self, state, frames):
        for frame in frames:
            try:
                allowed, clear_before = (
                    state.wake.process(frame, recording=state.gate.active is not None)
                    if state.wake
                    else (True, None)
                )
                activity = (
                    state.detector.detect(
                        frame.pcm, frame.start_sample, recording=state.gate.active is not None
                    )
                    if allowed
                    else VoiceActivity(False, False, False, False, 0)
                )
                if not isinstance(activity, VoiceActivity):
                    raise ASRError("VAD_FAILED", "detect_vad", "检测器应返回 VoiceActivity")
            except ASRError:
                raise
            except Exception as exc:
                raise ASRError("VAD_FAILED", "detect_vad", str(exc)) from exc
            with self._runtime.coordination:
                self._check_generation(state)
                if clear_before is not None:
                    state.gate.discard_history_before(clear_before)
                    if clear_before >= frame.end_sample:
                        state.gate.position = frame.end_sample
                        continue
                    if clear_before > frame.start_sample:
                        frame = AudioFrame(
                            frame.pcm[(clear_before - frame.start_sample) * 2 :],
                            clear_before,
                            frame.end_sample,
                        )
                if state.cancel_current:
                    # 转换已经接纳本块；取消掉本块尚未登记的段，保留转换时钟。
                    state.gate.position = frames[-1].end_sample
                    self._apply_cancellations(state)
                    raise ASRError("INPUT_CANCELLED", "feed_audio", "当前输入在检测期间被取消")
                if (
                    activity.speech
                    and state.gate.silence_samples > 0
                    and self._endpoint.early_transcription_silence > 0
                ):
                    state.early_requested = False
                    unit = self._active
                    if unit is not None:
                        unit.endpoint_epoch += 1
                        unit.early_result = None
                        if unit.auxiliary is not None and unit.auxiliary.purpose == "early":
                            self._cancel_auxiliary(unit)
                        self._count_optimization(unit, "speech_resumed")
                text = self._active.last_realtime_text if self._active is not None else ""
                pause = self._endpoint.pause_for(text, self._segmentation.post_speech_silence)
                was_recording = state.gate.active is not None
                state.segments.extend(state.gate.feed(frame, activity, post_speech_silence=pause))
                if state.wake:
                    if not was_recording and state.gate.active is not None:
                        state.wake.recording_started()
                    elif was_recording and state.gate.active is None:
                        state.wake.recording_finished(frame.end_sample)
                self._pump(state)
        if state.state not in ("failed", "finished"):
            state.state = "recording" if state.gate.active is not None else "listening"

    def _pump(self, state):
        """按段推进，容量不足保留原段；固定任务不再占自动缓冲。"""
        self._apply_cancellations(state)
        self._apply_sentence_split(state)
        while state.segments:
            segment = state.segments[0]
            try:
                if segment.turn is None:
                    segment.turn = self._begin_unit(AudioFormat())
                    self._active.stream_start_sample = segment.start_sample
                    self._active.wake_context = (
                        dict(state.wake.recording_context or state.wake.status)
                        if state.wake
                        else None
                    )
                unit = self._units[segment.turn.turn_id]
                samples = len(segment.pcm) // 2
                if segment.boundary is None:
                    samples = max(0, samples - int(self._segmentation.backdate_duration * 16000))
                pcm = bytes(segment.pcm[segment.delivered_samples * 2 : samples * 2])
                self._append_standard(unit, pcm)
                segment.delivered_samples = samples
                if segment.boundary is None:
                    if state.gate.active is segment:
                        self._schedule_endpoint(state, unit)
                    return
                unit.boundary = segment.boundary
                Session.end_turn(self)
                state.segments.popleft()
            except ASRError as exc:
                if exc.code in (
                    "TASK_CAPACITY_EXCEEDED",
                    "RESULT_CAPACITY_EXCEEDED",
                    "REALTIME_CAPACITY_EXCEEDED",
                ):
                    return
                raise

    def _new_unit(self, ref, format):
        return _AutomaticUnit(ref, AudioConverter(format))

    def _dispatch_final(self, unit, task, pcm):
        """只由本会话决定精确复用，调度器不读取提前策略状态。"""
        if self.config.completion_mode == "candidate" or self._completion_source == "streaming":
            return super()._dispatch_final(unit, task, pcm)
        early = unit.early_result
        if early is not None and early.matches_audio(pcm):
            unit.task = task
            self._count_optimization(unit, "early_reused")
            self._resolve(task, early.result, timing=early.timing)
            return
        if early is not None:
            unit.optimization["early_reuse_rejected"] = "audio-version-changed"
        super()._dispatch_final(unit, task, pcm)

    def _event(self, unit, kind, **fields):
        event = super()._event(unit, kind, **fields)
        return replace(
            event,
            optimization={
                **(event.optimization or {}),
                **unit.optimization,
                **({"wake": unit.wake_context} if unit.wake_context else {}),
            }
            or None,
            source="early-final"
            if kind == "final" and unit.optimization.get("early_reused")
            else event.source,
        )

    def _terminal(self, unit, kind, **fields):
        if unit.auxiliary is not None:
            self._runtime.cancel_task(unit.auxiliary.request.task.task_id)
            unit.endpoint_epoch += 1
        unit.auxiliary = None
        unit.early_result = unit.pending_split = None
        super()._terminal(unit, kind, **fields)

    def _cancel_auxiliary(self, unit):
        """只在实际移除队列任务后清空在途槽，执行中的容量仍保留。"""
        if self._runtime.cancel_task(unit.auxiliary.request.task.task_id):
            unit.auxiliary = None

    def _submit_auxiliary(self, unit, *, purpose, hint=None):
        """策略固定快照与用途；容量不足只记录优化诊断。"""
        if unit.auxiliary is not None:
            self._count_optimization(unit, "busy_skipped")
            return False
        pcm = bytes(unit.pcm)
        samples = len(pcm) // 2
        if not samples:
            return False
        task = FinalTask(unit.ref, uuid4().hex, samples, samples)
        request = InferenceRequest(
            task,
            pcm,
            self.config.language,
            word_timestamps=purpose == "punctuation" or self.config.word_timestamps,
            priority="auxiliary",
            role="final" if self._runtime.final_enabled else "candidate",
        )
        if not self._runtime.submit(request):
            self._count_optimization(unit, "capacity_skipped")
            return False
        unit.auxiliary = _AuxiliaryRequest(request, purpose, hint, unit.endpoint_epoch)
        self._count_optimization(unit, purpose + "_submitted")
        return True

    def _begin_unit(self, format):
        """新录音段建立独立文本策略，不沿用上一段稳定或候选。"""
        ref = super()._begin_unit(format)
        self._input.early_requested = False
        if self._endpoint.punctuation_split:
            self._active.sentence_policy = SentenceBoundaryPolicy()
        return ref

    def _record_realtime(self, unit, event):
        super()._record_realtime(unit, event)
        if (
            self._active is not unit
            or unit.events is not None
            or unit.sentence_policy is None
            or event.kind != "realtime"
            or event.realtime_role == "ultrafast"
        ):
            return
        stable = unit.realtime_events["accurate"].stabilization
        if not stable.accepted:
            return
        hint = unit.sentence_policy.observe(
            stable.stable_text, event.audio_samples, observation_text=stable.raw_text
        )
        if hint is not None:
            self._submit_auxiliary(unit, purpose="punctuation", hint=hint)

    def _apply_sentence_split(self, state):
        """在输入拥有者与协调锁内提交切分，保留快照之后新增的右侧音频。"""
        unit = self._active
        if unit is None or unit.pending_split is None:
            return
        request = unit.pending_split
        hint, cut, covered = request.hint, request.sample, request.covered_samples
        unit.pending_split = None
        segment = state.gate.active
        minimum = max(8000, int(self._segmentation.min_recording_duration * 16000))
        if (
            segment is None
            or segment.turn != unit.ref
            or unit.sentence_policy.current_hint != hint
            or not minimum <= cut <= covered - 8000
        ):
            self._count_optimization(unit, "punctuation_rejected")
            return
        right = state.gate.split_at(segment.start_sample + cut)
        # 已启动的旧实时观察可能覆盖右侧，先失效，再封口固定左段。
        if self.config.realtime:
            self._runtime.realtime.cancel_unit(unit.ref)
            unit.realtime_events.clear()
        unit.pcm[:] = segment.pcm
        segment.delivered_samples = cut
        unit.early_result = None
        unit.endpoint_epoch += 1
        state.early_requested = False
        state.segments.append(right)
        self._count_optimization(unit, "punctuation_splits")

    def _schedule_endpoint(self, state, unit):
        """静音期间只登记一次辅助快照；不在输入线程等待模型。"""
        threshold = self._endpoint.early_transcription_silence
        confirmed = int(self._segmentation.silence_confirmation * 16000)
        if (
            threshold > 0
            and not state.early_requested
            and len(unit.pcm) // 2 >= int(self._segmentation.min_recording_duration * 16000)
            and (state.gate.silence_samples >= confirmed + int(threshold * 16000))
        ):
            state.early_requested = True
            self._submit_auxiliary(unit, purpose="early")

    def accept_completion(self, request, result, error, timing):
        """通用结果以任务标识交回，策略用途只在本会话解析。"""
        if request.priority == "final":
            return super().accept_completion(request, result, error, timing)
        unit = self._units.get(request.task.turn.turn_id)
        if unit is None or unit.ref != request.task.turn:
            return
        context = unit.auxiliary
        if context is None or context.request.task.task_id != request.task.task_id:
            return (
                super().accept_completion(request, result, error, timing)
                if request.role == "candidate"
                else None
            )
        unit.auxiliary = None
        if (
            self._closed
            or unit.events is not None
            or self._active is not unit
            or (context.purpose == "early" and unit.endpoint_epoch != context.epoch)
        ):
            return
        if error is not None:
            self._count_optimization(unit, context.purpose + "_failed")
            unit.optimization["last_error"] = error.code
            return
        self._count_optimization(unit, context.purpose + "_completed")
        if context.purpose == "early":
            unit.optimization["early_samples"] = request.task.audio_samples
            unit.optimization["early_task_id"] = request.task.task_id
            unit.early_result = EarlyFinalResult(
                request.task.audio_samples, sha256(request.pcm).digest(), result, timing
            )
        else:
            cut = SentenceBoundaryPolicy.locate(context.hint, result)
            if cut is None:
                self._count_optimization(unit, "punctuation_unavailable")
                unit.optimization["punctuation_skip_reason"] = (
                    "timestamps-unavailable"
                    if result.timestamp_status != "reported"
                    else "anchor-mismatch"
                )
            else:
                unit.pending_split = _SentenceCut(context.hint, cut, request.task.audio_samples)

    def _apply_cancellations(self, state):
        """只在输入拥有者内改动分句状态，控制线程只登记失效目标。"""
        if state.cancel_current:
            state.gate.cancel_active()
            state.detector.reset()
            if state.wake:
                state.wake.cancel()
            state.segments = deque(
                segment for segment in state.segments if segment.boundary is not None
            )
            state.cancel_current = False
            state.early_requested = False
            state.state = "listening"
        if state.cancelled_units:
            state.segments = deque(
                segment for segment in state.segments if segment.turn not in state.cancelled_units
            )
            state.cancelled_units.clear()

    def cancel_turn(self, target=None):
        """立即失效任务；检测线程在下一边界清除自己的候选与缓存。"""
        with self._runtime.coordination:
            self._require_open()
            state = self._input
            current = state.gate.active
            ref = target if target is not None else (current.turn if current else None)
            changed = Session.cancel_turn(self, ref) if ref is not None else False
            if current is not None and (target is None or current.turn == target):
                state.cancel_current = True
                changed = True
            if changed and ref is not None:
                state.cancelled_units.add(ref)
            self._runtime.coordination.notify_all()
            return changed

    def end_input(self):
        """只刷新一次源尾部；终止监听但不取消已固定任务。"""
        with self._input_lock:
            with self._runtime.coordination:
                self._require_open()
                state = self._input
                if state.state == "failed":
                    raise ASRError("INPUT_ENDED", "end_input", "输入已失败，请重置")
                self._apply_cancellations(state)
            if state.state != "finished":
                try:
                    self._detect_frames(state, state.converter.finish())
                    state.gate.seal(reason="input_finished")
                    state.state = "finished"
                    if state.wake:
                        state.wake.state = "finished"
                        state.wake.close()
                except ASRError as exc:
                    self._fail_input(state, exc)
                    raise
            with self._runtime.coordination:
                self._check_generation(state)
                self._pump(state)

    def end_turn(self):
        """手动封口当前自动段，继续监听；容量失败可原样重试。"""
        with self._input_lock, self._runtime.coordination:
            state = self._require_input()
            self._apply_cancellations(state)
            segment = state.gate.seal()
            if segment is None:
                if state.segments and state.segments[0].boundary is not None:
                    segment = state.segments[0]
                else:
                    raise ASRError("NO_ACTIVE_UNIT", "end_turn", "没有活动录音段")
            self._pump(state)
            if segment in state.segments:
                raise ASRError("TASK_CAPACITY_EXCEEDED", "end_turn", "封口段已保留，等待容量后重试")
            state.state = "listening"
            if state.wake:
                state.wake.recording_finished(state.gate.position)
            return self._last_task

    def _wait_token(self):
        """只比较会影响递交或消费的公开流程状态，防止等待前漏通知。"""
        return (
            *self._runtime.progress_token,
            tuple(unit.ref for unit in self._units.values() if unit.pending_split is not None),
            self._next_event() is not None,
            id(self._input),
            self._closed,
        )

    def next_event(self, *, timeout=0.0):
        """消费和容量唤醒都推进待递交段，等待不阻塞持续输入。"""
        _validate_timeout(timeout, allow_none=True)
        deadline = None if timeout is None else monotonic() + timeout
        while True:
            with self._input_lock, self._runtime.coordination:
                if self._closed:
                    return None
                if self._runtime_error is not None:
                    return Session.next_event(self)
                state = self._input
                self._pump(state)
                event = Session.next_event(self)
                if event is not None:
                    self._pump(state)
                    return event
                token = self._wait_token()
            remaining = None if deadline is None else max(0, deadline - monotonic())
            if remaining == 0:
                return None
            with self._runtime.coordination:
                if token == self._wait_token():
                    self._runtime.coordination.wait(remaining)
                if self._input is not state and self._runtime_error is None:
                    return None

    def _fail_input(self, state, error):
        """检测错误只失败尚未固定的当前输入，不改写已提交段。"""
        with self._runtime.coordination:
            if self._input is not state or self._closed:
                return
            if self._active is not None and self._active.task is None:
                self._terminal(self._active, "failed", error=error)
            state.gate.cancel_active()
            state.segments.clear()
            state.converter.close()
            state.state = "failed"
            if state.wake:
                state.wake.state = "failed"
                state.wake.close()

    def fail_runtime(self, error):
        """失效自动输入并回收唤醒资源；已固定单元的事件由基础会话保留。"""
        with self._runtime.coordination:
            old = self._input
            # 不等待检测锁，正在检测的调用按原有代次检查丢弃迟到结果。
            self._input = None
            self._closed_wake = old.wake if old is not None else None
            try:
                super().fail_runtime(error)
            finally:
                if old is not None and old.wake is not None:
                    old.wake.state = "failed"
                    old.wake.close()

    def close(self):
        """控制不等待检测锁；正在执行的检测保留自己的本地引用。"""
        with self._runtime.coordination:
            old = self._input
            change = Session.close(self)
            self._input = None
            self._closed_wake = old.wake if old is not None else None
            if old is not None and old.wake is not None:
                old.wake.state = "closed"
                old.wake.close()
            return change

    def reset(self):
        """锁外重新准备检测；旧调用不能向新代次登记结果。"""
        with self._runtime.coordination:
            self._require_open()
        fresh = self._new_input()
        with self._runtime.coordination:
            old = self._input
            try:
                change = Session.reset(self)
            except ASRError:
                if fresh.wake:
                    fresh.wake.close()
                raise
            self._input = fresh
            self._vad_info = fresh.detector.info
            self._optimization_totals = {}
            if old is not None and old.wake is not None:
                old.wake.close()
            return change
