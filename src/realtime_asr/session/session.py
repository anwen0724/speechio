"""会话拥有音频单元、归属和消费顺序，不执行模型推理。"""

from collections import deque
from dataclasses import dataclass, field, replace
from math import isfinite
from uuid import uuid4

import numpy as np

from realtime_asr.audio.conversion import AudioConverter, AudioFormat
from realtime_asr.errors import ASRError
from realtime_asr.results import RecognitionResult
from realtime_asr.runtime.tasks import InferenceRequest
from realtime_asr.segmentation.acoustic import AcousticBoundary
from realtime_asr.segmentation.gate import SegmentBoundary
from realtime_asr.session.candidates import CandidateResult, CandidateState, RetainedCorrection
from realtime_asr.text.realtime_merge import (
    RealtimeMergeConfig,
    RealtimeMerger,
    RealtimeMergeResult,
)
from realtime_asr.text.stabilizer import (
    FinalizationResult,
    StabilizationConfig,
    StabilizationResult,
    TextObservation,
    TextStabilizer,
)


def _validate_timeout(timeout, *, allow_none=False):
    """只在两个等待接口的外部边界检查秒数。"""
    if timeout is None and allow_none:
        return
    if (
        not isinstance(timeout, (int, float))
        or isinstance(timeout, bool)
        or not isfinite(timeout)
        or timeout < 0
    ):
        raise ASRError("INVALID_CONFIGURATION", "wait", "超时应为有限非负秒数")


@dataclass(frozen=True, kw_only=True)
class SessionConfig:
    """固定一个会话的源格式、语言和缓冲上限。"""

    format: AudioFormat = field(default_factory=AudioFormat)
    language: str | None = None
    max_audio_samples: int = 480000
    max_pending_units: int = 8
    realtime: bool = False
    stabilization: StabilizationConfig | None = None
    word_timestamps: bool = False
    merge: RealtimeMergeConfig | None = None
    completion_mode: str = "final"
    retain_for_correction: bool = False

    def __post_init__(self):
        """创建时检查外部配置，不在每个输入块重复验证配置。"""
        if (
            not isinstance(self.realtime, bool)
            or type(self.word_timestamps) is not bool
            or self.completion_mode not in ("final", "candidate")
            or type(self.retain_for_correction) is not bool
            or (
                self.merge is not None
                and (not isinstance(self.merge, RealtimeMergeConfig) or not self.realtime)
            )
            or (
                self.stabilization is not None
                and (not isinstance(self.stabilization, StabilizationConfig) or not self.realtime)
            )
            or not isinstance(self.format, AudioFormat)
            or (
                self.language is not None
                and (not isinstance(self.language, str) or not self.language.strip())
            )
        ):
            raise ASRError("INVALID_CONFIGURATION", "configure", "会话格式或语言无效")
        for limit in (self.max_audio_samples, self.max_pending_units):
            if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
                raise ASRError("INVALID_CONFIGURATION", "configure", "会话容量应为正整数")


@dataclass(frozen=True)
class UnitRef:
    """单元归属；重置改变代次但不复用单元序号。"""

    session_id: str
    generation: int
    unit_id: int


@dataclass(frozen=True)
class AudioSnapshot:
    """当前单元已经确定归属的标准音频副本，范围为半开样本区间。"""

    unit: UnitRef
    pcm: bytes
    start_sample: int
    end_sample: int
    sample_rate: int = 16000


@dataclass(frozen=True)
class FinalTask:
    """已接受的封口任务描述，不表示模型已经完成。"""

    unit: UnitRef
    task_id: str
    audio_samples: int
    audio_version: int
    boundary: SegmentBoundary | None = None


@dataclass(frozen=True)
class EventError:
    """可交付错误，不持有异常栈及其中的音频引用。"""

    code: str
    operation: str
    message: str


@dataclass(frozen=True)
class SessionEvent:
    """一次有归属的交付；完成事件与正文事件分开。"""

    kind: str
    sequence: int
    unit: UnitRef
    task_id: str | None
    audio_samples: int
    audio_version: int
    result: RecognitionResult | None = None
    error: EventError | None = None
    source: str = "final"
    audio_start_sample: int = 0
    started_at_monotonic: float | None = None
    completed_at_monotonic: float | None = None
    stream_finished: bool = False
    boundary: SegmentBoundary | None = None
    stream_start_sample: int | None = None
    stream_end_sample: int | None = None
    submitted_at_monotonic: float | None = None
    observation_sequence: int | None = None
    trigger_reason: str | None = None
    acoustic_boundary: AcousticBoundary | None = None
    stabilization: StabilizationResult | None = None
    finalization: FinalizationResult | None = None
    optimization: dict | None = None
    realtime_role: str | None = None
    merge: RealtimeMergeResult | None = None
    candidate: CandidateResult | None = None
    resume_epoch: int = 0
    resume_id: str | None = None
    candidate_start_sample: int = 0


@dataclass(frozen=True)
class SessionChange:
    """重置或关闭交付取消记录，并说明清除和物理退出的范围。"""

    generation: int
    cancelled: tuple[SessionEvent, ...]
    discarded_events: int
    pending_tasks: int


@dataclass
class _AudioUnit:
    """会话独占的可变音频和事件，执行器只取得固定快照。"""

    ref: UnitRef
    converter: AudioConverter
    pcm: bytearray = field(default_factory=bytearray)
    source_bytes: int = 0
    task: FinalTask | None = None
    events: deque | None = None
    realtime_events: dict[str, SessionEvent] = field(default_factory=dict)
    merger: RealtimeMerger | None = None
    boundary: SegmentBoundary | None = None
    stream_start_sample: int | None = None
    stabilizer: TextStabilizer | None = None
    last_realtime_text: str = ""
    candidates: CandidateState | None = None
    candidate_events: dict[str, SessionEvent] = field(default_factory=dict)
    completion_source: str = "final"
    candidate_reused: bool = False
    sealed_pcm: bytes = b""


class Session:
    """隐藏连续输入、固定音频及输出归属，宿主只操作公开控制接口。"""

    def __init__(self, runtime, config: SessionConfig):
        if not runtime.final_enabled and config.completion_mode != "candidate":
            raise ASRError("BACKEND_INCOMPATIBLE", "create_session", "候选专用宿主要求候选完成策略")
        if (config.completion_mode == "candidate" or config.retain_for_correction) and (
            runtime.candidate_config is None
        ):
            raise ASRError("CANDIDATE_NOT_CONFIGURED", "create_session", "完成或修正需要候选角色")
        if config.merge is not None and (
            runtime.realtime is None or not runtime.realtime.has_ultrafast
        ):
            raise ASRError("REALTIME_NOT_CONFIGURED", "create_session", "运行时未配置极速角色")
        self._runtime = runtime
        self._config = config
        self._session_id = uuid4().hex
        self._generation = 0
        self._next_unit = 1
        self._sequence = 0
        self._units = {}
        self._active = None
        self._last_task = None
        self._closed = False
        self._close_result = None
        self._retained = None

    @property
    def active_unit(self):
        """返回当前输入归属，调用方无需读取会话内部音频容器。"""
        with self._runtime.coordination:
            return self._active.ref if self._active is not None else None

    @property
    def retained_audio_samples(self):
        """只计显式修正保留与封口保留，不把引用数当作额外音频副本。"""
        with self._runtime.coordination:
            recent = len(self._retained.pcm) // 2 if self._retained is not None else 0
            return recent + sum(len(unit.sealed_pcm) // 2 for unit in self._units.values())

    @property
    def pending_candidates(self):
        """用于公开排空，已失效的旧调用仍由运行时保留物理容量。"""
        with self._runtime.coordination:
            recent = (
                self._retained.state.pending_count + int(self._retained.event is not None)
                if self._retained is not None
                else 0
            )
            return recent + sum(
                unit.candidates.pending_count + len(unit.candidate_events)
                for unit in self._units.values()
                if unit.candidates is not None
            )

    @property
    def session_id(self):
        return self._session_id

    @property
    def config(self):
        return self._config

    @property
    def generation(self):
        with self._runtime.coordination:
            return self._generation

    @property
    def state(self):
        with self._runtime.coordination:
            if self._closed:
                return "closed"
            if self._active is None:
                return "idle"
            return "sealed" if self._active.converter.state == "finished" else "receiving"

    def _require_open(self):
        if self._closed:
            raise ASRError("SESSION_CLOSED", "session", "会话已经关闭")

    def start(self) -> UnitRef:
        """开始独立音频单元；不覆盖仍在接收的输入。"""
        with self._runtime.coordination:
            return self._begin_unit(self.config.format)

    def _begin_unit(self, format):
        """共享单元入口；自动会话递交的音频已经标准化。"""
        with self._runtime.coordination:
            self._require_open()
            if self._active is not None:
                raise ASRError("UNIT_ACTIVE", "start", "已有活动音频单元")
            if len(self._units) >= self.config.max_pending_units:
                raise ASRError("RESULT_CAPACITY_EXCEEDED", "start", "未消费单元达到容量上限")
            ref = UnitRef(self.session_id, self._generation, self._next_unit)
            unit = self._new_unit(ref, format)
            if self._runtime.candidate_config is not None:
                unit.candidates = CandidateState(ref)
            if self.config.merge is not None:
                unit.merger = RealtimeMerger(ref, self.config.merge)
            if self.config.stabilization is not None:
                unit.stabilizer = TextStabilizer(ref, self.config.stabilization)
            if self.config.realtime:
                self._runtime.realtime.add_unit(self, ref)
            self._clear_correction()
            self._next_unit += 1
            self._units[ref.unit_id] = unit
            self._active = unit
            self._last_task = None
            return ref

    def _new_unit(self, ref, format):
        return _AudioUnit(ref, AudioConverter(format))

    def audio_samples(self, ref):
        """协调条件内查询进度，定时轮询不复制整段历史。"""
        unit = self._units.get(ref.unit_id)
        return len(unit.pcm) // 2 if unit is not None and unit.ref == ref else 0

    def snapshot_audio(self, ref):
        """仅在领取任务时复制该单元的固定标准输入。"""
        unit = self._units.get(ref.unit_id)
        return bytes(unit.pcm) if unit is not None and unit.ref == ref else b""

    def accept_realtime(self, event):
        """执行器只交付有归属的观察，不访问可变单元。"""
        unit = self._units.get(event.unit.unit_id)
        if unit is not None and unit.ref == event.unit:
            self._record_realtime(unit, event)

    def accept_completion(self, request, result, error, timing):
        """协调条件内接收执行结果；发布有效性只由会话判断。"""
        if request.role == "candidate" and request.priority == "auxiliary":
            self._accept_candidate(request, result, error, timing)
            return
        self._resolve(request.task, result, error, timing=timing)

    def _accept_candidate(self, job, result, error, timing):
        context = self._retained
        if context is not None and job.task.unit == context.unit:
            if job.task.task_id == context.state.replacing("correction"):
                self._accept_correction(context, job, result, error, timing)
                return
        unit = self._units.get(job.task.unit.unit_id)
        if unit is None or unit.ref != job.task.unit or unit.events is not None:
            return
        value = unit.candidates.resolve(
            job.task.task_id,
            result if result is not None else RecognitionResult(text=""),
            unit.task.audio_version if unit.task is not None else len(unit.pcm) // 2,
        )
        if value is None:
            return
        if error is not None:
            value = replace(value, status="failed", **getattr(error, "candidate_execution", {}))
        submitted, started, completed = timing
        event = replace(
            self._event(
                unit, "candidate_failed" if error else "candidate", result=result, error=error
            ),
            task_id=value.request.request_id,
            audio_start_sample=value.request.start_sample,
            audio_samples=job.task.audio_samples,
            audio_version=value.request.audio_version,
            candidate=value,
            source=value.request.kind,
            submitted_at_monotonic=submitted,
            started_at_monotonic=started,
            completed_at_monotonic=completed,
        )
        unit.candidate_events.pop(value.request.kind, None)
        unit.candidate_events[value.request.kind] = event
        self._runtime.coordination.notify_all()

    def _accept_correction(self, context, job, result, error, timing):
        """修正只关联原最终任务，不产生第二个最终或完成事件。"""
        value = context.state.resolve(
            job.task.task_id,
            result if result is not None else RecognitionResult(text=""),
            len(context.pcm) // 2,
        )
        if value is None:
            return
        value = replace(value, replaces_task_id=context.task_id)
        if error is None and not value.text:
            error = ASRError("CANDIDATE_EMPTY", "correct", "重试后仍无修正正文，保留原最终正文")
        if error is not None:
            value = replace(value, status="failed", **getattr(error, "candidate_execution", {}))
        submitted, started, completed = timing
        context.event = SessionEvent(
            kind="correction_failed" if error else "correction",
            sequence=0,
            unit=context.unit,
            task_id=job.task.task_id,
            audio_samples=job.task.audio_samples,
            audio_version=job.task.audio_version,
            result=result,
            error=EventError(error.code, error.operation, error.message) if error else None,
            source="correction",
            candidate=value,
            submitted_at_monotonic=submitted,
            started_at_monotonic=started,
            completed_at_monotonic=completed,
            resume_epoch=value.request.resume_epoch,
            resume_id=value.request.resume_id,
            candidate_start_sample=value.request.candidate_start_sample,
        )
        self._runtime.coordination.notify_all()

    def request_correction(self, unit):
        """对最近保留的完整单元接纳修正，输入范围不依赖当前活动单元。"""
        with self._runtime.coordination:
            self._require_open()
            context = self._retained
            if context is None or context.unit != unit:
                raise ASRError("CORRECTION_UNAVAILABLE", "correct", "目标没有可用的整轮修正上下文")
            samples = len(context.pcm) // 2
            request = context.state.describe("correction", False, 0, samples)
            task = FinalTask(unit, request.request_id, samples, samples)
            job = InferenceRequest(
                task,
                context.pcm,
                self.config.language,
                self.config.word_timestamps,
                priority="auxiliary",
                role="candidate",
                retry_empty=True,
            )
            if not self._runtime.submit(job, replace_task_id=context.state.replacing("correction")):
                raise ASRError("CANDIDATE_CAPACITY_EXCEEDED", "correct", "修正任务容量不足")
            context.state.accept(request, "", job.pcm)
            context.event = None
            return request

    def _clear_correction(self):
        if self._retained is not None:
            for request_id in self._retained.state.clear():
                self._runtime.cancel_task(request_id)
            self._retained = None

    def _dispatch_final(self, unit, task, pcm):
        """登记完成前不释放协调条件，立即返回的执行也能关联原单元。"""
        if self.config.completion_mode == "candidate":
            unit.completion_source = "candidate-final"
            cached = unit.candidates.reusable_whole(pcm)
            if cached is not None:
                unit.task = task
                unit.candidate_reused = True
                self._resolve(task, cached)
                return
        self._runtime.submit(
            InferenceRequest(
                task,
                pcm,
                self.config.language,
                self.config.word_timestamps,
                role="candidate" if self.config.completion_mode == "candidate" else "final",
                retry_empty=self.config.completion_mode == "candidate",
            )
        )
        unit.task = task

    def _append_standard(self, unit, pcm):
        """接纳已转换的相邻采样，保持实时增量与单元缓冲一致。"""
        unit.pcm.extend(pcm)
        if self.config.realtime:
            self._runtime.realtime.audio_changed(unit.ref, pcm)

    def feed_audio(self, chunk) -> int:
        """只整理音频，不等待模型；复制已转换采样以隔离调用方修改。"""
        with self._runtime.coordination:
            self._require_open()
            unit = self._require_active()
            if unit.converter.state == "finished":
                raise ASRError("UNIT_SEALED", "feed_audio", "音频单元已经封口")
            scalar_bytes = 2 if self.config.format.sample_type == "int16" else 4
            if isinstance(chunk, (bytes, bytearray, memoryview)):
                source_bytes = len(chunk)
            elif isinstance(chunk, np.ndarray):
                source_bytes = chunk.size * scalar_bytes
            else:
                raise ASRError("INVALID_AUDIO", "feed_audio", "输入应为字节或 NumPy 数组")
            source_frames = self.config.max_audio_samples * self.config.format.sample_rate // 16000
            source_limit = source_frames * self.config.format.channels * scalar_bytes
            if unit.source_bytes + source_bytes > source_limit:
                raise ASRError("AUDIO_CAPACITY_EXCEEDED", "feed_audio", "当前块超过音频容量")
            before = len(unit.pcm)
            try:
                frames = unit.converter.feed(chunk)
            except ASRError as exc:
                if exc.code != "INVALID_AUDIO":
                    self._terminal(unit, "failed", error=exc)
                raise
            for frame in frames:
                unit.pcm.extend(frame.pcm)
            unit.source_bytes += source_bytes
            if self.config.realtime:
                self._runtime.realtime.audio_changed(unit.ref, bytes(unit.pcm[before:]))
            return (len(unit.pcm) - before) // 2

    def _require_active(self):
        if self._active is None:
            raise ASRError("NO_ACTIVE_UNIT", "session", "没有活动音频单元")
        return self._active

    def _drain_ready_audio(self, unit):
        """把已生成的标准短块加入本单元，不结束源转换。"""
        for frame in unit.converter.drain_ready():
            self._append_standard(unit, frame.pcm)

    def audio_snapshot(self):
        """观察已接纳标准短尾，不结束输入或暴露可变缓冲。"""
        with self._runtime.coordination:
            self._require_open()
            unit = self._require_active()
            if unit.task is not None:
                raise ASRError("NO_ACTIVE_UNIT", "audio_snapshot", "当前单元已经封口")
            self._drain_ready_audio(unit)
            pcm = bytes(unit.pcm)
            start = unit.stream_start_sample or 0
            return AudioSnapshot(unit.ref, pcm, start, start + len(pcm) // 2)

    def request_candidate(self, *, kind="whole", hint=False):
        """固定当前输入与准确正文，接纳后立即返回请求描述。"""
        with self._runtime.coordination:
            self._require_open()
            config = self._runtime.candidate_config
            if config is None:
                raise ASRError("CANDIDATE_NOT_CONFIGURED", "candidate", "运行时未配置候选角色")
            if kind not in ("tail", "whole") or type(hint) is not bool:
                raise ASRError("INVALID_CONFIGURATION", "candidate", "候选种类或提示开关无效")
            unit = self._require_active()
            if unit.converter.state != "open":
                raise ASRError("UNIT_SEALED", "candidate", "候选只接受未封口输入")
            self._drain_ready_audio(unit)
            end = len(unit.pcm) // 2
            if not end:
                raise ASRError("INVALID_AUDIO", "candidate", "候选输入没有标准样本")
            start = max(0, end - config.tail_samples) if kind == "tail" else 0
            request = unit.candidates.describe(kind, hint, start, end)
            task = FinalTask(unit.ref, request.request_id, end - start, end)
            job = InferenceRequest(
                task,
                bytes(unit.pcm[start * 2 :]),
                self.config.language,
                word_timestamps=self.config.word_timestamps,
                priority="auxiliary",
                role="candidate",
                retry_empty=not hint,
            )
            if not self._runtime.submit(job, replace_task_id=unit.candidates.replacing(kind)):
                raise ASRError("CANDIDATE_CAPACITY_EXCEEDED", "candidate", "候选任务容量不足")
            unit.candidates.accept(request, unit.last_realtime_text, job.pcm)
            unit.candidate_events.pop(kind, None)
            return request

    def resume(self, *, expected_audio_samples=None):
        """确认标准输入边界，保留同一单元的完整音频和源转换状态。"""
        with self._runtime.coordination:
            self._require_open()
            if self._runtime.candidate_config is None:
                raise ASRError("CANDIDATE_NOT_CONFIGURED", "resume", "运行时未配置候选角色")
            unit = self._require_active()
            self._drain_ready_audio(unit)
            samples = len(unit.pcm) // 2
            if expected_audio_samples is not None:
                if type(expected_audio_samples) is not int or expected_audio_samples < 0:
                    raise ASRError("INVALID_CONFIGURATION", "resume", "预期采样边界应为非负整数")
                if expected_audio_samples != samples:
                    raise ASRError(
                        "RESUME_BOUNDARY_MISMATCH", "resume", "采样边界与已接纳输入不一致"
                    )
            ack, old_requests = unit.candidates.resume(samples)
            for request_id in old_requests:
                self._runtime.cancel_task(request_id)
            unit.candidate_events.clear()
            unit.realtime_events.clear()
            unit.last_realtime_text = ""
            if unit.stabilizer is not None:
                unit.stabilizer = TextStabilizer(unit.ref, self.config.stabilization)
            if unit.merger is not None:
                unit.merger = RealtimeMerger(unit.ref, self.config.merge)
            self._runtime.coordination.notify_all()
            return ack

    def observation_context(self, ref, audio_end_sample):
        """执行器领取输入时固定继续归属，不在迟到回交时改写代次。"""
        unit = self._units.get(ref.unit_id)
        state = unit.candidates if unit is not None and unit.ref == ref else None
        return (
            (state.epoch, state.resume_id, state.start_sample)
            if state is not None
            else (0, None, 0)
        )

    def finish(self) -> FinalTask:
        """刷新尾部并提交不可变完整音频；调用返回后可开始下一单元。"""
        with self._runtime.coordination:
            self._require_open()
            if self._active is None and self._last_task is not None:
                return self._last_task
            unit = self._require_active()
            before = len(unit.pcm)
            try:
                for frame in unit.converter.finish():
                    unit.pcm.extend(frame.pcm)
                if not unit.pcm:
                    raise ASRError("INVALID_AUDIO", "finish", "完整输入没有标准采样")
            except ASRError as exc:
                self._terminal(unit, "failed", error=exc)
                raise
            if self.config.realtime:
                self._runtime.realtime.audio_changed(unit.ref, bytes(unit.pcm[before:]))
                self._runtime.realtime.finish_unit(unit.ref)
            pcm = bytes(unit.pcm)
            samples = len(pcm) // 2
            task = FinalTask(unit.ref, uuid4().hex, samples, samples, unit.boundary)
            if self.config.retain_for_correction:
                unit.sealed_pcm = pcm
            self._dispatch_final(unit, task, pcm)
            unit.pcm.clear()
            unit.converter.close()
            self._active = None
            self._last_task = task
            return task

    def _resolve(self, task, result=None, error=None, *, timing=None):
        """后台结果只登记到仍有效的单元；消费顺序由会话决定。"""
        unit = self._units.get(task.unit.unit_id)
        if unit is None or unit.ref != task.unit or unit.events is not None:
            return
        if error is None and unit.sealed_pcm and unit.ref.unit_id == self._next_unit - 1:
            state = unit.candidates
            retained_state = CandidateState(
                unit.ref,
                epoch=state.epoch,
                resume_id=state.resume_id,
                start_sample=state.start_sample,
            )
            self._clear_correction()
            self._retained = RetainedCorrection(
                unit.ref, task.task_id, unit.sealed_pcm, retained_state
            )
        if error is not None:
            self._terminal(unit, "failed", error=error)
        else:
            self._terminal(unit, "final", result=result)
            unit.events.append(self._event(unit, "completed"))
        if timing is not None:
            submitted, started, completed = timing
            unit.events = deque(
                replace(
                    event,
                    submitted_at_monotonic=submitted,
                    started_at_monotonic=started,
                    completed_at_monotonic=completed,
                )
                for event in unit.events
            )
        self._runtime.coordination.notify_all()

    def _record_realtime(self, unit, event):
        """每个角色保留一个最新槽；只用准确路更新稳定和端点证据。"""
        if self._units.get(unit.ref.unit_id) is unit and unit.events is None:
            if unit.candidates is not None and (
                event.resume_epoch != unit.candidates.epoch
                or (
                    unit.candidates.epoch > 0
                    and event.audio_samples <= unit.candidates.start_sample
                )
            ):
                return
            role = event.realtime_role
            if event.kind == "realtime":
                observation = TextObservation(
                    unit.ref,
                    event.observation_sequence,
                    event.result.text,
                    event.audio_samples,
                    event.completed_at_monotonic,
                )
                stable = None
                if role == "accurate" and unit.stabilizer is not None:
                    stable = unit.stabilizer.observe(observation)
                if (
                    role == "accurate"
                    and event.result.text.strip()
                    and (stable is None or stable.accepted)
                ):
                    unit.last_realtime_text = event.result.text
                event = replace(event, stabilization=stable)
                if unit.merger is not None:
                    unit.merger.observe(role, observation)
            if unit.stream_start_sample is not None:
                event = replace(
                    event,
                    stream_start_sample=unit.stream_start_sample + event.audio_start_sample,
                    stream_end_sample=unit.stream_start_sample + event.audio_samples,
                )
            # 更新角色槽的插入顺序，消费者按最近观察的到达顺序读取。
            unit.realtime_events.pop(role, None)
            unit.realtime_events[role] = event

    def cancel(self, target: UnitRef | None = None) -> bool:
        """取消当前输入或指定单元；不扩大到其他单元或会话。"""
        with self._runtime.coordination:
            self._require_open()
            if target is None:
                unit = self._active
            else:
                if (
                    not isinstance(target, UnitRef)
                    or target.session_id != self.session_id
                    or target.generation > self._generation
                    or target.unit_id < 1
                    or target.unit_id >= self._next_unit
                ):
                    raise ASRError("INVALID_TARGET", "cancel", "目标不属于已创建的会话单元")
                if target.generation < self._generation:
                    return False
                if self._retained is not None and self._retained.unit == target:
                    self._clear_correction()
                    return True
                unit = self._units.get(target.unit_id)
                if unit is not None and unit.ref != target:
                    raise ASRError("INVALID_TARGET", "cancel", "目标代次与音频单元不匹配")
            if unit is None or unit.events is not None:
                return False
            if unit.task is not None:
                self._runtime.cancel_task(unit.task.task_id)
            if self._active is unit:
                self._last_task = None
            self._terminal(unit, "cancelled")
            return True

    def _event(self, unit, kind, *, result=None, error=None):
        task = unit.task
        samples = task.audio_samples if task else len(unit.pcm) // 2
        return SessionEvent(
            kind=kind,
            sequence=0,
            unit=unit.ref,
            task_id=task.task_id if task else None,
            audio_samples=samples,
            audio_version=samples,
            result=result,
            merge=unit.merger.snapshot() if unit.merger is not None else None,
            error=EventError(error.code, error.operation, error.message) if error else None,
            boundary=unit.boundary,
            stream_start_sample=unit.stream_start_sample,
            stream_end_sample=(
                None if unit.stream_start_sample is None else unit.stream_start_sample + samples
            ),
            finalization=(
                unit.stabilizer.finalize(result.text)
                if kind == "final" and unit.stabilizer is not None
                else None
            ),
            resume_epoch=unit.candidates.epoch if unit.candidates is not None else 0,
            resume_id=unit.candidates.resume_id if unit.candidates is not None else None,
            candidate_start_sample=unit.candidates.start_sample
            if unit.candidates is not None
            else 0,
            source=unit.completion_source,
            optimization=(
                {"completion_mode": "candidate", "candidate_reused": unit.candidate_reused}
                if unit.completion_source == "candidate-final"
                else None
            ),
        )

    def reset(self) -> SessionChange:
        """失效旧代次并直接交付取消记录；保留输入配置。"""
        with self._runtime.coordination:
            self._require_open()
            change = self._clear_work()
            self._generation += 1
            self._runtime.coordination.notify_all()
            return replace(change, generation=self._generation)

    def _clear_work(self):
        """在协调锁内失效本会话，不释放仍在模型调用中的音频。"""
        cancelled = []
        discarded = 0
        for unit in self._units.values():
            discarded += len(unit.realtime_events)
            discarded += len(unit.candidate_events)
            if unit.events is not None:
                discarded += len(unit.events)
                continue
            if unit.task is not None:
                self._runtime.cancel_task(unit.task.task_id)
            self._terminal(unit, "cancelled")
            self._sequence += 1
            cancelled.append(replace(unit.events[0], sequence=self._sequence))
        self._units.clear()
        discarded += int(self._retained is not None and self._retained.event is not None)
        self._clear_correction()
        self._active = None
        self._last_task = None
        pending = self._runtime.pending_for(self.session_id)
        return SessionChange(self._generation, tuple(cancelled), discarded, pending)

    def close(self) -> SessionChange:
        """只取消自己的工作；共享模型由运行时拥有者关闭。"""
        with self._runtime.coordination:
            if self._close_result is None:
                self._close_result = self._clear_work()
                self._closed = True
                self._runtime.detach_session(self.session_id)
                self._runtime.coordination.notify_all()
            return self._close_result

    def _terminal(self, unit, kind, *, result=None, error=None):
        if unit.candidates is not None:
            for request_id in unit.candidates.clear():
                self._runtime.cancel_task(request_id)
        if kind == "cancelled" or (kind == "failed" and unit.task is None):
            unit.realtime_events.clear()
            unit.candidate_events.clear()
            if self.config.realtime:
                self._runtime.realtime.cancel_unit(unit.ref)
        unit.events = deque([self._event(unit, kind, result=result, error=error)])
        unit.sealed_pcm = b""
        unit.pcm.clear()
        unit.converter.close()
        if self._active is unit:
            self._active = None
        self._runtime.coordination.notify_all()

    def _next_event(self):
        """最终正文仍按单元排序；其他单元的实时观察可先交付。"""
        if self._retained is not None and self._retained.event is not None:
            return self._retained, "correction"
        first = next(iter(self._units.values()), None)
        if first is None:
            return None
        if first.events and first.events[0].kind == "completed":
            return first, "terminal"
        if first.realtime_events:
            return first, "realtime"
        if first.candidate_events:
            return first, "candidate"
        if first.events:
            return first, "terminal"
        for unit in self._units.values():
            if unit.realtime_events:
                return unit, "realtime"
            if unit.candidate_events:
                return unit, "candidate"
        return None

    def get_event(self, *, timeout=0.0) -> SessionEvent | None:
        """按单元顺序消费；慢消费者不持有模型执行锁。"""
        _validate_timeout(timeout, allow_none=True)
        with self._runtime.coordination:
            generation = self._generation

            def available():
                return (
                    self._closed or generation != self._generation or self._next_event() is not None
                )

            if not self._runtime.coordination.wait_for(available, timeout=timeout):
                return None
            if self._closed or generation != self._generation:
                return None
            unit, source = self._next_event()
            self._sequence += 1
            if source == "correction":
                event = unit.event
                unit.event = None
                return replace(event, sequence=self._sequence)
            if source == "candidate":
                event = unit.candidate_events.pop(next(iter(unit.candidate_events)))
                return replace(event, sequence=self._sequence)
            if source == "realtime":
                event = unit.realtime_events.pop(next(iter(unit.realtime_events)))
                return replace(
                    event,
                    sequence=self._sequence,
                    merge=unit.merger.snapshot() if unit.merger is not None else None,
                )
            event = replace(unit.events.popleft(), sequence=self._sequence)
            if not unit.events:
                del self._units[unit.ref.unit_id]
            return event
