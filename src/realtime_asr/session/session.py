"""会话拥有音频单元、归属和消费顺序，不执行模型推理。"""

from collections import deque
from dataclasses import dataclass, field, replace
from math import isfinite
from uuid import uuid4

import numpy as np

from realtime_asr.audio.conversion import AudioConverter, AudioFormat
from realtime_asr.errors import ASRError
from realtime_asr.results import RecognitionResult
from realtime_asr.segmentation.gate import SegmentBoundary


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

    def __post_init__(self):
        """创建时检查外部配置，不在每个输入块重复验证配置。"""
        if (
            not isinstance(self.realtime, bool)
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


@dataclass(frozen=True)
class SessionChange:
    """重置或关闭交付取消记录，并说明清除和物理退出的范围。"""

    generation: int
    cancelled: tuple[SessionEvent, ...]
    discarded_events: int
    pending_tasks: int


@dataclass
class _AudioUnit:
    """会话拥有音频和事件；后台在协调锁内登记实时退出状态。"""

    ref: UnitRef
    converter: AudioConverter
    pcm: bytearray = field(default_factory=bytearray)
    source_bytes: int = 0
    task: FinalTask | None = None
    events: deque | None = None
    realtime: object = None
    realtime_event: SessionEvent | None = None
    realtime_done: bool = True
    boundary: SegmentBoundary | None = None
    stream_start_sample: int | None = None


class Session:
    """隐藏连续输入、固定音频及输出归属，宿主只操作公开控制接口。"""

    def __init__(self, runtime, config: SessionConfig):
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

    @property
    def session_id(self):
        return self._session_id

    @property
    def config(self):
        return self._config

    @property
    def generation(self):
        with self._runtime._condition:
            return self._generation

    @property
    def state(self):
        with self._runtime._condition:
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
        with self._runtime._condition:
            return self._begin_unit(self.config.format)

    def _begin_unit(self, format):
        """共享单元入口；自动会话递交的音频已经标准化。"""
        with self._runtime._condition:
            self._require_open()
            if self._active is not None:
                raise ASRError("UNIT_ACTIVE", "start", "已有活动音频单元")
            if len(self._units) >= self.config.max_pending_units:
                raise ASRError("RESULT_CAPACITY_EXCEEDED", "start", "未消费单元达到容量上限")
            ref = UnitRef(self.session_id, self._generation, self._next_unit)
            unit = _AudioUnit(ref, AudioConverter(format))
            if self.config.realtime:
                self._runtime._realtime.add_unit(self, unit)
            self._next_unit += 1
            self._units[ref.unit_id] = unit
            self._active = unit
            self._last_task = None
            return ref

    def _append_standard(self, unit, pcm):
        """接纳已转换的相邻采样，保持实时增量与单元缓冲一致。"""
        unit.pcm.extend(pcm)
        if unit.realtime is not None:
            self._runtime._realtime.audio_changed(unit, pcm)

    def feed_audio(self, chunk) -> int:
        """只整理音频，不等待模型；复制已转换采样以隔离调用方修改。"""
        with self._runtime._condition:
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
            if unit.realtime is not None:
                self._runtime._realtime.audio_changed(unit, bytes(unit.pcm[before:]))
            return (len(unit.pcm) - before) // 2

    def _require_active(self):
        if self._active is None:
            raise ASRError("NO_ACTIVE_UNIT", "session", "没有活动音频单元")
        return self._active

    def finish(self) -> FinalTask:
        """刷新尾部并提交不可变完整音频；调用返回后可开始下一单元。"""
        with self._runtime._condition:
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
            if unit.realtime is not None:
                self._runtime._realtime.audio_changed(unit, bytes(unit.pcm[before:]))
                self._runtime._realtime.finish_unit(unit)
            task = self._runtime._submit(self, unit, bytes(unit.pcm))
            unit.task = task
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
        self._runtime._condition.notify_all()

    def _record_realtime(self, unit, event):
        """每个单元只保留最新观察；终态正文不被实时输出覆盖。"""
        if self._units.get(unit.ref.unit_id) is unit and unit.events is None:
            if unit.stream_start_sample is not None:
                event = replace(
                    event,
                    stream_start_sample=unit.stream_start_sample + event.audio_start_sample,
                    stream_end_sample=unit.stream_start_sample + event.audio_samples,
                )
            unit.realtime_event = event

    def cancel(self, target: UnitRef | None = None) -> bool:
        """取消当前输入或指定单元；不扩大到其他单元或会话。"""
        with self._runtime._condition:
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
                unit = self._units.get(target.unit_id)
                if unit is not None and unit.ref != target:
                    raise ASRError("INVALID_TARGET", "cancel", "目标代次与音频单元不匹配")
            if unit is None or unit.events is not None:
                return False
            if unit.task is not None:
                self._runtime._cancel_queued(unit.task.task_id)
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
            error=EventError(error.code, error.operation, error.message) if error else None,
            boundary=unit.boundary,
            stream_start_sample=unit.stream_start_sample,
            stream_end_sample=(
                None if unit.stream_start_sample is None else unit.stream_start_sample + samples
            ),
        )

    def reset(self) -> SessionChange:
        """失效旧代次并直接交付取消记录；保留输入配置。"""
        with self._runtime._condition:
            self._require_open()
            change = self._clear_work()
            self._generation += 1
            self._runtime._condition.notify_all()
            return replace(change, generation=self._generation)

    def _clear_work(self):
        """在协调锁内失效本会话，不释放仍在模型调用中的音频。"""
        cancelled = []
        discarded = 0
        for unit in self._units.values():
            discarded += int(unit.realtime_event is not None)
            if unit.events is not None:
                discarded += len(unit.events)
                continue
            if unit.task is not None:
                self._runtime._cancel_queued(unit.task.task_id)
            self._terminal(unit, "cancelled")
            self._sequence += 1
            cancelled.append(replace(unit.events[0], sequence=self._sequence))
        self._units.clear()
        self._active = None
        self._last_task = None
        running = self._runtime._running
        pending = int(running is not None and running.session is self)
        realtime = self._runtime._realtime
        pending += int(
            realtime is not None and realtime.busy is not None and realtime.busy.session is self
        )
        return SessionChange(self._generation, tuple(cancelled), discarded, pending)

    def close(self) -> SessionChange:
        """只取消自己的工作；共享模型由运行时拥有者关闭。"""
        with self._runtime._condition:
            if self._close_result is None:
                self._close_result = self._clear_work()
                self._closed = True
                self._runtime._sessions.pop(self.session_id, None)
                self._runtime._condition.notify_all()
            return self._close_result

    def _terminal(self, unit, kind, *, result=None, error=None):
        if kind == "cancelled" or (kind == "failed" and unit.task is None):
            unit.realtime_event = None
            if unit.realtime is not None:
                self._runtime._realtime.cancel_unit(unit)
        unit.events = deque([self._event(unit, kind, result=result, error=error)])
        unit.pcm.clear()
        unit.converter.close()
        if self._active is unit:
            self._active = None
        self._runtime._condition.notify_all()

    def _next_event(self):
        """最终正文仍按单元排序；其他单元的实时观察可先交付。"""
        first = next(iter(self._units.values()), None)
        if first is None:
            return None
        if first.events and first.events[0].kind == "completed":
            return first, False
        if first.realtime_event is not None:
            return first, True
        if first.events:
            return first, False
        for unit in self._units.values():
            if unit.realtime_event is not None:
                return unit, True
        return None

    def get_event(self, *, timeout=0.0) -> SessionEvent | None:
        """按单元顺序消费；慢消费者不持有模型执行锁。"""
        _validate_timeout(timeout, allow_none=True)
        with self._runtime._condition:
            generation = self._generation

            def available():
                return (
                    self._closed or generation != self._generation or self._next_event() is not None
                )

            if not self._runtime._condition.wait_for(available, timeout=timeout):
                return None
            if self._closed or generation != self._generation:
                return None
            unit, realtime = self._next_event()
            self._sequence += 1
            if realtime:
                event, unit.realtime_event = unit.realtime_event, None
                return replace(event, sequence=self._sequence)
            event = replace(unit.events.popleft(), sequence=self._sequence)
            if not unit.events:
                del self._units[unit.ref.unit_id]
            return event
