"""实时角色配置与后台执行，隐藏快照计时和原生流推进。"""

from dataclasses import dataclass, field, replace
from math import isfinite
from threading import Lock, Thread
from time import monotonic
from uuid import uuid4

import numpy as np

from realtime_asr.errors import ASRError
from realtime_asr.results import CloseError
from realtime_asr.runtime.recognizer import Recognizer, RecognizerConfig
from realtime_asr.segmentation.acoustic import AcousticBoundary, AcousticBoundaryDetector
from realtime_asr.session.session import EventError, SessionEvent


@dataclass(frozen=True, kw_only=True)
class RealtimeConfig:
    """一个实时角色；显式 model 创建独立实例，None 共用最终模型。"""

    mode: str = "snapshot"
    model: RecognizerConfig | None = None
    interval: float = 0.2
    initial_delay: float = 0.2
    min_snapshot_samples: int = 800
    max_units: int = 32
    max_audio_samples: int = 9600000
    acoustic_boundaries: bool = False
    boundary_sensitivity: float = 0.6
    boundary_followup_delays: tuple[float, ...] = (0.05, 0.2)
    boundary_fallback_interval: float = 0.2

    def __post_init__(self):
        if type(self.acoustic_boundaries) is not bool or (
            self.acoustic_boundaries and self.mode != "snapshot"
        ):
            raise ASRError("BACKEND_INCOMPATIBLE", "configure", "声学触发需要快照实时角色")
        if (
            not isinstance(self.boundary_sensitivity, (int, float))
            or not 0 <= self.boundary_sensitivity <= 1
        ):
            raise ASRError("INVALID_CONFIGURATION", "configure", "声学灵敏度无效")
        if (
            not isinstance(self.boundary_followup_delays, tuple)
            or len(self.boundary_followup_delays) > 8
        ):
            raise ASRError("INVALID_CONFIGURATION", "configure", "声学补识别最多八个时间")
        for value in (*self.boundary_followup_delays, self.boundary_fallback_interval):
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not isfinite(value)
                or value < 0
            ):
                raise ASRError("INVALID_CONFIGURATION", "configure", "声学时间应为有限非负秒数")
        if self.boundary_fallback_interval == 0:
            raise ASRError("INVALID_CONFIGURATION", "configure", "声学回退间隔应大于零")
        object.__setattr__(
            self, "boundary_followup_delays", tuple(sorted(set(self.boundary_followup_delays)))
        )
        if self.mode not in ("snapshot", "native") or (
            self.model is not None and not isinstance(self.model, RecognizerConfig)
        ):
            raise ASRError("INVALID_CONFIGURATION", "configure", "实时模式或模型配置无效")
        for seconds in (self.interval, self.initial_delay):
            if (
                not isinstance(seconds, (int, float))
                or isinstance(seconds, bool)
                or not isfinite(seconds)
                or seconds < 0
            ):
                raise ASRError("INVALID_CONFIGURATION", "configure", "实时秒数应为有限非负数")
        if self.interval == 0:
            raise ASRError("INVALID_CONFIGURATION", "configure", "实时执行间隔应大于零")
        for limit in (self.min_snapshot_samples, self.max_units, self.max_audio_samples):
            if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
                raise ASRError("INVALID_CONFIGURATION", "configure", "实时容量应为正整数")


@dataclass
class _RealtimeUnit:
    """跟踪单元的计时和在途观察，不持有可修改的快照。"""

    session: object
    unit: object
    created_at: float
    last_started: float | None = None
    last_samples: int = 0
    invalid: bool = False
    pending: bytearray = field(default_factory=bytearray)
    stream: object = None
    ended: bool = False
    accepted_samples: int = 0
    observation_sequence: int = 0
    detector: AcousticBoundaryDetector | None = None
    boundary: AcousticBoundary | None = None
    boundary_pending: bool = False
    followups: list[float] = field(default_factory=list)
    trigger_reason: str = "periodic"


class RealtimeExecutor:
    """在协调锁外执行模型调用，按单元轮转选择最新完整快照。"""

    def __init__(self, runtime, config):
        self.runtime = runtime
        self.config = config
        self.model = runtime._recognizer
        self.shared = config.model is None
        if not self.shared:
            self.model = Recognizer(config.model)
        expected = "whisper" if config.mode == "snapshot" else "nemotron"
        if self.model.config.backend != expected:
            raise ASRError("BACKEND_INCOMPATIBLE", "configure", "实时路径与模型后端不匹配")
        final_config = runtime._recognizer.config
        if self.model.config.device == final_config.device == "cuda" and (
            self.model.config.cuda_directory != final_config.cuda_directory
            or self.model.config.cudnn_folder != final_config.cudnn_folder
        ):
            raise ASRError("INVALID_CONFIGURATION", "configure", "CUDA 角色需要相同运行库目录")
        self.model_lock = runtime._model_lock if self.shared else Lock()
        self.states = {}
        self.busy = None
        self.stopping = False
        self.errors = []
        self.thread = Thread(target=self._run, name="ASRRealtimeWorker", daemon=True)

    def add_unit(self, session, unit):
        """在开始时预留最大采样容量，登记成功后才占用单元序号。"""
        reserved = sum(state.session.config.max_audio_samples for state in self.states.values())
        if len(self.states) >= self.config.max_units or (
            reserved + session.config.max_audio_samples > self.config.max_audio_samples
        ):
            raise ASRError("REALTIME_CAPACITY_EXCEEDED", "start", "实时单元预留容量不足")
        state = _RealtimeUnit(session, unit, monotonic())
        if self.config.acoustic_boundaries:
            state.detector = AcousticBoundaryDetector(self.config.boundary_sensitivity)
        self.states[unit.ref] = state
        unit.realtime = state
        unit.realtime_done = self.config.mode != "native"

    def audio_changed(self, unit, pcm=b""):
        """快照只唤醒调度；原生路径保存相邻增量，不覆盖未送音频。"""
        if self.config.mode == "native" and not unit.realtime.invalid:
            unit.realtime.pending.extend(pcm)
        if unit.realtime.detector is not None and not unit.realtime.invalid:
            events = unit.realtime.detector.feed(pcm)
            if events:
                unit.realtime.boundary = events[-1]
                unit.realtime.boundary_pending = True
                now = monotonic()
                unit.realtime.followups = [
                    now + delay for delay in self.config.boundary_followup_delays
                ]
        self.runtime._condition.notify_all()

    def finish_unit(self, unit):
        """停止快照或请求原生排空，控制调用不等待模型。"""
        if self.config.mode == "native":
            unit.realtime.ended = True
            self.runtime._condition.notify_all()
        else:
            self.cancel_unit(unit)

    def cancel_unit(self, unit):
        """失效待处理输入，执行中的调用及流由后台退出后清理。"""
        unit.realtime.invalid = True
        unit.realtime.pending.clear()
        unit.realtime.followups.clear()
        self.runtime._condition.notify_all()

    def stop(self):
        """使全部实时单元失效，唤醒执行线程完成资源清理。"""
        self.stopping = True
        for state in self.states.values():
            state.invalid = True
            state.pending.clear()
        self.runtime._condition.notify_all()

    def _select(self):
        """运行中的快照不可替换；尚未运行的版本在选择时固定。"""
        wait = None
        now = monotonic()
        # 封口和清理不等周期；最终任务不能抢占它依赖的原生排空。
        for ref, state in tuple(self.states.items()):
            if state.invalid and state.stream is None:
                self._retire(state)
            elif state.invalid or state.ended:
                pcm = bytes(state.pending)
                state.pending.clear()
                return state, pcm, wait
        for ref, state in tuple(self.states.items()):
            native = self.config.mode == "native"
            samples = len(state.pending) // 2 if native else len(state.unit.pcm) // 2
            acoustic = state.detector is not None
            followup_due = acoustic and bool(state.followups) and state.followups[0] <= now
            if (native and not samples) or (
                not native
                and (
                    samples < self.config.min_snapshot_samples
                    or (
                        samples <= state.last_samples
                        and not state.followups
                        and not state.boundary_pending
                    )
                )
            ):
                continue
            due = state.created_at + self.config.initial_delay
            if state.last_started is not None:
                if not acoustic:
                    due = max(due, state.last_started + self.config.interval)
                if acoustic and not state.boundary_pending and not followup_due:
                    fallback = state.last_started + self.config.boundary_fallback_interval
                    if state.followups:
                        fallback = min(fallback, state.followups[0])
                    due = max(due, fallback)
            if due > now:
                delay = due - now
                wait = delay if wait is None else min(wait, delay)
                continue
            if self.shared and (
                self.runtime._running is not None or self.runtime._has_work_ready()
            ):
                continue
            del self.states[ref]
            self.states[ref] = state
            pcm = bytes(state.pending if native else state.unit.pcm)
            if native:
                state.pending.clear()
            state.trigger_reason = "native" if native else "periodic"
            if acoustic:
                if state.boundary_pending:
                    state.trigger_reason = "acoustic-boundary"
                    state.boundary_pending = False
                elif followup_due:
                    state.trigger_reason = "boundary-followup"
                    state.followups = [deadline for deadline in state.followups if deadline > now]
                else:
                    state.trigger_reason = "boundary-fallback"
            return state, pcm, wait
        return None, None, wait

    def _retire(self, state):
        """调用和流清理退出后释放登记，唤醒依赖排空的最终任务。"""
        self.states.pop(state.unit.ref, None)
        state.pending.clear()
        state.unit.realtime_done = True
        self.runtime._condition.notify_all()

    def _close_stream(self, state):
        """清理错误独立保留，不能覆盖先发生的识别错误。"""
        stream, state.stream = state.stream, None
        if stream is not None:
            try:
                stream.close()
            except ASRError as exc:
                self.errors.append(CloseError(exc.code, exc.operation, exc.message))
                return EventError(exc.code, exc.operation, exc.message)
        return None

    def _native_step(self, state, pcm, *, finishing, cleaning):
        """按领取时的动作处理固定增量，后续封口由下一步补齐余量。"""
        if cleaning:
            return None, self._close_stream(state)
        if state.stream is None:
            state.stream = self.model._create_stream(state.session.config.language)
        if pcm:
            array = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768
            state.stream.accept_audio(array)
            state.accepted_samples += len(pcm) // 2
        if finishing:
            result = state.stream.finish()
            error = self._close_stream(state)
        else:
            state.stream.decode()
            result, error = state.stream.get_result(), None
        return replace(result, requested_language=state.session.config.language), error

    def _run(self):
        """发布时再次核对归属，取消后返回的模型结果不进入会话。"""
        condition = self.runtime._condition
        while True:
            with condition:
                state, pcm, delay = self._select()
                if state is None:
                    if self.stopping and not self.states:
                        break
                    condition.wait(delay)
                    continue
                self.busy = state
                state.last_samples = len(pcm) // 2
                # 输入可以在解码期间继续到达；动作必须与本次领取的增量一致。
                finishing, cleaning = state.ended, state.invalid
                trigger, boundary = state.trigger_reason, state.boundary
            result, error = None, None
            with self.model_lock:
                started = monotonic()
                state.last_started = started
                try:
                    if self.config.mode == "native":
                        result, error = self._native_step(
                            state, pcm, finishing=finishing, cleaning=cleaning
                        )
                    else:
                        result = self.model._snapshot(pcm, state.session.config.language)
                except ASRError as exc:
                    error = EventError(exc.code, exc.operation, exc.message)
                    self._close_stream(state)
            samples = state.accepted_samples if self.config.mode == "native" else len(pcm) // 2
            state.observation_sequence += 1
            event = SessionEvent(
                kind="realtime_failed" if error else "realtime",
                sequence=0,
                unit=state.unit.ref,
                task_id=uuid4().hex,
                audio_samples=samples,
                audio_version=samples,
                result=None if error else result,
                error=error,
                source=self.config.mode,
                started_at_monotonic=started,
                completed_at_monotonic=monotonic(),
                stream_finished=self.config.mode == "native" and finishing and error is None,
                observation_sequence=state.observation_sequence,
                trigger_reason=trigger,
                acoustic_boundary=boundary if self.config.acoustic_boundaries else None,
            )
            with condition:
                if not state.invalid:
                    state.session._record_realtime(state.unit, event)
                    if error:
                        state.invalid = True
                        state.pending.clear()
                if (state.invalid and state.stream is None) or finishing:
                    self._retire(state)
                self.busy = None
                condition.notify_all()
            state = pcm = result = error = event = None
        if not self.shared:
            self.errors.extend(self.model.close().errors)
