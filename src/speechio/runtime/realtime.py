"""实时角色配置与后台执行，隐藏快照计时和原生流推进。"""

from copy import deepcopy
from dataclasses import dataclass, field, replace
from math import isfinite
from threading import Thread
from time import monotonic
from uuid import uuid4

import numpy as np

from speechio.errors import ASRError
from speechio.results import CloseError, Transcript
from speechio.runtime.transcriber import ModelConfig
from speechio.segmentation.acoustic import AcousticBoundary, AcousticBoundaryDetector
from speechio.session.session import EventError, SessionEvent


@dataclass(frozen=True, kw_only=True)
class RealtimeConfig:
    """一个实时角色；model 或 engine 指定拥有者，缺省共用最终引擎。"""

    mode: str = "snapshot"
    model: ModelConfig | None = None
    engine: object = None
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
        if (
            self.mode not in ("snapshot", "native")
            or (self.model is not None and not isinstance(self.model, ModelConfig))
            or (self.model is not None and self.engine is not None)
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
    ref: object
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

    def __init__(self, runtime, config, role, model, model_lock):
        self.runtime = runtime
        self.config = config
        self.role = role
        self.model = model
        self.shared = model is runtime._recognizer
        capabilities = self.model.capabilities
        available = capabilities.snapshot if config.mode == "snapshot" else capabilities.streaming
        if not available:
            raise ASRError("BACKEND_INCOMPATIBLE", "configure", "实时路径与模型后端不匹配")
        self.model_lock = model_lock
        self.states = {}
        self.busy = None
        self.stopping = False
        self.errors = []
        self.thread = Thread(target=self._run, name=f"ASRRealtime-{role}", daemon=True)

    def check_capacity(self, session):
        """组装方先检查所有角色容量，再统一登记，避免单路孤立状态。"""
        reserved = sum(state.session.config.max_audio_samples for state in self.states.values())
        if len(self.states) >= self.config.max_units or (
            reserved + session.config.max_audio_samples > self.config.max_audio_samples
        ):
            raise ASRError("REALTIME_CAPACITY_EXCEEDED", "start", "实时单元预留容量不足")

    def add_unit(self, session, ref):
        """容量检查完成后登记本角色的独立计时、增量和流上下文。"""
        state = _RealtimeUnit(session, ref, monotonic())
        if self.config.acoustic_boundaries:
            state.detector = AcousticBoundaryDetector(self.config.boundary_sensitivity)
        self.states[ref] = state

    def audio_changed(self, ref, pcm=b""):
        """只接收相邻增量与单元标识，流状态由本执行器独占。"""
        state = self.states.get(ref)
        if state is None or state.invalid:
            return
        if self.config.mode == "native":
            state.pending.extend(pcm)
        if state.detector is not None:
            events = state.detector.feed(pcm)
            if events:
                state.boundary = events[-1]
                state.boundary_pending = True
                now = monotonic()
                state.followups = [now + delay for delay in self.config.boundary_followup_delays]
        self.runtime.coordination.notify_all()

    def finish_unit(self, ref):
        """停止快照或请求原生排空，调用方不等待模型。"""
        state = self.states.get(ref)
        if state is None:
            return
        if self.config.mode == "native":
            state.ended = True
            self.runtime.coordination.notify_all()
        else:
            self.cancel_unit(ref)

    def cancel_unit(self, ref):
        """失效待处理输入；执行中的调用及流实际退出后才回收。"""
        state = self.states.get(ref)
        if state is None:
            return
        state.invalid = True
        state.pending.clear()
        state.followups.clear()
        self.runtime.coordination.notify_all()

    def is_complete(self, ref):
        """本执行器维护排空屏障，不把退出标记写回会话单元。"""
        return self.config.mode != "native" or ref not in self.states

    def stop(self):
        """使全部实时单元失效，唤醒执行线程完成资源清理。"""
        self.stopping = True
        for state in self.states.values():
            state.invalid = True
            state.pending.clear()
        self.runtime.coordination.notify_all()

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
            samples = len(state.pending) // 2 if native else state.session.audio_samples(state.ref)
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
            pcm = bytes(state.pending) if native else state.session.turn_audio(state.ref)
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
        self.states.pop(state.ref, None)
        state.pending.clear()
        self.runtime.coordination.notify_all()

    def _close_stream(self, state):
        """清理错误独立保留，不能覆盖先发生的识别错误。"""
        stream, state.stream = state.stream, None
        if stream is not None:
            try:
                stream.close()
            except Exception as exc:
                code = exc.code if isinstance(exc, ASRError) else "CLEANUP_FAILED"
                operation = exc.operation if isinstance(exc, ASRError) else "close_stream"
                message = exc.message if isinstance(exc, ASRError) else str(exc)
                self.errors.append(CloseError(code, operation, message))
                return EventError(code, operation, message)
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
        if not isinstance(result, Transcript):
            raise ASRError("BACKEND_INCOMPATIBLE", "read_stream", "流应返回 Transcript")
        return replace(
            result,
            requested_language=state.session.config.language,
            segments=deepcopy(result.segments),
            words=deepcopy(result.words),
            backend_metadata=deepcopy(result.backend_metadata),
        ), error

    def _run(self):
        """发布时再次核对归属，取消后返回的模型结果不进入会话。"""
        condition = self.runtime.coordination
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
                input_end = (
                    state.accepted_samples + len(pcm) // 2
                    if self.config.mode == "native"
                    else len(pcm) // 2
                )
                epoch, resume_id, candidate_start = state.session.observation_context(
                    state.ref, input_end
                )
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
                except Exception as exc:
                    error = EventError("INFERENCE_FAILED", "realtime", str(exc))
                    self._close_stream(state)
            samples = state.accepted_samples if self.config.mode == "native" else len(pcm) // 2
            state.observation_sequence += 1
            event = SessionEvent(
                kind="realtime_failed" if error else "realtime",
                sequence=0,
                turn=state.ref,
                task_id=uuid4().hex,
                audio_samples=samples,
                audio_version=samples,
                result=None if error else result,
                error=error,
                source=self.config.mode,
                realtime_role=self.role,
                resume_epoch=epoch,
                resume_id=resume_id,
                candidate_start_sample=candidate_start,
                started_at_monotonic=started,
                completed_at_monotonic=monotonic(),
                stream_finished=self.config.mode == "native" and finishing and error is None,
                observation_sequence=state.observation_sequence,
                trigger_reason=trigger,
                acoustic_boundary=boundary if self.config.acoustic_boundaries else None,
            )
            with condition:
                if not state.invalid:
                    state.session.accept_realtime(event)
                    if error:
                        state.invalid = True
                        state.pending.clear()
                if (state.invalid and state.stream is None) or finishing:
                    self._retire(state)
                self.busy = None
                condition.notify_all()
            state = pcm = result = error = event = None


class RealtimeGroup:
    """隐藏两路的登记、共享引擎、排空和释放，向会话提供同一交接接口。"""

    def __init__(self, runtime, accurate, ultrafast=None):
        if ultrafast is not None and (
            not isinstance(ultrafast, RealtimeConfig)
            or (ultrafast.model is None and ultrafast.engine is None)
        ):
            raise ASRError("INVALID_CONFIGURATION", "configure", "极速角色需要明确的模型或引擎")
        self.runtime = runtime
        self.executors = {}
        for role, config in (("accurate", accurate), ("ultrafast", ultrafast)):
            if config is None:
                continue
            model, lock = runtime._engines.assign(model=config.model, engine=config.engine)
            self.executors[role] = RealtimeExecutor(runtime, config, role, model, lock)

    @property
    def has_ultrafast(self):
        return "ultrafast" in self.executors

    def capabilities(self, role):
        executor = self.executors.get(role)
        return executor.model.capabilities if executor is not None else None

    @property
    def accurate_config(self):
        return self.executors["accurate"].config

    @property
    def unit_count(self):
        return sum(len(executor.states) for executor in self.executors.values())

    @property
    def reserved_samples(self):
        return sum(
            state.session.config.max_audio_samples
            for executor in self.executors.values()
            for state in executor.states.values()
        )

    def pending_for(self, session_id=None):
        return sum(
            executor.busy is not None
            and (session_id is None or executor.busy.ref.session_id == session_id)
            for executor in self.executors.values()
        )

    def start(self):
        for executor in self.executors.values():
            executor.thread.start()

    def join(self):
        for executor in self.executors.values():
            executor.thread.join()

    @property
    def errors(self):
        """只聚合流清理错误；角色模型统一由资源拥有者关闭。"""
        return tuple(error for executor in self.executors.values() for error in executor.errors)

    def add_unit(self, session, ref):
        enabled = tuple(
            executor
            for role, executor in self.executors.items()
            if role == "accurate" or session.config.merge is not None
        )
        for executor in enabled:
            executor.check_capacity(session)
        for executor in enabled:
            executor.add_unit(session, ref)

    def audio_changed(self, ref, pcm=b""):
        for executor in self.executors.values():
            executor.audio_changed(ref, pcm)

    def finish_unit(self, ref):
        for executor in self.executors.values():
            executor.finish_unit(ref)

    def cancel_unit(self, ref):
        for executor in self.executors.values():
            executor.cancel_unit(ref)

    def is_complete(self, ref):
        return all(executor.is_complete(ref) for executor in self.executors.values())

    def stop(self):
        for executor in self.executors.values():
            executor.stop()
