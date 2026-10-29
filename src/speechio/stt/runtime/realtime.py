"""实时角色配置与后台执行，隐藏快照计时和原生流推进。"""

from copy import deepcopy
from dataclasses import dataclass, field, replace
from math import isfinite
from threading import Thread
from time import monotonic
from uuid import uuid4

import numpy as np

from speechio.cancellation import CancellationToken
from speechio.errors import STTError
from speechio.models.config import LocalModelConfig, RemoteModelConfig
from speechio.resources import CloseError
from speechio.stt.results import Transcript
from speechio.stt.segmentation.acoustic import AcousticBoundary, AcousticBoundaryDetector
from speechio.stt.session.session import EventError, SessionEvent


@dataclass(frozen=True, kw_only=True)
class RealtimeConfig:
    """一个实时角色；配置省略时共用主模型，自定义适配器使用 from_adapter。"""

    mode: str = "snapshot"
    config: LocalModelConfig | RemoteModelConfig | None = None
    _adapter: object = field(default=None, init=False, repr=False, compare=False)
    interval_seconds: float = 0.2
    initial_delay_seconds: float = 0.2
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
            raise STTError("BACKEND_INCOMPATIBLE", "configure", "声学触发需要快照实时角色")
        if (
            not isinstance(self.boundary_sensitivity, (int, float))
            or not 0 <= self.boundary_sensitivity <= 1
        ):
            raise STTError("INVALID_CONFIGURATION", "configure", "声学灵敏度无效")
        if (
            not isinstance(self.boundary_followup_delays, tuple)
            or len(self.boundary_followup_delays) > 8
        ):
            raise STTError("INVALID_CONFIGURATION", "configure", "声学补识别最多八个时间")
        for value in (*self.boundary_followup_delays, self.boundary_fallback_interval):
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not isfinite(value)
                or value < 0
            ):
                raise STTError("INVALID_CONFIGURATION", "configure", "声学时间应为有限非负秒数")
        if self.boundary_fallback_interval == 0:
            raise STTError("INVALID_CONFIGURATION", "configure", "声学回退间隔应大于零")
        object.__setattr__(
            self, "boundary_followup_delays", tuple(sorted(set(self.boundary_followup_delays)))
        )
        if (
            self.mode not in ("snapshot", "streaming")
            or (
                self.config is not None
                and not isinstance(self.config, (LocalModelConfig, RemoteModelConfig))
            )
            or (self.config is not None and self._adapter is not None)
        ):
            raise STTError("INVALID_CONFIGURATION", "configure", "实时模式或模型配置无效")
        for seconds in (self.interval_seconds, self.initial_delay_seconds):
            if (
                not isinstance(seconds, (int, float))
                or isinstance(seconds, bool)
                or not isfinite(seconds)
                or seconds < 0
            ):
                raise STTError("INVALID_CONFIGURATION", "configure", "实时秒数应为有限非负数")
        if self.interval_seconds == 0:
            raise STTError("INVALID_CONFIGURATION", "configure", "实时执行间隔应大于零")
        for limit in (self.min_snapshot_samples, self.max_units, self.max_audio_samples):
            if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
                raise STTError("INVALID_CONFIGURATION", "configure", "实时容量应为正整数")

    @classmethod
    def from_adapter(
        cls,
        adapter,
        *,
        mode="snapshot",
        interval_seconds=0.2,
        initial_delay_seconds=0.2,
        min_snapshot_samples=800,
        max_units=32,
        max_audio_samples=9600000,
        acoustic_boundaries=False,
        boundary_sensitivity=0.6,
        boundary_followup_delays=(0.05, 0.2),
        boundary_fallback_interval=0.2,
    ):
        """使用已经构造的识别适配器；运行时负责共享与关闭。"""
        from speechio.adapters.contracts import validate_engine

        validate_engine(adapter)
        instance = cls(
            mode=mode,
            interval_seconds=interval_seconds,
            initial_delay_seconds=initial_delay_seconds,
            min_snapshot_samples=min_snapshot_samples,
            max_units=max_units,
            max_audio_samples=max_audio_samples,
            acoustic_boundaries=acoustic_boundaries,
            boundary_sensitivity=boundary_sensitivity,
            boundary_followup_delays=boundary_followup_delays,
            boundary_fallback_interval=boundary_fallback_interval,
        )
        object.__setattr__(instance, "_adapter", adapter)
        return instance


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
    last_result: Transcript | None = None
    ended: bool = False
    accepted_samples: int = 0
    observation_sequence: int = 0
    detector: AcousticBoundaryDetector | None = None
    boundary: AcousticBoundary | None = None
    boundary_pending: bool = False
    followups: list[float] = field(default_factory=list)
    trigger_reason: str = "periodic"
    cancellation: CancellationToken = field(default_factory=CancellationToken)


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
            raise STTError("BACKEND_INCOMPATIBLE", "configure", "实时路径与模型后端不匹配")
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
            raise STTError("REALTIME_CAPACITY_EXCEEDED", "start", "实时单元预留容量不足")

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
        if self.config.mode == "streaming":
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
        if self.config.mode == "streaming":
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
        state.cancellation.cancel()
        state.pending.clear()
        state.followups.clear()
        self._cancel_stream(state)
        self.runtime.coordination.notify_all()

    def _cancel_stream(self, state):
        """可选中断入口必须非阻塞；旧自定义流仍可在调用返回后清理。"""
        state.cancellation.cancel()
        cancel = getattr(state.stream, "cancel", None)
        if cancel is not None:
            try:
                cancel()
            except Exception as exc:
                self.errors.append(CloseError("CLEANUP_FAILED", "cancel_stream", str(exc)))

    def is_complete(self, ref):
        """本执行器维护排空屏障，不把退出标记写回会话单元。"""
        return self.config.mode != "streaming" or ref not in self.states

    def stop(self):
        """使全部实时单元失效，唤醒执行线程完成资源清理。"""
        self.stopping = True
        for state in self.states.values():
            state.invalid = True
            state.pending.clear()
            self._cancel_stream(state)
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
            streaming = self.config.mode == "streaming"
            samples = (
                len(state.pending) // 2 if streaming else state.session.audio_samples(state.ref)
            )
            acoustic = state.detector is not None
            followup_due = acoustic and bool(state.followups) and state.followups[0] <= now
            if (streaming and not samples and state.stream is None) or (
                not streaming
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
            due = state.created_at + self.config.initial_delay_seconds
            if state.last_started is not None:
                if not acoustic:
                    due = max(due, state.last_started + self.config.interval_seconds)
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
            pcm = bytes(state.pending) if streaming else state.session.turn_audio(state.ref)
            if streaming:
                state.pending.clear()
            state.trigger_reason = "streaming" if streaming else "periodic"
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
                code = exc.code if isinstance(exc, STTError) else "CLEANUP_FAILED"
                operation = exc.operation if isinstance(exc, STTError) else "close_stream"
                message = exc.message if isinstance(exc, STTError) else str(exc)
                self.errors.append(CloseError(code, operation, message))
                return EventError(code, operation, message)
        return None

    def _streaming_step(self, state, pcm, *, finishing, cleaning):
        """按领取时的动作处理固定增量，后续封口由下一步补齐余量。"""
        if cleaning:
            return None, self._close_stream(state)
        if state.stream is None:
            state.stream = self.model._create_stream(
                state.session.config.options.language, cancellation=state.cancellation
            )
        # 建流期间取消时还没有流句柄；建流返回后不能继续发送。
        if state.invalid:
            self._cancel_stream(state)
            return None, self._close_stream(state)
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
            raise STTError("BACKEND_INCOMPATIBLE", "read_stream", "流应返回 Transcript")
        return replace(
            result,
            requested_language=state.session.config.options.language,
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
                    if self.config.mode == "streaming"
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
                    if self.config.mode == "streaming":
                        result, error = self._streaming_step(
                            state, pcm, finishing=finishing, cleaning=cleaning
                        )
                    else:
                        result = self.model._snapshot(
                            pcm,
                            state.session.config.options.language,
                            cancellation=state.cancellation,
                        )
                except STTError as exc:
                    error = EventError(exc.code, exc.operation, exc.message)
                    self._close_stream(state)
                except Exception as exc:
                    error = EventError("INFERENCE_FAILED", "realtime", str(exc))
                    self._close_stream(state)
            samples = state.accepted_samples if self.config.mode == "streaming" else len(pcm) // 2
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
                stream_finished=self.config.mode == "streaming" and finishing and error is None,
                observation_sequence=state.observation_sequence,
                trigger_reason=trigger,
                acoustic_boundary=boundary if self.config.acoustic_boundaries else None,
            )
            with condition:
                if not state.invalid:
                    # 远程结果可在没有新音频时到达；轮询相同结果不重复发布观察。
                    previous = state.last_result
                    if (
                        self.config.mode != "streaming"
                        or pcm
                        or finishing
                        or error
                        or previous is None
                        or (result.text, result.words, result.timestamp_status)
                        != (previous.text, previous.words, previous.timestamp_status)
                    ):
                        state.session.accept_realtime(event)
                    state.last_result = result
                    if error:
                        state.invalid = True
                        state.pending.clear()
                if (state.invalid and state.stream is None) or finishing:
                    self._retire(state)
                    state.session.complete_realtime(state.ref)
                self.busy = None
                condition.notify_all()
            state = pcm = result = error = event = None


class RealtimeGroup:
    """隐藏两路的登记、共享引擎、排空和释放，向会话提供同一交接接口。"""

    def __init__(self, runtime, accurate, ultrafast=None):
        if ultrafast is not None and (
            not isinstance(ultrafast, RealtimeConfig)
            or (ultrafast.config is None and ultrafast._adapter is None)
        ):
            raise STTError("INVALID_CONFIGURATION", "configure", "极速角色需要明确的模型或引擎")
        self.runtime = runtime
        self.executors = {}
        for role, config in (("accurate", accurate), ("ultrafast", ultrafast)):
            if config is None:
                continue
            model, lock = runtime._engines.assign(config=config.config, adapter=config._adapter)
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
            (not state.invalid or executor.busy is state)
            and (session_id is None or state.ref.session_id == session_id)
            for executor in self.executors.values()
            for state in executor.states.values()
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
