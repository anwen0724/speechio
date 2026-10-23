"""共享最终模型与有界任务；协调锁不覆盖模型调用。"""

import logging
from collections import deque
from dataclasses import dataclass, replace
from threading import Condition, Lock, Thread
from time import monotonic

from speechio.audio.formats import AudioFormat
from speechio.errors import STTError
from speechio.resources import CloseError
from speechio.stt.options import TranscriptionOptions
from speechio.stt.runtime.candidates import OnDemandConfig, execute_candidate
from speechio.stt.runtime.engines import EngineResources
from speechio.stt.runtime.realtime import RealtimeConfig, RealtimeGroup
from speechio.stt.runtime.tasks import InferenceRequest
from speechio.stt.session.session import Session, SessionConfig, _validate_timeout
from speechio.stt.transcriber import Transcriber

LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler())


@dataclass(frozen=True, kw_only=True)
class RuntimeLimits:
    """限制会话数和物理任务资源，不随输入分包变化。"""

    max_sessions: int = 16
    max_tasks: int = 32
    max_task_audio_samples: int = 9600000

    def __post_init__(self):
        for limit in (self.max_sessions, self.max_tasks, self.max_task_audio_samples):
            if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
                raise STTError("INVALID_CONFIGURATION", "configure", "运行时容量应为正整数")


@dataclass(frozen=True)
class RuntimeCapacity:
    """当前仍由运行时持有的会话和任务容量。"""

    open_sessions: int
    tasks: int
    task_audio_samples: int
    realtime_units: int = 0
    realtime_reserved_samples: int = 0
    retained_audio_samples: int = 0


@dataclass(frozen=True)
class RuntimeCloseResult:
    """区分等待超时和已经完成清理；pending 不代表已释放模型。"""

    status: str
    pending_tasks: int = 0
    errors: tuple[CloseError, ...] = ()


class STTRuntime:
    """拥有一个同步模型组件及后台调度，供多个独立会话使用。"""

    def __init__(self, config, *, limits=None, realtime=None, ultrafast=None, on_demand=None):
        """持有主模型及可选识别角色，资源由运行时统一关闭。"""
        self._initialize(Transcriber(config), limits, realtime, ultrafast, on_demand)

    @classmethod
    def from_adapter(cls, adapter, *, limits=None, realtime=None, ultrafast=None, on_demand=None):
        """从调用方适配器创建共享运行时。"""
        instance = cls.__new__(cls)
        instance._initialize(
            Transcriber.from_adapter(adapter), limits, realtime, ultrafast, on_demand
        )
        return instance

    @classmethod
    def from_on_demand(cls, on_demand, *, limits=None, realtime=None, ultrafast=None):
        """只使用按需模型，不需要虚构一个主模型配置。"""
        if not isinstance(on_demand, OnDemandConfig) or (
            on_demand.config is None and on_demand._adapter is None
        ):
            raise STTError("INVALID_CONFIGURATION", "configure", "需要明确的按需模型配置")
        recognizer = (
            Transcriber(on_demand.config)
            if on_demand._adapter is None
            else Transcriber.from_adapter(on_demand._adapter)
        )
        instance = cls.__new__(cls)
        instance._initialize(
            recognizer, limits, realtime, ultrafast, on_demand, final_enabled=False
        )
        return instance

    def _initialize(
        self, recognizer, limits, realtime, ultrafast, candidate, *, final_enabled=True
    ):
        if candidate is not None and not isinstance(candidate, OnDemandConfig):
            raise STTError("INVALID_CONFIGURATION", "configure", "需要 OnDemandConfig")
        self._final_enabled = final_enabled
        self._recognizer = recognizer
        if not self._recognizer.capabilities.transcription:
            raise STTError("BACKEND_INCOMPATIBLE", "configure", "主角色需要整段识别能力")
        if realtime is not None and not isinstance(realtime, RealtimeConfig):
            raise STTError("INVALID_CONFIGURATION", "configure", "需要 RealtimeConfig")
        self._realtime_config = realtime
        if ultrafast is not None and realtime is None:
            raise STTError("INVALID_CONFIGURATION", "configure", "极速角色需要准确实时角色")
        self._ultrafast_config = ultrafast
        self._on_demand_config = candidate
        if limits is not None and not isinstance(limits, RuntimeLimits):
            raise STTError("INVALID_CONFIGURATION", "configure", "需要 RuntimeLimits")
        self._limits = limits if limits is not None else RuntimeLimits()
        # 会话控制和任务登记共用协调锁，避免跨对象锁顺序和提交竞态。
        self._condition = Condition()
        self._state = "unprepared"
        self._sessions = {}
        self._queue = deque()
        self._running = None
        self._tasks = 0
        self._task_samples = 0
        self._worker = None
        self._close_result = None
        self._worker_error = None
        self._wake_resources = {}
        self._wake_errors = ()
        self._model_lock = Lock()
        self._engines = EngineResources(self._recognizer, self._model_lock)
        self._realtime = RealtimeGroup(self, realtime, ultrafast) if realtime is not None else None
        self._assemble_candidate()

    def _assemble_candidate(self):
        config = self._on_demand_config
        if not self._final_enabled:
            self._candidate = self._recognizer, self._model_lock
        else:
            self._candidate = (
                self._engines.assign(config=config.config, adapter=config._adapter)
                if config is not None
                else None
            )
        if self._candidate is not None and not self._candidate[0].capabilities.transcription:
            raise STTError("BACKEND_INCOMPATIBLE", "configure", "候选角色需要整段识别能力")

    @property
    def final_enabled(self):
        return self._final_enabled

    @property
    def on_demand_config(self):
        return self._on_demand_config

    @property
    def on_demand_capabilities(self):
        return self._candidate[0].capabilities if self._candidate is not None else None

    @property
    def state(self):
        with self._condition:
            return self._state

    @property
    def capabilities(self):
        """最终引擎的固定声明供 SDK 和服务使用，不触发准备。"""
        return self._recognizer.capabilities

    @property
    def realtime_capabilities(self):
        """实时角色可以独立选择引擎，不从最终模型名称推导能力。"""
        return self._realtime.capabilities("accurate") if self._realtime is not None else None

    @property
    def ultrafast_capabilities(self):
        return self._realtime.capabilities("ultrafast") if self._realtime is not None else None

    @property
    def close_result(self):
        with self._condition:
            return self._close_result

    @property
    def capacity(self):
        with self._condition:
            reserved = self._realtime.reserved_samples if self._realtime else 0
            units = self._realtime.unit_count if self._realtime else 0
            return RuntimeCapacity(
                len(self._sessions),
                self._tasks,
                self._task_samples,
                units,
                reserved,
                sum(session.retained_audio_samples for session in self._sessions.values()),
            )

    def __enter__(self):
        self.prepare()
        return self

    def __exit__(self, exc_type, exc, tb):
        result = self.close()
        if exc_type is None and result.status != "ok":
            code = "CLOSE_TIMEOUT" if result.status == "pending" else "CLEANUP_FAILED"
            raise STTError(code, "close", "运行时关闭未成功")
        return False

    def prepare(self):
        """显式准备后才启动执行线程；模型加载错误沿用 P1。"""
        with self._condition:
            self._require_ready(allow_unprepared=True)
            if self._state == "ready":
                return
            try:
                self._engines.prepare(
                    defer_primary=self._final_enabled
                    and self._realtime is not None
                    and self._realtime.accurate_config.mode == "streaming"
                )
            except BaseException as exc:
                # 第二角色失败时一起回收，重试使用新资源拥有者。
                errors = self._engines.close()
                if errors:
                    exc.cleanup_errors = errors
                injected = self._engines.injected
                if injected:
                    self._state = "closed"
                    self._close_result = RuntimeCloseResult(
                        "failed" if errors else "ok", errors=errors
                    )
                else:
                    self._recognizer = Transcriber(self._recognizer.config)
                    self._engines = EngineResources(self._recognizer, self._model_lock)
                    if self._realtime is not None:
                        self._realtime = RealtimeGroup(
                            self, self._realtime_config, self._ultrafast_config
                        )
                    self._assemble_candidate()
                raise
            if self._realtime is not None:
                self._realtime.start()
            self._worker = Thread(target=self._run, name="ASRFinalWorker", daemon=True)
            self._worker.start()
            self._state = "ready"

    def _require_ready(self, *, allow_unprepared=False):
        if self._worker_error is not None:
            error = self._worker_error
            raise STTError(error.code, error.operation, error.message)
        codes = {
            "closing": "RUNTIME_CLOSING",
            "closed": "RUNTIME_CLOSED",
            "unprepared": "RUNTIME_NOT_READY",
        }
        if self._state != "ready" and not (allow_unprepared and self._state == "unprepared"):
            raise STTError(codes[self._state], "runtime", "运行时不能接受新工作")

    def create_session(self, config=None) -> Session:
        """创建独立输入与结果范围，复用当前运行时的模型。"""
        with self._condition:
            self._require_ready()
            if len(self._sessions) >= self._limits.max_sessions:
                raise STTError("SESSION_CAPACITY_EXCEEDED", "create_session", "开放会话达到上限")
            config = config if config is not None else SessionConfig()
            if not isinstance(config, SessionConfig):
                raise STTError("INVALID_CONFIGURATION", "configure", "需要 SessionConfig")
            if config.realtime and self._realtime is None:
                raise STTError("REALTIME_NOT_CONFIGURED", "create_session", "运行时未配置实时角色")
            session = Session(self, config)
            self._sessions[session.session_id] = session
            return session

    def create_auto_session(
        self,
        *,
        config=None,
        vad=None,
        segmentation=None,
        endpoint=None,
        wake=None,
        wake_detector_factory=None,
    ):
        """自动分段；未指定 VAD 时使用随包提供的检测模型。"""
        from speechio.bundled import default_vad_model
        from speechio.stt.segmentation.vad import VadConfig

        vad = vad if vad is not None else VadConfig(model_path=default_vad_model())
        return self._create_auto_session(
            config=config,
            vad=vad,
            segmentation=segmentation,
            endpoint=endpoint,
            wake=wake,
            wake_detector_factory=wake_detector_factory,
        )

    def create_auto_session_from_detector(
        self,
        detector_factory,
        *,
        config=None,
        segmentation=None,
        endpoint=None,
        wake=None,
        wake_detector_factory=None,
    ):
        """由调用方提供语音检测器工厂；不同时接收 VAD 配置。"""
        return self._create_auto_session(
            config=config,
            detector_factory=detector_factory,
            segmentation=segmentation,
            endpoint=endpoint,
            wake=wake,
            wake_detector_factory=wake_detector_factory,
        )

    def _create_auto_session(
        self,
        *,
        vad=None,
        detector_factory=None,
        segmentation=None,
        config=None,
        endpoint=None,
        wake=None,
        wake_detector_factory=None,
    ):
        """显式准备检测器；加载不持有共享协调锁，失败不注册会话。"""
        from speechio.stt.segmentation.gate import SegmentationConfig
        from speechio.stt.segmentation.vad import VadConfig
        from speechio.stt.session.automatic import AutoSession

        if (
            (vad is None) == (detector_factory is None)
            or (vad is not None and not isinstance(vad, VadConfig))
            or (detector_factory is not None and not callable(detector_factory))
        ):
            raise STTError("INVALID_CONFIGURATION", "configure", "VAD 配置与工厂应且仅应提供一个")
        config = config if config is not None else SessionConfig()
        segmentation = segmentation if segmentation is not None else SegmentationConfig()
        if not isinstance(config, SessionConfig) or not isinstance(
            segmentation, SegmentationConfig
        ):
            raise STTError("INVALID_CONFIGURATION", "configure", "自动会话配置类型无效")
        with self._condition:
            self._require_ready()
            if config.realtime and self._realtime is None:
                raise STTError("REALTIME_NOT_CONFIGURED", "create_session", "运行时未配置实时角色")
        session = AutoSession(
            self,
            config,
            vad,
            segmentation,
            endpoint,
            detector_factory=detector_factory,
            wake=wake,
            wake_detector_factory=wake_detector_factory,
        )
        with self._condition:
            try:
                self._require_ready()
                if len(self._sessions) >= self._limits.max_sessions:
                    raise STTError(
                        "SESSION_CAPACITY_EXCEEDED", "create_session", "开放会话达到上限"
                    )
                self._sessions[session.session_id] = session
            except STTError:
                session.close()
                raise
        return session

    def own_wake_resource(self, session_id, resource):
        """只跟踪实际检测生命周期，不拥有门控策略或输入代次。"""
        with self._condition:
            self._wake_resources[resource] = session_id

    def release_wake_resource(self, resource):
        with self._condition:
            self._wake_resources.pop(resource, None)
            self._wake_errors += resource.errors
            self._condition.notify_all()

    @property
    def coordination(self):
        """会话与执行器共同使用的交接条件，不保护外部模型调用。"""
        return self._condition

    @property
    def limits(self):
        return self._limits

    @property
    def realtime(self):
        """实时拥有者的交接入口；会话不操作其内部状态容器。"""
        return self._realtime

    @property
    def progress_token(self):
        """容量或实时退出改变时，自动输入应再次尝试递交。"""
        return self._tasks, self._realtime.unit_count if self._realtime else 0

    def pending_for(self, session_id):
        """原子统计排队、执行和未结束的流；取消中的真实调用仍计入等待。"""
        with self._condition:
            pending = sum(job.task.turn.session_id == session_id for job in self._queue)
            pending += int(
                self._running is not None and self._running.task.turn.session_id == session_id
            )
            if self._realtime is not None:
                pending += self._realtime.pending_for(session_id)
            pending += sum(
                resource.busy
                for resource, owner in self._wake_resources.items()
                if owner == session_id
            )
            return pending

    def detach_session(self, session_id):
        """注销属于本运行时的会话，已执行任务仍保留自己的固定输入。"""
        self._sessions.pop(session_id, None)
        self._condition.notify_all()

    def submit(self, request: InferenceRequest, *, replace_task_id=None):
        """只接纳固定任务；辅助容量不足返回 False，最终容量不足保留输入。"""
        self._require_ready()
        samples = request.reserved_samples
        replaced = next((job for job in self._queue if job.task.task_id == replace_task_id), None)
        old_samples = replaced.reserved_samples if replaced is not None else 0
        if (
            self._tasks - int(replaced is not None) >= self._limits.max_tasks
            or self._task_samples - old_samples + samples > self._limits.max_task_audio_samples
        ):
            if request.priority == "auxiliary":
                return False
            raise STTError("TASK_CAPACITY_EXCEEDED", "finish", "最终任务容量不足，保留封口音频")
        if replaced is not None:
            self.cancel_task(replaced.task.task_id)
        self._queue.append(replace(request, submitted_at=monotonic()))
        self._tasks += 1
        self._task_samples += samples
        self._condition.notify_all()
        return True

    def _run(self):
        """未预期异常终止宿主；正常模型失败仍由任务循环逐项交付。"""
        errors = []
        try:
            self._work_loop()
        except BaseException as exc:
            LOGGER.exception("最终识别工作线程异常退出")
            failure = STTError("RUNTIME_WORKER_FAILED", "worker", f"{type(exc).__name__}: {exc}")
            errors.append(CloseError(failure.code, failure.operation, failure.message))
            with self._condition:
                self._worker_error = errors[0]
                self._state = "closing"
                # 推理调用已经退出，排队任务不再执行，立即释放物理登记。
                self._queue.clear()
                self._running = None
                self._tasks = self._task_samples = 0
                for session in tuple(self._sessions.values()):
                    try:
                        session.fail_runtime(failure)
                    except BaseException as notify_error:
                        errors.append(
                            CloseError("CLEANUP_FAILED", "fail_session", str(notify_error))
                        )
                self._sessions.clear()
                if self._realtime is not None:
                    self._realtime.stop()
                self._condition.notify_all()
        finally:
            # 仅在其他执行器退出后释放共用模型，不销毁仍在使用的资源。
            try:
                if self._realtime is not None:
                    self._realtime.join()
                with self._condition:
                    while self._wake_resources:
                        self._condition.wait()
                errors.extend(self._engines.close())
            except BaseException as cleanup_error:
                errors.append(CloseError("CLEANUP_FAILED", "close", str(cleanup_error)))
            finally:
                with self._condition:
                    errors.extend(self._realtime.errors if self._realtime else ())
                    errors.extend(self._wake_errors)
                    self._close_result = RuntimeCloseResult(
                        "failed" if errors else "ok", errors=tuple(errors)
                    )
                    self._state = "closed"
                    self._condition.notify_all()

    def _work_loop(self):
        """锁外串行推理；锁内回交归属及释放物理容量。"""
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._has_work_ready() or self._state == "closing")
                if not self._queue:
                    break
                job = self._next_job()
                self._running = job
            result, error = None, None
            try:
                model, model_lock = (
                    self._candidate
                    if job.role == "candidate"
                    else (self._recognizer, self._model_lock)
                )
                with model_lock:
                    started_at = monotonic()
                    if job.role == "candidate":
                        result = execute_candidate(model, job)
                    else:
                        result = model.transcribe(
                            job.pcm,
                            format=AudioFormat(),
                            options=TranscriptionOptions(
                                language=job.language, word_timestamps=job.word_timestamps
                            ),
                        )
            except STTError as exc:
                error = exc
            completed_at = monotonic()
            with self._condition:
                timing = (job.submitted_at, started_at, completed_at)
                session = self._sessions.get(job.task.turn.session_id)
                if session is not None:
                    session.accept_completion(job, result, error, timing)
                self._tasks -= 1
                self._task_samples -= job.reserved_samples
                self._running = None
                self._condition.notify_all()
            # 异常栈也可能引用音频；物理任务退出后不在空闲线程中保留它。
            job = result = error = None

    def cancel_task(self, task_id):
        """只移除尚未执行的任务；运行引用在调用实际返回后释放。"""
        for job in self._queue:
            if job.task.task_id == task_id:
                self._queue.remove(job)
                self._tasks -= 1
                self._task_samples -= job.reserved_samples
                self._condition.notify_all()
                return True
        return False

    def _realtime_ready(self, ref):
        """最终任务的排空屏障由实时拥有者提供，不读取会话单元。"""
        return self._realtime is None or self._realtime.is_complete(ref)

    def _has_work_ready(self):
        """本单元原生排空之前，最终任务不能领取模型执行资源。"""
        return any(
            job.priority != "final" or self._realtime_ready(job.task.turn) for job in self._queue
        )

    def _next_job(self):
        """在有界队列中轮转会话，保持每个会话的任务相对顺序。"""
        ready = [
            job
            for job in self._queue
            if job.priority != "final" or self._realtime_ready(job.task.turn)
        ]
        job = next((job for job in ready if job.priority == "final"), ready[0])
        self._queue.remove(job)
        others, same_session = deque(), deque()
        for queued in self._queue:
            if queued.task.turn.session_id == job.task.turn.session_id:
                same_session.append(queued)
            else:
                others.append(queued)
        self._queue = others
        self._queue.extend(same_session)
        return job

    def close(self, *, timeout=5.0) -> RuntimeCloseResult:
        """先失效会话，再限时等待；不释放仍在推理中使用的模型。"""
        _validate_timeout(timeout)
        with self._condition:
            if self._state == "closed":
                return self._close_result
            self._state = "closing"
            for session in tuple(self._sessions.values()):
                session.close()
            if self._realtime is not None:
                self._realtime.stop()
            self._condition.notify_all()
            if self._worker is None:
                errors = self._engines.close()
                self._close_result = RuntimeCloseResult("failed" if errors else "ok", errors=errors)
                self._state = "closed"
                return self._close_result
        self._worker.join(timeout)
        with self._condition:
            if self._worker.is_alive():
                realtime_pending = self._realtime.pending_for() if self._realtime else 0
                self._close_result = RuntimeCloseResult(
                    "pending", self._tasks + realtime_pending + len(self._wake_resources)
                )
            return self._close_result
