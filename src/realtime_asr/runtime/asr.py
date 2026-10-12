"""共享最终模型与有界任务；协调锁不覆盖模型调用。"""

from collections import deque
from dataclasses import dataclass, replace
from threading import Condition, Lock, Thread
from time import monotonic

from realtime_asr.audio.conversion import AudioFormat
from realtime_asr.errors import ASRError
from realtime_asr.results import CloseError
from realtime_asr.runtime.realtime import RealtimeConfig, RealtimeGroup
from realtime_asr.runtime.recognizer import Recognizer
from realtime_asr.runtime.tasks import InferenceRequest
from realtime_asr.session.session import Session, SessionConfig, _validate_timeout


@dataclass(frozen=True, kw_only=True)
class RuntimeLimits:
    """限制会话数和物理任务资源，不随输入分包变化。"""

    max_sessions: int = 16
    max_tasks: int = 32
    max_task_audio_samples: int = 9600000

    def __post_init__(self):
        for limit in (self.max_sessions, self.max_tasks, self.max_task_audio_samples):
            if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
                raise ASRError("INVALID_CONFIGURATION", "configure", "运行时容量应为正整数")


@dataclass(frozen=True)
class RuntimeCapacity:
    """当前仍由运行时持有的会话和任务容量。"""

    open_sessions: int
    tasks: int
    task_audio_samples: int
    realtime_units: int = 0
    realtime_reserved_samples: int = 0


@dataclass(frozen=True)
class RuntimeCloseResult:
    """区分等待超时和已经完成清理；pending 不代表已释放模型。"""

    status: str
    pending_tasks: int = 0
    errors: tuple[CloseError, ...] = ()


class ASRRuntime:
    """拥有一个同步模型组件及后台调度，供多个独立会话使用。"""

    def __init__(self, config=None, *, engine=None, limits=None, realtime=None, ultrafast=None):
        self._recognizer = Recognizer(config, engine=engine)
        if not self._recognizer.capabilities.transcription:
            raise ASRError("BACKEND_INCOMPATIBLE", "configure", "最终角色需要整段识别能力")
        if realtime is not None and not isinstance(realtime, RealtimeConfig):
            raise ASRError("INVALID_CONFIGURATION", "configure", "需要 RealtimeConfig")
        self._realtime_config = realtime
        if ultrafast is not None and realtime is None:
            raise ASRError("INVALID_CONFIGURATION", "configure", "极速角色需要准确实时角色")
        self._ultrafast_config = ultrafast
        if limits is not None and not isinstance(limits, RuntimeLimits):
            raise ASRError("INVALID_CONFIGURATION", "configure", "需要 RuntimeLimits")
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
        self._model_lock = Lock()
        self._realtime = RealtimeGroup(self, realtime, ultrafast) if realtime is not None else None

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
                len(self._sessions), self._tasks, self._task_samples, units, reserved
            )

    def __enter__(self):
        self.prepare()
        return self

    def __exit__(self, exc_type, exc, tb):
        result = self.close()
        if exc_type is None and result.status != "ok":
            code = "CLOSE_TIMEOUT" if result.status == "pending" else "CLEANUP_FAILED"
            raise ASRError(code, "close", "运行时关闭未成功")
        return False

    def prepare(self):
        """显式准备后才启动执行线程；模型加载错误沿用 P1。"""
        with self._condition:
            self._require_ready(allow_unprepared=True)
            if self._state == "ready":
                return
            try:
                self._recognizer.prepare()
                if self._realtime is not None:
                    self._realtime.prepare()
            except BaseException as exc:
                # 第二角色失败时一起回收，重试使用新资源拥有者。
                errors = self._recognizer.close().errors
                if self._realtime is not None:
                    errors += self._realtime.close_models()
                if errors:
                    exc.cleanup_errors = errors
                injected = self._recognizer.config is None or (
                    self._realtime is not None and self._realtime.injected
                )
                if injected:
                    self._state = "closed"
                    self._close_result = RuntimeCloseResult(
                        "failed" if errors else "ok", errors=errors
                    )
                else:
                    self._recognizer = Recognizer(self._recognizer.config)
                    if self._realtime is not None:
                        self._realtime = RealtimeGroup(
                            self, self._realtime_config, self._ultrafast_config
                        )
                raise
            if self._realtime is not None:
                self._realtime.start()
            self._worker = Thread(target=self._run, name="ASRFinalWorker", daemon=True)
            self._worker.start()
            self._state = "ready"

    def _require_ready(self, *, allow_unprepared=False):
        codes = {
            "closing": "RUNTIME_CLOSING",
            "closed": "RUNTIME_CLOSED",
            "unprepared": "RUNTIME_NOT_READY",
        }
        if self._state != "ready" and not (allow_unprepared and self._state == "unprepared"):
            raise ASRError(codes[self._state], "runtime", "运行时不能接受新工作")

    def create_session(self, config=None) -> Session:
        """创建独立输入与结果范围，复用当前运行时的模型。"""
        with self._condition:
            self._require_ready()
            if len(self._sessions) >= self._limits.max_sessions:
                raise ASRError("SESSION_CAPACITY_EXCEEDED", "create_session", "开放会话达到上限")
            config = config if config is not None else SessionConfig()
            if not isinstance(config, SessionConfig):
                raise ASRError("INVALID_CONFIGURATION", "configure", "需要 SessionConfig")
            if config.realtime and self._realtime is None:
                raise ASRError("REALTIME_NOT_CONFIGURED", "create_session", "运行时未配置实时角色")
            session = Session(self, config)
            self._sessions[session.session_id] = session
            return session

    def create_auto_session(
        self, *, vad=None, detector_factory=None, segmentation=None, config=None, endpoint=None
    ):
        """显式准备检测器；加载不持有共享协调锁，失败不注册会话。"""
        from realtime_asr.segmentation.gate import SegmentationConfig
        from realtime_asr.segmentation.vad import VadConfig
        from realtime_asr.session.automatic import AutomaticSession

        if (
            (vad is None) == (detector_factory is None)
            or (vad is not None and not isinstance(vad, VadConfig))
            or (detector_factory is not None and not callable(detector_factory))
        ):
            raise ASRError("INVALID_CONFIGURATION", "configure", "VAD 配置与工厂应且仅应提供一个")
        config = config if config is not None else SessionConfig()
        segmentation = segmentation if segmentation is not None else SegmentationConfig()
        if not isinstance(config, SessionConfig) or not isinstance(
            segmentation, SegmentationConfig
        ):
            raise ASRError("INVALID_CONFIGURATION", "configure", "自动会话配置类型无效")
        with self._condition:
            self._require_ready()
            if config.realtime and self._realtime is None:
                raise ASRError("REALTIME_NOT_CONFIGURED", "create_session", "运行时未配置实时角色")
        session = AutomaticSession(
            self, config, vad, segmentation, endpoint, detector_factory=detector_factory
        )
        with self._condition:
            self._require_ready()
            if len(self._sessions) >= self._limits.max_sessions:
                raise ASRError("SESSION_CAPACITY_EXCEEDED", "create_session", "开放会话达到上限")
            self._sessions[session.session_id] = session
        return session

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
        """只报告仍在真实调用中的工作，不把逻辑取消视作物理退出。"""
        pending = int(
            self._running is not None and self._running.task.unit.session_id == session_id
        )
        if self._realtime is not None:
            pending += self._realtime.pending_for(session_id)
        return pending

    def detach_session(self, session_id):
        """注销属于本运行时的会话，已执行任务仍保留自己的固定输入。"""
        self._sessions.pop(session_id, None)
        self._condition.notify_all()

    def submit(self, request: InferenceRequest):
        """只接纳固定任务；辅助容量不足返回 False，最终容量不足保留输入。"""
        self._require_ready()
        samples = request.task.audio_samples
        if (
            self._tasks >= self._limits.max_tasks
            or self._task_samples + samples > self._limits.max_task_audio_samples
        ):
            if request.priority == "auxiliary":
                return False
            raise ASRError("TASK_CAPACITY_EXCEEDED", "finish", "最终任务容量不足，保留封口音频")
        self._queue.append(replace(request, submitted_at=monotonic()))
        self._tasks += 1
        self._task_samples += samples
        self._condition.notify_all()
        return True

    def _run(self):
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
                with self._model_lock:
                    started_at = monotonic()
                    result = self._recognizer.transcribe(
                        job.pcm,
                        format=AudioFormat(),
                        language=job.language,
                        word_timestamps=job.word_timestamps,
                    )
            except ASRError as exc:
                error = exc
            completed_at = monotonic()
            with self._condition:
                timing = (job.submitted_at, started_at, completed_at)
                session = self._sessions.get(job.task.unit.session_id)
                if session is not None:
                    session.accept_completion(job, result, error, timing)
                self._tasks -= 1
                self._task_samples -= job.task.audio_samples
                self._running = None
                self._condition.notify_all()
            # 异常栈也可能引用音频；物理任务退出后不在空闲线程中保留它。
            job = result = error = None
        if self._realtime is not None:
            self._realtime.join()
        cleanup = self._recognizer.close()
        with self._condition:
            errors = cleanup.errors + (self._realtime.close_models() if self._realtime else ())
            self._close_result = RuntimeCloseResult("failed" if errors else "ok", errors=errors)
            self._state = "closed"
            self._condition.notify_all()

    def cancel_task(self, task_id):
        """只移除尚未执行的任务；运行引用在调用实际返回后释放。"""
        for job in self._queue:
            if job.task.task_id == task_id:
                self._queue.remove(job)
                self._tasks -= 1
                self._task_samples -= job.task.audio_samples
                self._condition.notify_all()
                return True
        return False

    def _realtime_ready(self, ref):
        """最终任务的排空屏障由实时拥有者提供，不读取会话单元。"""
        return self._realtime is None or self._realtime.is_complete(ref)

    def _has_work_ready(self):
        """本单元原生排空之前，最终任务不能领取模型执行资源。"""
        return any(
            job.priority != "final" or self._realtime_ready(job.task.unit) for job in self._queue
        )

    def _next_job(self):
        """在有界队列中轮转会话，保持每个会话的任务相对顺序。"""
        ready = [
            job
            for job in self._queue
            if job.priority != "final" or self._realtime_ready(job.task.unit)
        ]
        job = next((job for job in ready if job.priority == "final"), ready[0])
        self._queue.remove(job)
        others, same_session = deque(), deque()
        for queued in self._queue:
            if queued.task.unit.session_id == job.task.unit.session_id:
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
                cleanup = self._recognizer.close()
                errors = cleanup.errors
                if self._realtime is not None:
                    errors += self._realtime.close_models()
                self._close_result = RuntimeCloseResult("failed" if errors else "ok", errors=errors)
                self._state = "closed"
                return self._close_result
        self._worker.join(timeout)
        with self._condition:
            if self._worker.is_alive():
                realtime_pending = self._realtime.pending_for() if self._realtime else 0
                self._close_result = RuntimeCloseResult("pending", self._tasks + realtime_pending)
            return self._close_result
