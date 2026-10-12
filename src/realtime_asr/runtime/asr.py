"""共享最终模型与有界任务；协调锁不覆盖模型调用。"""

from collections import deque
from dataclasses import dataclass
from threading import Condition, Lock, Thread
from time import monotonic
from uuid import uuid4

from realtime_asr.audio.conversion import AudioFormat
from realtime_asr.errors import ASRError
from realtime_asr.results import CloseError
from realtime_asr.runtime.realtime import RealtimeConfig, RealtimeExecutor
from realtime_asr.runtime.recognizer import Recognizer
from realtime_asr.session.session import FinalTask, Session, SessionConfig, _validate_timeout


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


@dataclass(frozen=True)
class _Job:
    """执行器持有固定字节，不使用活动缓冲。"""

    session: Session
    task: FinalTask
    pcm: bytes
    unit: object
    submitted_at: float
    purpose: str = "final"
    hint: object = None
    epoch: int = 0
    word_timestamps: bool = False


class ASRRuntime:
    """拥有一个同步模型组件及后台调度，供多个独立会话使用。"""

    def __init__(self, config, *, limits=None, realtime=None):
        self._recognizer = Recognizer(config)
        if realtime is not None and not isinstance(realtime, RealtimeConfig):
            raise ASRError("INVALID_CONFIGURATION", "configure", "需要 RealtimeConfig")
        self._realtime_config = realtime
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
        self._realtime = RealtimeExecutor(self, realtime) if realtime is not None else None

    @property
    def state(self):
        with self._condition:
            return self._state

    @property
    def close_result(self):
        with self._condition:
            return self._close_result

    @property
    def capacity(self):
        with self._condition:
            states = self._realtime.states.values() if self._realtime else ()
            reserved = sum(state.session.config.max_audio_samples for state in states)
            units = len(self._realtime.states) if self._realtime else 0
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
                if self._realtime is not None and not self._realtime.shared:
                    self._realtime.model.prepare()
            except BaseException as exc:
                # 第二角色失败时一起回收，重试使用新资源拥有者。
                errors = self._recognizer.close().errors
                if self._realtime is not None and not self._realtime.shared:
                    errors += self._realtime.model.close().errors
                if errors:
                    exc.cleanup_errors = errors
                self._recognizer = Recognizer(self._recognizer.config)
                if self._realtime is not None:
                    self._realtime = RealtimeExecutor(self, self._realtime_config)
                raise
            if self._realtime is not None:
                self._realtime.thread.start()
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

    def create_auto_session(self, *, vad, segmentation=None, config=None, endpoint=None):
        """显式准备检测器；加载不持有共享协调锁，失败不注册会话。"""
        from realtime_asr.segmentation.gate import SegmentationConfig
        from realtime_asr.segmentation.vad import VadConfig
        from realtime_asr.session.automatic import AutomaticSession

        if not isinstance(vad, VadConfig):
            raise ASRError("INVALID_CONFIGURATION", "configure", "需要 VadConfig")
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
        session = AutomaticSession(self, config, vad, segmentation, endpoint)
        with self._condition:
            self._require_ready()
            if len(self._sessions) >= self._limits.max_sessions:
                raise ASRError("SESSION_CAPACITY_EXCEEDED", "create_session", "开放会话达到上限")
            self._sessions[session.session_id] = session
        return session

    def _submit(self, session, unit, pcm):
        self._require_ready()
        samples = len(pcm) // 2
        early = unit.early_result
        if early is not None and early.matches_audio(pcm):
            task = FinalTask(unit.ref, uuid4().hex, samples, samples, unit.boundary)
            unit.task = task
            session._count_optimization(unit, "early_reused")
            session._resolve(task, early.result, timing=early.timing)
            return task
        if early is not None:
            unit.optimization["early_reuse_rejected"] = "audio-version-changed"
        if (
            self._tasks >= self._limits.max_tasks
            or self._task_samples + samples > self._limits.max_task_audio_samples
        ):
            raise ASRError("TASK_CAPACITY_EXCEEDED", "finish", "最终任务容量不足，保留封口音频")
        task = FinalTask(unit.ref, uuid4().hex, samples, samples, unit.boundary)
        self._queue.append(
            _Job(
                session,
                task,
                pcm,
                unit,
                monotonic(),
                word_timestamps=session.config.word_timestamps,
            )
        )
        self._tasks += 1
        self._task_samples += task.audio_samples
        self._condition.notify_all()
        return task

    def _submit_auxiliary(self, session, unit, *, purpose, hint=None):
        """辅助快照共用物理预算，容量不足只跳过优化，不拒绝输入。"""
        samples = len(unit.pcm) // 2
        if unit.auxiliary is not None:
            session._count_optimization(unit, "busy_skipped")
            return False
        if not samples:
            return False
        if (
            self._tasks >= self._limits.max_tasks
            or self._task_samples + samples > self._limits.max_task_audio_samples
        ):
            session._count_optimization(unit, "capacity_skipped")
            return False
        task = FinalTask(unit.ref, uuid4().hex, samples, samples)
        job = _Job(
            session,
            task,
            bytes(unit.pcm),
            unit,
            monotonic(),
            purpose=purpose,
            hint=hint,
            epoch=unit.endpoint_epoch,
            word_timestamps=purpose == "punctuation" or session.config.word_timestamps,
        )
        unit.auxiliary = job
        self._queue.append(job)
        self._tasks += 1
        self._task_samples += samples
        session._count_optimization(unit, purpose + "_submitted")
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
                        language=job.session.config.language,
                        word_timestamps=job.word_timestamps,
                    )
            except ASRError as exc:
                error = exc
            completed_at = monotonic()
            with self._condition:
                timing = (job.submitted_at, started_at, completed_at)
                if job.purpose == "final":
                    job.session._resolve(job.task, result, error, timing=timing)
                else:
                    job.session._resolve_auxiliary(job, result, error, timing)
                self._tasks -= 1
                self._task_samples -= job.task.audio_samples
                self._running = None
                self._condition.notify_all()
            # 异常栈也可能引用音频；物理任务退出后不在空闲线程中保留它。
            job = result = error = None
        if self._realtime is not None:
            self._realtime.thread.join()
        cleanup = self._recognizer.close()
        with self._condition:
            errors = cleanup.errors + tuple(self._realtime.errors if self._realtime else ())
            self._close_result = RuntimeCloseResult("failed" if errors else "ok", errors=errors)
            self._state = "closed"
            self._condition.notify_all()

    def _cancel_queued(self, task_id):
        """只移除尚未执行的任务；运行引用在调用实际返回后释放。"""
        for job in self._queue:
            if job.task.task_id == task_id:
                self._queue.remove(job)
                self._tasks -= 1
                self._task_samples -= job.task.audio_samples
                self._condition.notify_all()
                if job.purpose != "final" and job.unit.auxiliary is job:
                    job.unit.auxiliary = None
                return True
        return False

    def _has_work_ready(self):
        """本单元原生排空之前，最终任务不能领取模型执行资源。"""
        return any(job.purpose != "final" or job.unit.realtime_done for job in self._queue)

    def _next_job(self):
        """在有界队列中轮转会话，保持每个会话的任务相对顺序。"""
        ready = [job for job in self._queue if job.purpose != "final" or job.unit.realtime_done]
        job = next((job for job in ready if job.purpose == "final"), ready[0])
        self._queue.remove(job)
        others, same_session = deque(), deque()
        for queued in self._queue:
            if queued.session is job.session:
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
                if self._realtime is not None and not self._realtime.shared:
                    errors += self._realtime.model.close().errors
                self._close_result = RuntimeCloseResult("failed" if errors else "ok", errors=errors)
                self._state = "closed"
                return self._close_result
        self._worker.join(timeout)
        with self._condition:
            if self._worker.is_alive():
                realtime_pending = int(
                    self._realtime is not None and self._realtime.busy is not None
                )
                self._close_result = RuntimeCloseResult("pending", self._tasks + realtime_pending)
            return self._close_result
