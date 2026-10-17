"""进程级运行时与会话拥有者；协议连接不拥有共享模型。"""

import asyncio
import logging
from collections import Counter
from dataclasses import asdict, replace
from functools import partial
from pathlib import Path
from time import monotonic

from anyio import CancelScope
from starlette.requests import ClientDisconnect

from realtime_asr import (
    ASRError,
    ASRRuntime,
    EndpointConfig,
    RealtimeConfig,
    RealtimeMergeConfig,
    RuntimeLimits,
    SegmentationConfig,
    SessionConfig,
    StabilizationConfig,
    VadConfig,
)
from realtime_asr.segmentation.wakeword import WakeGate


class Service:
    """将模型准备和关闭隐藏在应用生命周期，状态查询不等待模型锁。"""

    def __init__(self, config):
        self.config = config
        if (
            config.realtime
            and config.realtime_config is None
            and config.capabilities.realtime_mode is None
        ):
            raise ASRError("BACKEND_INCOMPATIBLE", "configure_server", "引擎未声明实时能力")
        realtime = config.realtime_config or (
            RealtimeConfig(
                mode="snapshot"
                if config.acoustic_boundaries
                else config.capabilities.realtime_mode,
                acoustic_boundaries=config.acoustic_boundaries,
            )
            if config.realtime
            else None
        )
        self.runtime = ASRRuntime(
            config.recognizer,
            engine=config.engine,
            realtime=realtime,
            ultrafast=config.ultrafast_config,
            candidate=config.candidate_config,
            final_enabled=config.final_enabled,
            limits=RuntimeLimits(
                max_sessions=config.max_sessions,
                max_tasks=config.max_tasks,
                max_task_audio_samples=config.max_task_audio_samples,
            ),
        )
        self.phase = "preparing"
        self.startup_error = None
        self.close_result = None
        self.startup_task = None
        self.shutdown_task = None
        self.sessions = {}
        self.reserved = 0
        self.creating = set()
        self.links = set()
        self.counts = Counter()
        self.durations = {}

    async def open_session(self, options):
        """先保留名额，创建失败或取消后再释放；实际会话归对应请求拥有。"""
        config = self.config
        if options.wake:
            if options.mode != "auto":
                raise ASRError("MODE_UNAVAILABLE", "admit", "唤醒只适用于自动输入")
            if config.wake_config is None:
                raise ASRError("WAKE_NOT_CONFIGURED", "admit", "宿主未配置唤醒")
        if self.phase != "ready":
            raise ASRError("SERVICE_NOT_READY", "admit", "服务未就绪")
        if len(self.sessions) + self.reserved >= config.max_sessions:
            raise ASRError("SESSION_CAPACITY_EXCEEDED", "admit", "会话容量已满")
        seconds = options.max_audio_seconds or config.max_audio_seconds
        pending = options.max_pending_units or config.max_pending_units
        if seconds > config.max_audio_seconds or pending > config.max_pending_units:
            raise ASRError("INVALID_CONFIGURATION", "admit", "请求超过服务新会话上限")
        if options.realtime and not config.realtime:
            raise ASRError("REALTIME_NOT_CONFIGURED", "admit", "服务未配置基本实时")
        if options.dual_realtime and (
            not options.realtime or self.runtime.ultrafast_capabilities is None
        ):
            raise ASRError("REALTIME_NOT_CONFIGURED", "admit", "双实时需要启用实时并配置极速角色")
        if options.completion_mode == "candidate" or options.retain_for_correction:
            if self.runtime.candidate_capabilities is None:
                raise ASRError("CANDIDATE_NOT_CONFIGURED", "admit", "服务未配置候选角色")
        if not config.final_enabled and options.completion_mode != "candidate":
            raise ASRError("MODE_UNAVAILABLE", "admit", "服务只支持候选完成策略")
        if options.mode == "auto" and config.vad_model is None:
            raise ASRError("MODE_UNAVAILABLE", "admit", "服务未配置自动模式")
        if options.mode != "auto" and (
            options.early_transcription_silence
            or options.dynamic_silence
            or options.punctuation_split
        ):
            raise ASRError("INVALID_CONFIGURATION", "admit", "结束优化需要自动模式")
        session_config = SessionConfig(
            format=options.format.to_audio_format(),
            language=options.language,
            realtime=options.realtime,
            max_audio_samples=int(seconds * 16000),
            max_pending_units=pending,
            stabilization=StabilizationConfig() if options.stable_text else None,
            word_timestamps=options.word_timestamps,
            merge=RealtimeMergeConfig() if options.dual_realtime else None,
            completion_mode=options.completion_mode,
            retain_for_correction=options.retain_for_correction,
        )

        def create():
            if options.mode == "manual":
                return self.runtime.create_session(session_config)
            return self.runtime.create_auto_session(
                config=session_config,
                vad=VadConfig(
                    model_path=config.vad_model,
                    webrtc_mode=options.webrtc_mode,
                    silero_end_detection=options.silero_end_detection,
                ),
                segmentation=SegmentationConfig(
                    trim_pre_roll=options.trim_pre_roll,
                    backdate_duration=options.backdate_seconds,
                    max_buffer_samples=max(1920000, session_config.max_audio_samples + 17024),
                ),
                endpoint=EndpointConfig(
                    early_transcription_silence=options.early_transcription_silence,
                    dynamic_silence=options.dynamic_silence,
                    punctuation_split=options.punctuation_split,
                ),
                wake=config.wake_config if options.wake else None,
                wake_detector_factory=config.wake_detector_factory if options.wake else None,
            )

        self.reserved += 1
        task = asyncio.create_task(asyncio.to_thread(create), name="ASRSessionCreate")
        self.creating.add(task)

        def release(finished):
            self.reserved -= 1
            self.creating.discard(finished)

        task.add_done_callback(release)
        try:
            session = await asyncio.shield(task)
        except asyncio.CancelledError:
            # 线程中的创建不能强制中断；返回后仍回收它取得的会话。
            def discard(finished):
                if not finished.cancelled() and finished.exception() is None:
                    asyncio.get_running_loop().run_in_executor(None, finished.result().close)

            task.add_done_callback(discard)
            raise
        if self.phase != "ready":
            await asyncio.to_thread(session.close)
            raise ASRError("SERVICE_NOT_READY", "admit", "服务正在关闭")
        self.sessions[session.session_id] = session
        return session

    async def release_session(self, session):
        """只关闭对应会话；共享模型不随连接释放。"""
        with CancelScope(shield=True):
            change = await asyncio.to_thread(session.close)
            self.sessions.pop(session.session_id, None)
            return change

    def measure(self, name, seconds):
        stats = self.durations.setdefault(name, {"count": 0, "total_ms": 0.0, "max_ms": 0.0})
        milliseconds = seconds * 1000
        stats["count"] += 1
        stats["total_ms"] += milliseconds
        stats["max_ms"] = max(stats["max_ms"], milliseconds)

    def observe(self, event):
        """仅计算有实际执行时间的事件，不对取消和未知值填零。"""
        if event.error is not None:
            self.counts["task_failures"] += 1
        if event.kind not in ("final", "failed", "realtime", "realtime_failed"):
            return
        if event.started_at_monotonic is not None and event.completed_at_monotonic is not None:
            self.measure("execution", event.completed_at_monotonic - event.started_at_monotonic)
        if event.submitted_at_monotonic is not None and event.started_at_monotonic is not None:
            self.measure("queue_wait", event.started_at_monotonic - event.submitted_at_monotonic)

    def metrics(self):
        return {
            "protocol_version": 1,
            "type": "metrics",
            "phase": self.phase,
            "sessions_open": len(self.sessions),
            "sessions_reserved": self.reserved,
            "counts": dict(self.counts),
            "durations": self.durations,
            "runtime_capacity": asdict(self.runtime.capacity) if self.phase == "ready" else None,
        }

    def public_config(self):
        """运行配置按生效范围展示；凭据和私钥不作为响应数据。"""
        active = ("idle_timeout", "drain_timeout", "send_timeout")
        new_sessions = ("max_audio_seconds", "max_pending_units")
        startup = (
            "host",
            "port",
            "max_sessions",
            "max_tasks",
            "max_task_audio_samples",
            "max_http_bytes",
            "max_packet_bytes",
            "max_metadata_bytes",
            "max_pending_messages",
            "open_timeout",
            "body_timeout",
            "close_timeout",
            "realtime",
            "proxy_tls",
            "trusted_proxy_ips",
            "allowed_origins",
        )
        return {
            "protocol_version": 1,
            "type": "config",
            "values": {
                name: getattr(self.config, name) for name in active + new_sessions + startup
            },
            "scope": {
                "active_next_wait": active,
                "new_sessions": new_sessions,
                "startup_only": startup
                + (
                    "recognizer",
                    "engine",
                    "realtime_config",
                    "ultrafast_config",
                    "candidate_config",
                    "final_enabled",
                    "wake_config",
                    "wake_detector_factory",
                    "vad_model",
                    "token",
                    "ssl_certfile",
                    "ssl_keyfile",
                ),
            },
        }

    def update_config(self, patch):
        updated = replace(self.config, **patch)
        self.config = updated
        return self.public_config()

    def capabilities(self):
        config = self.config
        engine = self.runtime.capabilities
        return {
            "protocol_version": 1,
            "type": "capabilities",
            "ready": self.phase == "ready",
            "engine": engine.engine,
            "device": engine.device,
            "manual": True,
            "automatic": config.vad_model is not None,
            "realtime": config.realtime,
            "stable_text": config.realtime,
            "dual_realtime": self.runtime.ultrafast_capabilities is not None,
            "final_enabled": config.final_enabled,
            "wake": {
                "enabled": config.wake_config is not None,
                "backend": config.wake_config.backend if config.wake_config else None,
                "keywords": list(
                    config.wake_config.keywords
                    or tuple(Path(path).stem for path in config.wake_config.model_paths)
                )
                if config.wake_config
                else [],
            },
            "candidate": {
                "enabled": self.runtime.candidate_capabilities is not None,
                "engine": asdict(self.runtime.candidate_capabilities)
                if self.runtime.candidate_capabilities is not None
                else None,
                "kinds": ["tail", "whole"] if config.candidate_config is not None else [],
                "resume": config.candidate_config is not None,
                "correction": config.candidate_config is not None,
                "tail_seconds": config.candidate_config.tail_seconds
                if config.candidate_config is not None
                else None,
                "completion_modes": (["final"] if config.final_enabled else [])
                + (["candidate"] if config.candidate_config is not None else []),
            },
            "realtime_roles": {
                role: asdict(capability) if capability is not None else None
                for role, capability in (
                    ("accurate", self.runtime.realtime_capabilities),
                    ("ultrafast", self.runtime.ultrafast_capabilities),
                )
            },
            "word_timestamps": engine.word_timestamps,
            "acoustic_boundaries": config.realtime
            and self.runtime.realtime.accurate_config.acoustic_boundaries,
            "endpoint_optimization": config.vad_model is not None,
            "punctuation_split": config.vad_model is not None
            and config.realtime
            and engine.word_timestamps,
            "sample_types": ["int16", "float32"],
            "sample_rate_range": [1, 192000],
            "channels_range": [1, 8],
            "authentication_required": config.token is not None,
            "controls": ["start", "finish", "end_input", "cancel", "reset", "ping", "close"]
            + (["candidate", "resume", "correct"] if config.candidate_config is not None else []),
            "limits": {
                name: getattr(config, name)
                for name in (
                    "max_sessions",
                    "max_tasks",
                    "max_task_audio_samples",
                    "max_audio_seconds",
                    "max_pending_units",
                    "max_packet_bytes",
                    "max_metadata_bytes",
                    "max_pending_messages",
                )
            },
        }

    async def transcribe(self, request, options):
        """有限上传、后台最终与完成交付；所有路径回收本请求会话。"""
        if not self.config.final_enabled:
            raise ASRError("MODE_UNAVAILABLE", "transcribe", "候选专用宿主不提供独立 HTTP 最终识别")
        session = await self.open_session(options)
        self.counts["http_requests"] += 1
        request_id = session.session_id
        try:
            await asyncio.to_thread(session.start)

            async def upload():
                count = 0
                frame_bytes = session.config.format.channels * (
                    2 if session.config.format.sample_type == "int16" else 4
                )
                async for chunk in request.stream():
                    before = count
                    count += len(chunk)
                    if count > self.config.max_http_bytes:
                        raise ASRError("HTTP_BODY_TOO_LARGE", "upload", "请求体超过上限")
                    started = monotonic()
                    await asyncio.to_thread(session.feed_audio, chunk)
                    self.measure("input_accept", monotonic() - started)
                    self.counts["input_bytes"] += len(chunk)
                    self.counts["source_frames"] += count // frame_bytes - before // frame_bytes

            try:
                await asyncio.wait_for(upload(), self.config.body_timeout)
            except ClientDisconnect as exc:
                raise ASRError("CLIENT_DISCONNECTED", "upload", "上传期间请求方已断开") from exc
            except TimeoutError as exc:
                raise ASRError("HTTP_BODY_TIMEOUT", "upload", "上传超时") from exc
            await asyncio.to_thread(session.finish)
            deadline = monotonic() + self.config.drain_timeout
            final = None
            while monotonic() < deadline:
                if await request.is_disconnected():
                    raise ASRError("CLIENT_DISCONNECTED", "receive", "请求方已断开")
                event = await asyncio.to_thread(session.get_event, timeout=0.05)
                if event is None:
                    continue
                self.observe(event)
                if event.kind == "failed":
                    raise ASRError(event.error.code, event.error.operation, event.error.message)
                if event.kind == "final":
                    final = event
                if event.kind == "completed":
                    return {
                        "protocol_version": 1,
                        "type": "transcription",
                        "request_id": request_id,
                        "session_id": session.session_id,
                        "event": asdict(final),
                        "result": asdict(final.result),
                    }
            raise ASRError("DRAIN_TIMEOUT", "transcribe", "等待最终与完成超时")
        finally:
            await self.release_session(session)

    def start(self):
        """先返回应用控制权，让 live 在模型准备期间仍可响应。"""
        self.startup_task = asyncio.create_task(self._prepare(), name="ASRServicePrepare")

    async def _prepare(self):
        try:
            await asyncio.to_thread(self.runtime.prepare)
            if self.config.wake_config is not None:
                await asyncio.to_thread(self._prepare_wake)
        except ASRError as exc:
            self.counts["prepare_failures"] += 1
            self.startup_error = {
                "code": exc.code,
                "operation": exc.operation,
                "message": exc.message,
            }
            if self.phase == "preparing":
                self.phase = "failed"
        else:
            if self.phase == "preparing":
                self.phase = "ready"
                scheme = "https" if self.config.ssl_certfile or self.config.proxy_tls else "http"
                logging.getLogger("apps.server.startup").info(
                    "[ready] 模型已就绪：%s://%s:%s（终端持续运行，Ctrl+C 停止）",
                    scheme,
                    self.config.host,
                    self.config.port,
                )

    def _prepare_wake(self):
        """宿主预检与真实会话实例分开，不共享检测特征状态。"""
        probe = WakeGate(
            self.config.wake_config,
            16000,
            self.config.wake_detector_factory,
            on_release=lambda resource: None,
        )
        probe.close()
        if probe.errors:
            error = probe.errors[0]
            raise ASRError(error.code, error.operation, error.message)

    def readiness(self):
        """服务可接纳状态独立于运行时容量，不在加载期间查询其协调锁。"""
        return {
            "protocol_version": 1,
            "type": "ready",
            "phase": self.phase,
            "ready": self.phase == "ready",
            "error": self.startup_error,
        }

    async def _finish_shutdown(self):
        # 先把模型清理交给线程。即使 lifespan 到期并关闭事件循环，
        # 等待准备锁的 close 仍由执行器完成，不依赖已取消的准备协程。
        physical_close = asyncio.get_running_loop().run_in_executor(
            None, partial(self.runtime.close, timeout=self.config.close_timeout)
        )
        for link in tuple(self.links):
            link.abort("SERVICE_SHUTDOWN", "服务关闭", code=1001)
        for session in tuple(self.sessions.values()):
            await self.release_session(session)
        await self.startup_task
        if self.creating:
            await asyncio.gather(*tuple(self.creating), return_exceptions=True)
        result = await asyncio.shield(physical_close)
        self.close_result = asdict(result)
        self.phase = "closed" if result.status != "pending" else "closing"

    async def shutdown(self):
        """限时等待自有模型；未完成准备仍由清理任务继续负责。"""
        self.phase = "closing"
        self.shutdown_task = asyncio.create_task(self._finish_shutdown(), name="ASRServiceClose")
        try:
            await asyncio.wait_for(asyncio.shield(self.shutdown_task), self.config.close_timeout)
        except TimeoutError:
            operations = [task.get_name() for task in self.creating if not task.done()]
            if not self.startup_task.done():
                operations.append("prepare")
            operations.append("runtime_close")
            self.close_result = {
                "status": "pending",
                "pending_tasks": None,
                "pending_operations": operations,
                "open_sessions": len(self.sessions),
            }
