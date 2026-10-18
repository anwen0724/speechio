"""单任务应用服务：通过公开 API 管理模型、会话和结果排空。"""

import asyncio
from dataclasses import asdict, replace

from speechio import (
    ASRError,
    AudioFormat,
    CandidateConfig,
    RealtimeConfig,
    SegmentationConfig,
    SessionConfig,
    StabilizationConfig,
    Transcriber,
    VadConfig,
)
from speechio.bundled import default_vad_model
from speechio.runtime.asr import ASRRuntime

from .protocol import event_message


class SpeechService:
    """一个进程只拥有一个活动任务；相同配置可复用已准备的模型。"""

    def __init__(self, settings, engine_factory=None):
        self.settings = settings
        self.engine_factory = engine_factory
        self.owner = None
        self.runtime = None
        self.batch = None
        self.model_key = None

    def claim(self, owner):
        """在事件循环内占用任务，禁止另一个页面接管会话。"""
        if self.owner is not None:
            raise ASRError("WEB_BUSY", "start", "另一个页面正在使用模型，请先结束该任务")
        if self.runtime is not None and self.runtime.state == "closing":
            raise ASRError("WEB_BUSY", "start", "模型仍在关闭，请等待后台推理返回")
        self.owner = owner

    def validate(self, config, *, debug=False):
        """预设和高级操作在服务端验证，不能只依赖页面隐藏按钮。"""
        if config.model not in self.settings.models:
            raise ASRError("INVALID_CONFIGURATION", "configure", "未知模型预设")
        if config.realtime_model and config.realtime_model not in self.settings.models:
            raise ASRError("INVALID_CONFIGURATION", "configure", "未知实时模型预设")
        if config.wake and config.wake not in self.settings.wakes:
            raise ASRError("INVALID_CONFIGURATION", "configure", "未知唤醒预设")
        if not debug and (config.input_mode == "chunks" or config.candidates or config.correction):
            raise ASRError("DEBUG_DISABLED", "configure", "该功能只对启用的调试台开放")
        model = self.settings.models[config.model]
        realtime_model = self.settings.models.get(config.realtime_model, model)
        caps = realtime_model.capabilities
        if config.realtime and config.input_mode != "file":
            supported = caps.snapshot if config.realtime_mode == "snapshot" else caps.streaming
            if not supported:
                raise ASRError("BACKEND_INCOMPATIBLE", "configure", "模型不支持所选实时方式")

    async def prepare(self, config):
        """准备可能耗时的模型；不阻塞 Web 事件循环。"""
        key = (
            config.model,
            config.realtime,
            config.realtime_mode,
            config.realtime_model,
            config.interval,
            config.candidates,
            config.input_mode == "file",
            config.max_audio_seconds,
        )
        if self.batch is not None and self.model_key == key:
            return
        if self.runtime is not None and self.model_key == key and self.runtime.state == "ready":
            return
        await self.release()
        model = self.settings.models[config.model]
        if config.input_mode == "file":
            options = (
                {"engine": self.engine_factory(model)}
                if self.engine_factory
                else {
                    "config": model,
                }
            )
            self.batch = Transcriber(**options, max_audio_seconds=config.max_audio_seconds)
            self.model_key = key
            return
        realtime = (
            RealtimeConfig(
                mode=config.realtime_mode,
                interval=config.interval,
                model=self.settings.models.get(config.realtime_model),
            )
            if config.realtime
            else None
        )
        candidate = CandidateConfig() if config.candidates else None
        options = {"realtime": realtime, "candidate": candidate}
        if self.engine_factory:
            options["engine"] = self.engine_factory(model)
        else:
            options["config"] = model
        self.runtime = ASRRuntime(**options)
        self.model_key = key
        try:
            await asyncio.to_thread(self.runtime.prepare)
        except BaseException:
            await self.release()
            raise

    async def release(self):
        """保留 pending 的模型引用，不能把超时误报为已释放。"""
        if self.batch is not None:
            result = await asyncio.to_thread(self.batch.close)
            self.batch = None
            self.model_key = None
            if result.status == "failed":
                raise ASRError("CLEANUP_FAILED", "close", str(result.errors))
        if self.runtime is None:
            return {"status": "ok"}
        result = await asyncio.to_thread(self.runtime.close)
        if result.status == "pending":
            raise ASRError("CLOSE_PENDING", "close", "后台推理尚未退出，模型正在关闭")
        self.runtime = None
        self.model_key = None
        if result.status == "failed":
            raise ASRError("CLEANUP_FAILED", "close", "; ".join(e.message for e in result.errors))
        return asdict(result)

    async def open_run(self, config, audio_format=None):
        await self.prepare(config)
        session_config = SessionConfig(
            format=audio_format
            or AudioFormat(sample_rate=config.sample_rate, channels=config.channels),
            language=config.language,
            realtime=config.realtime,
            stabilization=StabilizationConfig() if config.realtime and config.stabilize else None,
            word_timestamps=config.word_timestamps,
            max_audio_samples=config.max_audio_samples,
            max_pending_units=config.max_pending_units,
            retain_for_correction=config.correction,
        )
        if config.segmentation == "manual":
            session = self.runtime.create_session(session_config)
            session.start()
        else:
            wake = self.settings.wakes.get(config.wake)
            if wake:
                wake = replace(
                    wake, sensitivity=config.wake_sensitivity, speech_timeout=config.wake_timeout
                )
            session = await asyncio.to_thread(
                self.runtime.create_auto_session,
                config=session_config,
                vad=VadConfig(
                    model_path=default_vad_model(), silero_sensitivity=config.vad_sensitivity
                ),
                segmentation=SegmentationConfig(
                    post_speech_silence=config.silence,
                    max_buffer_samples=max(1920000, config.max_audio_samples + 17024),
                ),
                wake=wake,
            )
        return ActiveRun(self.runtime, session, config)

    async def transcribe_file(self, config, path):
        """文件整段识别使用 Transcriber，自由配置的时长限制直接交给核心包。"""
        await self.prepare(config)
        return await asyncio.to_thread(
            self.batch.transcribe_file,
            path,
            language=config.language,
            word_timestamps=config.word_timestamps,
        )


class ActiveRun:
    """保留单次输入的状态；停止输入后继续消费核心包的完成事件。"""

    def __init__(self, runtime, session, config):
        self.runtime, self.session, self.config = runtime, session, config
        self.ended = False
        self.finished = False
        self.failed = False
        self.last_turn = None
        self.input_samples = 0

    def feed(self, audio):
        if self.ended:
            raise ASRError("INPUT_ENDED", "feed", "输入已停止")
        if isinstance(audio, bytes) and len(audio) % (2 * self.config.channels):
            raise ASRError("INVALID_AUDIO", "feed", "音频必须是完整的 int16 采样帧")
        self.session.feed_audio(audio)
        self.input_samples += (
            len(audio) // (2 * self.config.channels) if isinstance(audio, bytes) else len(audio)
        )

    def command(self, name):
        """输入、封口和重置由连接串行调用，避免末尾音频越过结束消息。"""
        manual = self.config.segmentation == "manual"
        if name == "end_input":
            if not self.ended:
                if manual:
                    try:
                        self.session.end_turn()
                    except ASRError as error:
                        # 没有采样的手动段仍交付核心包的失败事件，随后结束等待。
                        if error.code != "INVALID_AUDIO" or self.session.active_turn is not None:
                            raise
                else:
                    self.session.end_input()
                self.ended = True
        elif name == "end_turn":
            if self.ended:
                raise ASRError("INPUT_ENDED", name, "输入已停止")
            try:
                self.session.end_turn()
            finally:
                if manual and self.session.active_turn is None:
                    self.session.start()
        elif name == "cancel_turn":
            self.session.cancel_turn()
            if manual and not self.ended:
                self.session.start()
        elif name == "reset":
            self.session.reset()
            self.ended = self.finished = False
            self.last_turn = None
            self.failed = False
            if manual:
                self.session.start()
        elif name == "candidate":
            self.session.request_candidate()
        elif name == "correct":
            if self.last_turn is None:
                raise ASRError("NO_COMPLETED_TURN", name, "还没有可修正的完成段")
            self.session.request_correction(self.last_turn)

    def poll(self):
        messages = []
        while (event := self.session.next_event(timeout=0)) is not None:
            messages.append(event_message(event))
            if event.kind == "failed":
                self.failed = True
            if event.kind == "completed":
                self.last_turn = event.turn
        pending = self.runtime.pending_for(self.session.session_id)
        automatic_pending = getattr(self.session, "pending_segments", 0)
        if self.ended and not self.finished and not pending and not automatic_pending:
            self.finished = True
            messages.append({"type": "finished", "success": not self.failed})
        return messages

    def metrics(self):
        return {
            "type": "metrics",
            "capacity": asdict(self.runtime.capacity),
            "input_seconds": self.input_samples / self.config.sample_rate,
            "input_state": getattr(self.session, "input_state", self.session.state),
            "wake": getattr(self.session, "wake_status", None),
            "retained_audio_samples": self.session.retained_audio_samples,
            "model_state": self.runtime.state,
        }

    def close(self):
        return self.session.close()
