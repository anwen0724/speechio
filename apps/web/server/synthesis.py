"""Web 合成任务、临时 WAV 与资源释放；不承担语音模型实现。"""

import asyncio
import contextlib
import wave
from dataclasses import asdict, replace
from time import monotonic
from uuid import uuid4

from anyio import CancelScope
from fastapi import HTTPException, WebSocketDisconnect
from fastapi.responses import FileResponse

from speechio import (
    LocalModelConfig,
    RemoteModelConfig,
    SpeechioError,
    SynthesisOptions,
    Synthesizer,
    TTSError,
    Voice,
)

from .protocol import CloneVoiceRequest, ImportVoiceRequest, SynthesisCommand, SynthesisConfig
from .settings import model_summary
from .voices import Voices


class SynthesisFiles:
    """临时结果在 30 分钟后过期；单次和已完成结果各有 128 MiB 上限。"""

    limit = 128 * 1024 * 1024
    lifetime = 30 * 60

    def __init__(self, directory):
        self.directory = directory / "synthesis"
        self.results = {}

    def create(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        token = uuid4().hex
        return token, self.directory / f"{token}.wav"

    def expire(self, *, all_files=False):
        for token, (path, created, _) in list(self.results.items()):
            if all_files or monotonic() - created >= self.lifetime:
                path.unlink(missing_ok=True)
                del self.results[token]

    def keep(self, token, path):
        self.expire()
        size = path.stat().st_size
        while self.results and sum(item[2] for item in self.results.values()) + size > self.limit:
            oldest = next(iter(self.results))
            self.results.pop(oldest)[0].unlink(missing_ok=True)
        self.results[token] = (path, monotonic(), size)

    def download(self, token):
        self.expire()
        if token not in self.results:
            raise HTTPException(404, "音频已过期或不存在，请重新生成")
        return FileResponse(self.results[token][0], media_type="audio/wav", filename="speechio.wav")


class SynthesisService:
    """复用识别服务的任务占用；关闭未完成时不允许下一次任务进入。"""

    def __init__(self, settings, gate, uploads):
        self.settings = settings
        self.gate = gate
        self.files = SynthesisFiles(settings.upload_dir)
        self.voices = Voices(settings.voices_dir)
        self.uploads = uploads
        self.synth = None
        self.key = None

    def resolve(self, request, *, debug):
        if request.model not in self.settings.synthesis_models:
            raise ValueError("请选择已配置的合成模型")
        if not debug and (
            request.mode == "incremental"
            or request.timeout_seconds is not None
            or request.max_buffer_seconds is not None
        ):
            raise ValueError("此参数只在调试台开放")
        preset = self.settings.synthesis_models[request.model]
        options = asdict(preset.options)
        options.update(request.model_dump(include=set(options), exclude_unset=True))
        if request.saved_voice_id:
            options["voice"] = self.voices.load(request.saved_voice_id, preset.config)
        if request.reference_audio_token:
            if not preset.capabilities.voice_cloning:
                raise TTSError("CAPABILITY_UNSUPPORTED", "clone_voice", "所选模型不支持声音克隆")
            # 本次音源替换默认音色，不能同时将默认 ID 交给核心组合接口。
            options["voice"] = None
        selected = SynthesisOptions(**options)
        model = preset.config
        if request.device is not None:
            if not isinstance(model, LocalModelConfig):
                raise ValueError("运行设备只适用于本地模型")
            # 替换本次配置对象，不改写服务端默认值，也不自动回退到 CPU。
            model = replace(model, device=request.device)
        if request.timeout_seconds is not None:
            if not isinstance(model, RemoteModelConfig):
                raise ValueError("网络等待时间只适用于云端模型")
            model = replace(model, timeout_seconds=request.timeout_seconds)
        buffer_seconds = request.max_buffer_seconds or preset.max_buffer_seconds
        return model, selected, buffer_seconds

    async def prepare(self, request, *, debug):
        model, options, buffer_seconds = self.resolve(request, debug=debug)
        key = (model, buffer_seconds)
        if self.key != key:
            await self.release()
            # 释放前一次本地识别模型，避免闲置模型一直占用内存或显存。
            await self.gate.release()
            self.synth = Synthesizer(model, max_buffer_seconds=buffer_seconds)
            self.key = key
        await asyncio.to_thread(self.synth.prepare)
        caps = self.synth.capabilities
        capability = {
            "full": "synthesis",
            "stream": "audio_streaming",
            "incremental": "text_streaming",
        }[request.mode]
        if not getattr(caps, capability):
            raise TTSError("CAPABILITY_UNSUPPORTED", "synthesize", "模型不支持所选合成方式")
        effective = {
            "model": model_summary(request.model, model, synthesis=True),
            "mode": request.mode,
            "options": {
                **asdict(options),
                "voice": {"saved_voice_id": request.saved_voice_id}
                if isinstance(options.voice, Voice)
                else options.voice,
                "sample_rate": options.sample_rate or caps.default_sample_rate,
            },
            "max_buffer_seconds": buffer_seconds,
        }
        return options, effective

    async def clone(self, request, *, debug):
        """占用同一任务槽，调用独立克隆并保存；断线也等待 SDK 释放。"""
        config = SynthesisConfig(
            model=request.model, mode="full", timeout_seconds=request.timeout_seconds
        )
        model, _, _ = self.resolve(config, debug=debug)
        if not self.settings.synthesis_models[request.model].capabilities.voice_cloning:
            raise TTSError("CAPABILITY_UNSUPPORTED", "clone_voice", "所选模型不支持声音克隆")
        self.gate.claim("voice-clone")
        path = None
        work = None
        started = monotonic()
        try:
            path = self.uploads.take(request.reference_audio_token)
            destination = self.voices.prepare_directory(request.save_dir)
            await self.prepare(config, debug=debug)
            work = asyncio.create_task(
                asyncio.to_thread(
                    self.synth.clone_voice, path, reference_text=request.reference_text
                )
            )
            voice = await asyncio.shield(work)
            record = self.voices.save(
                voice, name=request.name, model_key=request.model, model=model, save_dir=destination
            )
            return {
                "voice": record,
                "elapsed_seconds": monotonic() - started,
                "effective_config": model_summary(request.model, model, synthesis=True),
            }
        finally:
            with CancelScope(shield=True):
                if work is not None:
                    await asyncio.gather(work, return_exceptions=True)
                if path is not None:
                    path.unlink(missing_ok=True)
                await self.release()
                self.gate.owner = None

    async def release(self):
        if self.synth is None:
            return {"status": "ok"}
        # close 的 pending 表示后台仍在退出；保持引用与任务占用，直到退出完成。
        while True:
            result = await asyncio.to_thread(self.synth.close, timeout=0.25)
            if result.status != "pending":
                break
        if result.status != "ok":
            raise TTSError("CLOSE_FAILED", "close", "合成资源清理失败，需重启服务")
        self.synth = self.key = None
        return {"status": "ok"}

    async def run(self, socket, *, debug):
        """控制消息与音频读取并行等待；不取消正在执行 SDK 的线程。"""
        receive = work = None
        stream = output = path = token = None
        reference_path = None
        claimed = success = cancelled = False
        metrics = {"chunks": 0, "bytes": 0, "first_audio_seconds": None}
        started = monotonic()
        effective = None
        error = None

        async def send(event):
            # 断线或慢客户端不能无限挂住服务端音频发送。
            await asyncio.wait_for(socket.send_json(event), timeout=10)

        async def wait_work(operation):
            nonlocal work, receive, cancelled
            work = asyncio.create_task(asyncio.to_thread(operation))
            while not work.done():
                done, _ = await asyncio.wait([work, receive], return_when=asyncio.FIRST_COMPLETED)
                if receive in done:
                    message = receive.result()
                    if message["type"] == "websocket.disconnect":
                        raise WebSocketDisconnect()
                    command = SynthesisCommand.model_validate_json(message.get("text", ""))
                    if command.type == "cancel":
                        cancelled = True
                        raise TTSError("CANCELLED", "synthesize", "已取消生成")
                    raise ValueError("任务准备或完整生成期间只能取消")
            result = work.result()
            work = None
            return result

        def write_audio(data, rate):
            nonlocal output, token, path
            if metrics["bytes"] + len(data) + 44 > self.files.limit:
                raise TTSError("OUTPUT_LIMIT", "synthesize", "生成音频超过 Web 的 128 MiB 上限")
            if output is None:
                token, path = self.files.create()
                output = wave.open(str(path), "wb")
                output.setnchannels(1)
                output.setsampwidth(2)
                output.setframerate(rate)
            output.writeframesraw(data)
            metrics["bytes"] += len(data)
            metrics["chunks"] += 1

        try:
            command = SynthesisCommand.model_validate(await socket.receive_json())
            if command.type != "start" or command.config is None:
                raise ValueError("第一条消息必须是带配置的 start")
            config = command.config
            self.resolve(config, debug=debug)
            if config.mode != "incremental" and not command.text.strip():
                raise ValueError("请输入需要合成的文字")
            self.gate.claim(socket)
            claimed = True
            if config.reference_audio_token:
                reference_path = self.uploads.take(config.reference_audio_token)
            options, effective = await self.prepare(config, debug=debug)
            rate = effective["options"]["sample_rate"]
            receive = asyncio.create_task(socket.receive())
            ready = {
                "type": "ready",
                "effective_config": effective,
                "format": {"sample_rate": rate, "channels": 1, "sample_type": "int16"},
            }
            if config.mode == "full":
                await send(ready)
                result = await wait_work(
                    lambda: self.synth.synthesize(
                        command.text,
                        options=options,
                        reference_audio=reference_path,
                        reference_text=config.reference_text,
                    )
                )
                write_audio(result.data, rate)
            else:
                stream = await wait_work(
                    lambda: (
                        self.synth.open_stream(options=options)
                        if config.mode == "incremental"
                        else self.synth.stream(command.text, options=options)
                    )
                )
                await send(ready)
                finished_input = config.mode != "incremental"
                while True:
                    work = asyncio.create_task(asyncio.to_thread(stream.read, timeout=0.1))
                    while not work.done():
                        done, _ = await asyncio.wait(
                            [work, receive], return_when=asyncio.FIRST_COMPLETED
                        )
                        if receive not in done:
                            continue
                        message = receive.result()
                        if message["type"] == "websocket.disconnect":
                            raise WebSocketDisconnect()
                        update = SynthesisCommand.model_validate_json(message.get("text", ""))
                        if update.type == "cancel":
                            cancelled = True
                            raise TTSError("CANCELLED", "synthesize", "已取消生成")
                        if config.mode != "incremental" or finished_input:
                            raise ValueError("文本输入已结束；可取消生成")
                        if update.type == "text":
                            await asyncio.to_thread(stream.send_text, update.text)
                            await send({"type": "text_sent", "characters": len(update.text)})
                        elif update.type == "finish_input":
                            await asyncio.to_thread(stream.finish_input)
                            finished_input = True
                            await send({"type": "input_finished"})
                        else:
                            raise ValueError("当前任务不接受此命令")
                        receive = asyncio.create_task(socket.receive())
                    try:
                        chunk = work.result()
                    except TTSError as exc:
                        if exc.code != "READ_TIMEOUT":
                            raise
                        work = None
                        continue
                    work = None
                    if chunk is None:
                        break
                    write_audio(chunk.data, rate)
                    if metrics["first_audio_seconds"] is None:
                        metrics["first_audio_seconds"] = monotonic() - started
                    await asyncio.wait_for(socket.send_bytes(chunk.data), timeout=10)
                    if debug:
                        await send({"type": "audio", "bytes": len(chunk.data), "metrics": metrics})
            if output is None:
                raise TTSError("EMPTY_AUDIO", "synthesize", "模型未返回音频")
            metrics.update(
                duration_seconds=metrics["bytes"] / (rate * 2),
                elapsed_seconds=monotonic() - started,
            )
            success = True
        except WebSocketDisconnect:
            cancelled = True
        except Exception as exc:
            error = exc
        finally:
            with CancelScope(shield=True):
                if receive is not None:
                    receive.cancel()
                    await asyncio.gather(receive, return_exceptions=True)
                if claimed:
                    try:
                        if stream is not None and not success:
                            stream.cancel()
                        if work is not None:
                            # 未启动成功的流也可能在断线后返回；其拥有者由 synth.close 清理。
                            await asyncio.gather(work, return_exceptions=True)
                        await self.release()
                        self.gate.owner = None
                    except Exception as exc:
                        success, error = False, exc
                if output is not None:
                    output.close()
                if reference_path is not None:
                    reference_path.unlink(missing_ok=True)
                if path is not None:
                    if success:
                        self.files.keep(token, path)
                    else:
                        path.unlink(missing_ok=True)
                with contextlib.suppress(WebSocketDisconnect, RuntimeError, OSError, TimeoutError):
                    if success:
                        await send(
                            {
                                "type": "completed",
                                "audio_url": f"/api/synthesis/audio/{token}",
                                "metrics": metrics,
                            }
                        )
                    elif cancelled:
                        await send({"type": "cancelled"})
                    else:
                        await send(
                            {
                                "type": "error",
                                "code": getattr(error, "code", "INVALID_REQUEST"),
                                "message": str(error)
                                if isinstance(error, (SpeechioError, ValueError))
                                else "合成任务失败，请检查模型配置与服务连接",
                            }
                        )
                    await socket.close()


def install_synthesis_routes(app, service, same_origin):
    """挂载合成专用路由；沿用页面的同源与调试开关。"""
    from fastapi import WebSocket

    @app.get("/api/voice-directories")
    def voice_directories(path: str | None = None):
        return service.voices.directories(path)

    @app.get("/api/voices")
    async def voices(model: str):
        preset = service.settings.synthesis_models.get(model)
        if preset is None:
            raise HTTPException(400, "请选择已配置的合成模型")
        return {"voices": service.voices.list(preset.config), "save_dir": service.voices.save_dir}

    @app.post("/api/voices/clone")
    async def clone(request: CloneVoiceRequest, debug: bool = False):
        if debug and not service.settings.debug:
            raise HTTPException(403, "调试台未启用")
        try:
            return await service.clone(request, debug=debug)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from None

    @app.get("/api/voices/{voice_id}/download")
    async def download_voice(voice_id: str):
        return service.voices.download(voice_id)

    @app.post("/api/voices/import")
    async def import_voice(request: ImportVoiceRequest):
        preset = service.settings.synthesis_models.get(request.model)
        if preset is None or not preset.capabilities.voice_cloning:
            raise HTTPException(400, "请选择支持克隆音色的模型")
        service.gate.claim("voice-import")
        path = None
        try:
            path = service.uploads.take(request.file_token)
            voice = service.voices.import_file(
                path, name=request.name, model_key=request.model, model=preset.config
            )
            return {"voice": voice}
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from None
        finally:
            if path is not None:
                path.unlink(missing_ok=True)
            service.gate.owner = None

    @app.delete("/api/voices/{voice_id}")
    async def remove_voice(voice_id: str):
        if service.gate.owner is not None:
            raise HTTPException(409, "任务正在使用模型，请先结束任务")
        service.voices.remove(voice_id)
        return {"status": "ok"}

    @app.websocket("/api/synthesis")
    async def synthesis_socket(socket: WebSocket):
        debug = socket.query_params.get("debug") == "1"
        if not same_origin(socket) or (debug and not service.settings.debug):
            await socket.close(code=1008)
            return
        await socket.accept()
        await service.run(socket, debug=debug)

    @app.get("/api/synthesis/audio/{token}")
    async def audio(token: str):
        return service.files.download(token)

    if service.settings.debug:

        @app.post("/api/debug/synthesis/prepare")
        async def prepare(config: SynthesisConfig):
            try:
                service.resolve(config, debug=True)
            except ValueError as exc:
                raise HTTPException(400, str(exc)) from None
            service.gate.claim("tts-prepare")
            try:
                _, effective = await service.prepare(config, debug=True)
            except Exception:
                await service.release()
                service.gate.owner = None
                raise
            service.gate.owner = None
            return {"status": "ready", "effective_config": effective}

        @app.post("/api/debug/synthesis/release")
        async def release():
            service.gate.claim("tts-release")
            result = await service.release()
            service.gate.owner = None
            return result
