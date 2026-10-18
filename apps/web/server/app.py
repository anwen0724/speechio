"""页面、受限上传与实时连接；所有识别调用集中在应用服务中。"""

import asyncio
import contextlib
import logging
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from time import monotonic
from urllib.parse import urlsplit

import soundfile as sf
from anyio import CancelScope
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware

from speechio import ASRError
from speechio.audio.files import audio_file_chunks

from .files import Uploads
from .protocol import Command, RunConfig
from .service import SpeechService
from .settings import Settings

LOGGER = logging.getLogger(__name__)
STATIC = Path(__file__).resolve().parents[1] / "static"


def same_origin(connection):
    """限制浏览器跨站调用；命令行无 Origin 的本机调用仍然可用。"""
    origin = connection.headers.get("origin")
    if origin is None:
        return True
    parsed = urlsplit(origin)
    return parsed.scheme in ("http", "https") and parsed.netloc == connection.headers.get("host")


def failure(error, *, fatal=True):
    return {
        "type": "error",
        "code": getattr(error, "code", "INVALID_REQUEST"),
        "message": str(error),
        "fatal": fatal,
    }


def create_app(settings=None, *, engine_factory=None):
    """创建独立应用实例；注入点仅用于替换外部模型，不替换会话逻辑。"""
    settings = settings if settings is not None else Settings.load()
    service = SpeechService(settings, engine_factory)
    uploads = Uploads(settings)

    async def expire_uploads():
        while True:
            await asyncio.sleep(30)
            uploads.expire()

    @asynccontextmanager
    async def lifespan(app):
        expiry = asyncio.create_task(expire_uploads())
        try:
            yield
        finally:
            expiry.cancel()
            await asyncio.gather(expiry, return_exceptions=True)
            uploads.expire(all_files=True)
            await service.release()

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(
        TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "testserver"]
    )

    @app.middleware("http")
    async def origin_guard(request, call_next):
        if not same_origin(request):
            return JSONResponse({"detail": "只接受同源页面请求"}, status_code=403)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
            "connect-src 'self'; media-src 'self' blob:; object-src 'none'; frame-ancestors 'none'"
        )
        return response

    @app.exception_handler(ASRError)
    async def asr_error(request, error):
        return JSONResponse(failure(error), status_code=409 if error.code == "WEB_BUSY" else 400)

    @app.get("/api/config")
    async def configuration():
        model = service.runtime or service.batch
        return {**settings.public(), "model_state": model.state if model else "unprepared"}

    @app.post("/api/uploads")
    async def upload(request: Request):
        if service.owner is not None:
            raise HTTPException(409, "另一个任务正在运行")
        return await uploads.receive(request)

    @app.delete("/api/uploads/{token}")
    async def delete_upload(token: str):
        uploads.discard(token)
        return {"status": "ok"}

    @app.websocket("/api/session")
    async def session_socket(socket: WebSocket):
        if not same_origin(socket):
            await socket.close(code=1008)
            return
        await socket.accept()
        run = receive = batch = chunks = file_path = None
        claimed = normal_close = False
        debug = socket.query_params.get("debug") == "1"
        try:
            if debug and not settings.debug:
                raise ASRError("DEBUG_DISABLED", "connect", "调试台未启用")
            command = Command.model_validate(await socket.receive_json())
            if command.type != "start":
                raise ValueError("第一条消息必须是 start")
            config = command.config or RunConfig()
            service.validate(config, debug=debug)
            service.claim(socket)
            claimed = True
            if config.input_mode != "live":
                file_path = uploads.take(command.file_token)
            if config.input_mode == "file":
                batch = asyncio.create_task(service.transcribe_file(config, file_path))
            elif config.input_mode == "chunks":
                info = await asyncio.to_thread(sf.info, str(file_path))
                if config.max_audio_seconds and info.duration > config.max_audio_seconds:
                    raise ASRError("AUDIO_CAPACITY_EXCEEDED", "read_file", "音频超过配置的时长上限")
                chunks = audio_file_chunks(
                    file_path, chunk_frames=max(1, int(info.samplerate * config.chunk_ms / 1000))
                )
                first = await asyncio.to_thread(next, chunks)
                audio_format, audio = first
                config = config.model_copy(
                    update={
                        "sample_rate": audio_format.sample_rate,
                        "channels": audio_format.channels,
                    }
                )
                run = await service.open_run(config, audio_format)
                await asyncio.to_thread(run.feed, audio)
            else:
                run = await service.open_run(config)
            await socket.send_json(
                {"type": "ready", "effective_config": service.effective_config(config)}
            )
            receive = asyncio.create_task(socket.receive())
            next_chunk = monotonic() + config.chunk_ms / 1000
            next_metrics = 0
            while True:
                done, _ = await asyncio.wait([receive], timeout=0.02)
                if done:
                    message = receive.result()
                    if message["type"] == "websocket.disconnect":
                        break
                    if message.get("bytes") is not None:
                        if config.input_mode != "live":
                            raise ValueError("只有实时输入接受浏览器音频")
                        await asyncio.to_thread(run.feed, message["bytes"])
                    else:
                        command = Command.model_validate_json(message["text"])
                        if command.type == "close":
                            normal_close = run is not None and run.finished
                            if config.input_mode == "file":
                                normal_close = batch is None
                            break
                        try:
                            if command.type in (
                                "start",
                                "cancel_turn",
                                "reset",
                                "candidate",
                                "correct",
                            ):
                                if command.type == "start" or not debug:
                                    raise ValueError("当前连接不支持此操作")
                            if run is None:
                                raise ValueError("文件整段识别运行中；可关闭会话，等待本次计算释放")
                            await asyncio.to_thread(run.command, command.type)
                            if command.type == "end_input" and chunks is not None:
                                chunks.close()
                                chunks = None
                            await socket.send_json({"type": "ack", "command": command.type})
                        except (ASRError, ValueError) as error:
                            if command.type == "end_input":
                                raise
                            await socket.send_json(failure(error, fatal=False))
                    receive = asyncio.create_task(socket.receive())
                if chunks is not None and not run.ended and monotonic() >= next_chunk:
                    block = await asyncio.to_thread(next, chunks, None)
                    if block is None:
                        chunks.close()
                        chunks = None
                        await asyncio.to_thread(run.command, "end_input")
                        file_path.unlink(missing_ok=True)
                        file_path = None
                    else:
                        await asyncio.to_thread(run.feed, block[1])
                    next_chunk = monotonic() + (
                        config.chunk_ms / 1000 if config.pace == "realtime" else 0
                    )
                if batch is not None and batch.done():
                    result = batch.result()
                    batch = None
                    file_path.unlink(missing_ok=True)
                    file_path = None
                    await socket.send_json(
                        {"type": "event", "kind": "final", "result": asdict(result)}
                    )
                    await socket.send_json({"type": "finished"})
                if run is not None:
                    for event in await asyncio.to_thread(run.poll):
                        await socket.send_json(event)
                    if monotonic() >= next_metrics:
                        metrics = run.metrics()
                        await socket.send_json(
                            metrics
                            if debug
                            else {
                                "type": "input_state",
                                "wake": metrics["wake"],
                                "input_state": metrics["input_state"],
                            }
                        )
                        next_metrics = monotonic() + 0.3
        except WebSocketDisconnect:
            pass
        except Exception as error:
            LOGGER.exception("Web 任务失败")
            with contextlib.suppress(WebSocketDisconnect, RuntimeError, OSError):
                await socket.send_json(failure(error))
        finally:
            # 即使连接任务被取消，也完成本任务的资源清理。
            with CancelScope(shield=True):
                # 不取消正在使用音频文件的模型线程；它真正返回后才能删除文件和释放模型。
                if receive is not None:
                    receive.cancel()
                    await asyncio.gather(receive, return_exceptions=True)
                if batch is not None:
                    await asyncio.gather(batch, return_exceptions=True)
                # 文件清理失败不能阻止会话和模型清理。
                cleanup = []
                if chunks is not None:
                    cleanup.append(chunks.close)
                if file_path is not None:
                    cleanup.append(lambda: file_path.unlink(missing_ok=True))
                for close_resource in cleanup:
                    try:
                        close_resource()
                    except Exception as error:
                        normal_close = False
                        LOGGER.exception("音频文件清理失败")
                        with contextlib.suppress(WebSocketDisconnect, RuntimeError, OSError):
                            await socket.send_json(failure(error))
                try:
                    if run is not None:
                        result = await asyncio.to_thread(run.close)
                        if result.pending_tasks:
                            normal_close = False
                except Exception as error:
                    normal_close = False
                    LOGGER.exception("会话关闭失败")
                    with contextlib.suppress(WebSocketDisconnect, RuntimeError, OSError):
                        await socket.send_json(failure(error))
                finally:
                    try:
                        if claimed and not normal_close:
                            await service.release()
                    except Exception as error:
                        LOGGER.exception("资源清理未完成")
                        with contextlib.suppress(WebSocketDisconnect, RuntimeError, OSError):
                            await socket.send_json(failure(error))
                    finally:
                        if claimed:
                            service.owner = None
                        with contextlib.suppress(WebSocketDisconnect, RuntimeError, OSError):
                            await socket.close()

    if settings.debug:

        @app.post("/api/debug/prepare")
        async def prepare(config: RunConfig):
            service.validate(config, debug=True)
            service.claim("prepare")
            try:
                await service.prepare(config)
                if service.batch is not None:
                    await asyncio.to_thread(service.batch.prepare)
                return {"status": "ready", "effective_config": service.effective_config(config)}
            finally:
                service.owner = None

        @app.post("/api/debug/release")
        async def release():
            service.claim("release")
            try:
                return await service.release()
            finally:
                service.owner = None

        @app.get("/debug")
        @app.get("/debug/")
        async def debug_page():
            return FileResponse(STATIC / "debug/index.html")

    @app.get("/")
    async def index():
        return FileResponse(STATIC / "index.html")

    # 调试页单独挂载，关闭调试时不能从静态路径绕过开关。
    for directory in ("shared", "example", *(["debug"] if settings.debug else [])):
        if (STATIC / directory).exists():
            app.mount(f"/static/{directory}", StaticFiles(directory=STATIC / directory))
    return app
