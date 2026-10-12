"""ASGI 应用入口；网络契约在这里转交共享服务。"""

import asyncio
import secrets
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError

from realtime_asr import ASRError
from realtime_asr.server.connection import Connection
from realtime_asr.server.schemas import ConfigPatch, HttpInput, Open, validation_details
from realtime_asr.server.service import Service


def create_app(config):
    """工厂不准备模型，lifespan 为本应用创建和关闭唯一运行时。"""
    service = Service(config)

    @asynccontextmanager
    async def lifespan(app):
        service.start()
        try:
            yield
        finally:
            await service.shutdown()

    app = FastAPI(lifespan=lifespan)
    app.state.service = service

    # 安装包自带构建资源；缺少构建时仍保留所有 API 和明确的 UI 状态。
    static = Path(__file__).with_name("static")
    if (static / "index.html").is_file():
        app.mount("/assets", StaticFiles(directory=static / "assets"), name="assets")

        @app.get("/audio-worklet.js")
        async def audio_worklet():
            return FileResponse(static / "audio-worklet.js", media_type="text/javascript")

        @app.get("/")
        async def browser():
            return FileResponse(static / "index.html")
    else:

        @app.get("/")
        async def browser_unavailable():
            return PlainTextResponse(
                "浏览器入口尚未构建。请在 web 执行 npm ci 和 npm run build。", status_code=503
            )

    @app.get("/api/v1/live")
    async def live():
        return {"protocol_version": 1, "type": "live", "live": True}

    @app.get("/api/v1/ready")
    async def ready(request: Request):
        if not authorized(request.headers, service.config.token):
            return error_response(
                service, "UNAUTHORIZED", "authenticate", "需要有效 Bearer Token", 401
            )
        return JSONResponse(
            service.readiness(), status_code=200 if service.phase == "ready" else 503
        )

    @app.get("/api/v1/metrics")
    async def metrics(request: Request):
        if not authorized(request.headers, service.config.token):
            return error_response(
                service, "UNAUTHORIZED", "authenticate", "需要有效 Bearer Token", 401
            )
        return await asyncio.to_thread(service.metrics)

    @app.get("/api/v1/capabilities")
    async def capabilities(request: Request):
        if not authorized(request.headers, service.config.token):
            return error_response(
                service, "UNAUTHORIZED", "authenticate", "需要有效 Bearer Token", 401
            )
        return service.capabilities()

    @app.get("/api/v1/config")
    async def configuration(request: Request):
        if not authorized(request.headers, service.config.token):
            return error_response(
                service, "UNAUTHORIZED", "authenticate", "需要有效 Bearer Token", 401
            )
        return service.public_config()

    @app.patch("/api/v1/config")
    async def update_configuration(request: Request):
        if not authorized(request.headers, service.config.token):
            return error_response(
                service, "UNAUTHORIZED", "authenticate", "需要有效 Bearer Token", 401
            )
        try:
            # 管理消息也是外部请求，不使用任意大小的 request.json()。
            body = bytearray()
            async for chunk in request.stream():
                if len(body) + len(chunk) > service.config.max_metadata_bytes:
                    return error_response(
                        service, "HTTP_BODY_TOO_LARGE", "config", "配置消息过大", 413
                    )
                body.extend(chunk)
            patch = ConfigPatch.model_validate_json(body).model_dump(exclude_unset=True)
            return service.update_config(patch)
        except ValidationError as exc:
            return error_response(
                service,
                "INVALID_MESSAGE",
                "config",
                "配置字段无效",
                422,
                details=validation_details(exc),
            )
        except ASRError as exc:
            return error_response(service, exc.code, exc.operation, exc.message, 422)

    @app.post("/api/v1/transcribe")
    async def transcribe(request: Request):
        if not authorized(request.headers, service.config.token):
            return error_response(
                service, "UNAUTHORIZED", "authenticate", "需要有效 Bearer Token", 401
            )
        if request.headers.get("content-type", "").split(";")[0] != "application/octet-stream":
            return error_response(
                service, "UNSUPPORTED_CONTENT_TYPE", "upload", "只接收原始二进制音频", 415
            )
        try:
            query = HttpInput.model_validate(dict(request.query_params), strict=False)
            options = Open(
                protocol_version=1,
                type="open",
                request_id="http",
                realtime=False,
                format={
                    "sample_rate": query.sample_rate,
                    "channels": query.channels,
                    "sample_type": query.sample_type,
                },
                language=query.language,
            )
            return await service.transcribe(request, options)
        except ValidationError as exc:
            return error_response(
                service,
                "INVALID_MESSAGE",
                "transcribe",
                "请求参数无效",
                422,
                details=validation_details(exc),
            )
        except ASRError as exc:
            return error_response(
                service, exc.code, exc.operation, exc.message, http_status(exc.code)
            )

    @app.websocket("/api/v1/ws")
    async def websocket_endpoint(websocket: WebSocket):
        await websocket.accept()
        session = None
        try:
            origin = websocket.headers.get("origin")
            expected = (
                ("https" if websocket.url.scheme == "wss" else "http")
                + "://"
                + websocket.headers.get("host", "")
            )
            if (
                origin is not None
                and origin != expected
                and origin not in service.config.allowed_origins
            ):
                raise ASRError("ORIGIN_REJECTED", "authenticate", "浏览器来源未允许")
            payload = await asyncio.wait_for(websocket.receive_json(), service.config.open_timeout)
            options = Open.model_validate(payload)
            if not authorized(websocket.headers, service.config.token, options.token):
                raise ASRError("UNAUTHORIZED", "authenticate", "需要有效 Bearer Token")
            options = options.model_copy(update={"token": None})
            session = await service.open_session(options)
            connection = Connection(service, websocket, session, options)
            session = None  # 拥有权交给连接，失败建连仍由入口的 finally 回收。
            await connection.run()
        except (ValidationError, ValueError, UnicodeError):
            service.counts["errors"] += 1
            await websocket.send_json(
                {
                    "protocol_version": 1,
                    "type": "error",
                    "code": "INVALID_MESSAGE",
                    "operation": "open",
                    "message": "open 字段无效",
                }
            )
            await websocket.close(code=1008)
        except TimeoutError:
            await websocket.close(code=1008, reason="OPEN_TIMEOUT")
        except ASRError as exc:
            service.counts["errors"] += 1
            await websocket.send_json(
                {
                    "protocol_version": 1,
                    "type": "error",
                    "code": exc.code,
                    "operation": exc.operation,
                    "message": exc.message,
                }
            )
            await websocket.close(
                code=1013 if "CAPACITY" in exc.code or exc.code == "SERVICE_NOT_READY" else 1008
            )
        except WebSocketDisconnect:
            pass
        finally:
            if session is not None:
                await service.release_session(session)

    return app


def authorized(headers, token, supplied_token=None):
    """只读取明确的凭据，不从 URL 或业务会话中取得 Token。"""
    if token is None:
        return True
    header = headers.get("authorization", "")
    supplied = header[7:].strip() if header.lower().startswith("bearer ") else supplied_token
    return isinstance(supplied, str) and secrets.compare_digest(
        supplied.encode("utf-8"), token.encode("utf-8")
    )


def error_response(service, code, operation, message, status, **fields):
    service.counts["errors"] += 1
    return JSONResponse(
        {
            "protocol_version": 1,
            "type": "error",
            "code": code,
            "operation": operation,
            "message": message,
            **fields,
        },
        status_code=status,
    )


def http_status(code):
    """运行错误的稳定 HTTP 分类，不用空结果隐藏输入或容量失败。"""
    if "CAPACITY" in code:
        return 429
    if code == "SERVICE_NOT_READY":
        return 503
    if code == "HTTP_BODY_TOO_LARGE":
        return 413
    if code == "HTTP_BODY_TIMEOUT":
        return 408
    if code == "DRAIN_TIMEOUT":
        return 504
    if code == "CLIENT_DISCONNECTED":
        return 499
    if code.startswith(("INVALID", "AUDIO_", "MODE_")) or code == "REALTIME_NOT_CONFIGURED":
        return 422
    return 500
