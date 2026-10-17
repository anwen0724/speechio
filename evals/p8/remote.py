"""受控远程模型边界：证明真实网络接入，不评价厂商或语音识别质量。"""

import hashlib
import json
import socket
from contextlib import contextmanager
from threading import Event, Lock, Thread
from time import monotonic, sleep

import httpx
import numpy as np
import uvicorn
import websocket
from fastapi import FastAPI, Request, WebSocket

from apps.client import RemoteClient
from apps.server import ServerConfig
from evals.p6.network import running_server
from realtime_asr import (
    ASRError,
    ASRRuntime,
    BaseTranscriptionEngine,
    EngineCapabilities,
    RealtimeConfig,
    RecognitionResult,
    SessionConfig,
    StreamingSession,
)


def audio_record(pcm):
    return {"samples": len(pcm) // 4, "digest": hashlib.sha256(pcm).hexdigest()}


@contextmanager
def model_fixture():
    """远程端仅统计输入并生成标记文本，业务会话仍由正式核心执行。"""
    app = FastAPI()
    records = {"whole": [], "streams": [], "closed_streams": 0}

    @app.get("/stats")
    async def stats():
        return records

    @app.post("/transcribe")
    async def transcribe(request: Request):
        audio = await request.body()
        record = audio_record(audio)
        records["whole"].append(record)
        return {"text": f"受控整段 {record['samples']} 个样本", **record}

    @app.websocket("/stream")
    async def stream(connection: WebSocket):
        await connection.accept()
        await connection.receive_json()
        audio = bytearray()
        try:
            while True:
                message = await connection.receive()
                if message["type"] == "websocket.disconnect":
                    break
                if message.get("bytes") is not None:
                    audio.extend(message["bytes"])
                    await connection.send_json(
                        {"text": "受控实时观察", "final": False, **audio_record(audio)}
                    )
                elif json.loads(message["text"])["type"] == "finish":
                    record = audio_record(audio)
                    records["streams"].append(record)
                    await connection.send_json({"text": "受控实时结束", "final": True, **record})
                    await connection.close()
                    break
        finally:
            records["closed_streams"] += 1

    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(16)
        server = uvicorn.Server(uvicorn.Config(app, log_level="warning", access_log=False))
        thread = Thread(
            target=server.run, kwargs={"sockets": [listener]}, name="RemoteModelFixture"
        )
        thread.start()
        url = f"http://127.0.0.1:{listener.getsockname()[1]}"
        try:
            deadline = monotonic() + 10
            while not server.started:
                if monotonic() >= deadline:
                    raise TimeoutError("受控远程模型未启动")
                sleep(0.01)
            yield url
        finally:
            server.should_exit = True
            thread.join(10)
            if thread.is_alive():
                raise RuntimeError("受控远程模型尚未退出")


class RemoteModelStream(StreamingSession):
    """演示由适配器拥有 WebSocket 和接收线程，核心只调用流契约。"""

    def __init__(self, url, language):
        self.socket = websocket.create_connection(
            url.replace("http://", "ws://") + "/stream", timeout=5
        )
        self.socket.send(json.dumps({"language": language}))
        self.language = language
        self.lock = Lock()
        self.finished = Event()
        self.result = RecognitionResult()
        self.error = None
        self.worker = Thread(target=self._receive, name="RemoteModelReceive")
        self.worker.start()

    def _receive(self):
        try:
            while True:
                message = json.loads(self.socket.recv())
                with self.lock:
                    self.result = RecognitionResult(
                        text=message["text"],
                        reported_language=self.language,
                        language_source="hint" if self.language else "unavailable",
                        backend_metadata=audio_record_fields(message),
                    )
                if message["final"]:
                    self.finished.set()
                    break
        except Exception as exc:
            self.error = exc
            self.finished.set()

    def accept_audio(self, audio, sample_rate=None):
        self.socket.send_binary(audio.astype("<f4", copy=False).tobytes())

    def get_result(self):
        if self.error is not None:
            raise ASRError("REMOTE_READ_FAILED", "read_stream", str(self.error)) from self.error
        with self.lock:
            return self.result

    def finish(self):
        if not self.finished.is_set():
            self.socket.send(json.dumps({"type": "finish"}))
            if not self.finished.wait(5):
                raise ASRError("REMOTE_TIMEOUT", "finish_stream", "远程流未完成排空")
        return self.get_result()

    def close(self):
        self.socket.close(timeout=2)
        self.worker.join(5)
        if self.worker.is_alive():
            raise ASRError("CLOSE_TIMEOUT", "close_stream", "远程接收线程尚未退出")


def audio_record_fields(message):
    return {"samples": message["samples"], "digest": message["digest"]}


class RemoteModelEngine(BaseTranscriptionEngine):
    """只在 eval 中提供的 HTTP／WebSocket 适配器，无本地模型路径。"""

    capabilities = EngineCapabilities(engine="controlled-remote", device="remote", streaming=True)

    def __init__(self, url):
        self.url = url
        self.http = None
        self.preparations = 0
        self.closes = 0

    def prepare(self):
        self.http = httpx.Client(base_url=self.url, timeout=5, trust_env=False)
        self.http.get("/stats").raise_for_status()
        self.preparations += 1

    def transcribe(self, audio, language=None, *, role="final", word_timestamps=False):
        response = self.http.post("/transcribe", content=audio.astype("<f4", copy=False).tobytes())
        response.raise_for_status()
        message = response.json()
        return RecognitionResult(
            text=message["text"], backend_metadata=audio_record_fields(message)
        )

    def create_streaming_session(self, language=None):
        return RemoteModelStream(self.url, language)

    def close(self):
        if self.http is not None:
            self.http.close()
            self.http = None
        self.closes += 1


def run():
    """SDK 和真实 ASR 服务均调用受控远程模型，核对输入与资源。"""
    sources = (np.arange(525, dtype=np.int16), -np.arange(513, dtype=np.int16))
    expected = [audio_record((source.astype(np.float32) / 32768).tobytes()) for source in sources]
    with model_fixture() as url:
        engine = RemoteModelEngine(url)
        with ASRRuntime(
            engine=engine, realtime=RealtimeConfig(mode="native", initial_delay=0)
        ) as runtime:
            session = runtime.create_session(SessionConfig(realtime=True))
            for source in sources:
                session.start()
                session.feed_audio(source[:512])
                session.feed_audio(source[512:])
                session.finish()
            finals, finished_streams = [], []
            while len(finals) < 2:
                event = session.get_event(timeout=10)
                if event is None or event.error is not None:
                    raise AssertionError(event)
                if event.kind == "realtime" and event.stream_finished:
                    finished_streams.append(event.result.backend_metadata)
                if event.kind == "completed":
                    finals.append(event.unit.unit_id)
            session.close()
        if finished_streams != expected or (engine.preparations, engine.closes) != (1, 1):
            raise AssertionError((finished_streams, engine.preparations, engine.closes))

        service_engine = RemoteModelEngine(url)
        with running_server(ServerConfig(engine=service_engine)) as service_url:
            with RemoteClient(service_url) as client:
                whole = client.transcribe(sources[0].tobytes())
                if whole["result"]["backend_metadata"] != expected[0]:
                    raise AssertionError(whole)
                session = client.connect(mode="manual")
                session.start()
                session.feed_audio(sources[1].tobytes())
                session.finish()
                kinds = []
                while "completed" not in kinds:
                    message = session.get_message(timeout=10)
                    if message is None or message["type"] == "error":
                        raise AssertionError(message)
                    if message["type"] == "event":
                        event = message["event"]
                        kinds.append(event["kind"])
                        if (
                            event["kind"] == "final"
                            and event["result"]["backend_metadata"] != expected[1]
                        ):
                            raise AssertionError(event)
                session.close()
        with httpx.Client(base_url=url, timeout=5, trust_env=False) as http:
            records = http.get("/stats").json()
        if records["whole"] != expected + expected or records["streams"] != expected + expected[1:]:
            raise AssertionError(records)
        if records["closed_streams"] != 3 or service_engine.closes != 1:
            raise AssertionError(records)
        return {
            "scope": "受控远程协议与正式核心，不代表厂商识别质量",
            "sdk_whole_and_stream": True,
            "http_and_websocket_service_injection": True,
            "source_audio_matched": True,
            "resources_closed": True,
            "records": records,
        }
