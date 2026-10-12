"""HTTP 与 WebSocket 客户端；只承担传输，不复制识别调度。"""

import json
import ssl
from contextlib import closing
from dataclasses import asdict
from queue import Empty, Full, Queue
from threading import Event, Lock, Thread
from urllib.parse import urlsplit, urlunsplit
from uuid import uuid4

from realtime_asr import ASRError, AudioFormat
from realtime_asr.audio.files import audio_file_chunks
from realtime_asr.protocol import encode_audio_packet


def remote_error(message):
    return ASRError(message["code"], message.get("operation", "remote"), message["message"])


def read_message(text):
    """网络对象在一次边界校验后交给调用方，保留完整协议字段。"""
    value = json.loads(text)
    if not isinstance(value, dict) or value.get("protocol_version") != 1:
        raise ASRError("INVALID_RESPONSE", "receive", "服务响应协议无效")
    if value.get("type") not in ("ready", "ack", "pong", "event", "error", "closed"):
        raise ASRError("INVALID_RESPONSE", "receive", "服务响应类别无效")
    return value


class RemoteClient:
    """拥有 HTTP 连接池；每条 WebSocket 由返回的 RemoteSession 独立拥有。"""

    def __init__(self, url, *, token=None, timeout=30, verify=True):
        import httpx

        parts = urlsplit(url)
        if (
            parts.scheme not in ("http", "https")
            or not parts.netloc
            or parts.query
            or parts.fragment
        ):
            raise ASRError("INVALID_CONFIGURATION", "connect", "URL 必须为 HTTP(S) 服务地址")
        self.url = url.rstrip("/")
        self.token, self.timeout, self.verify = token, timeout, verify
        self.http = httpx.Client(
            base_url=self.url,
            headers={"Authorization": f"Bearer {token}"} if token else {},
            timeout=timeout,
            verify=verify,
            trust_env=False,
        )

    def transcribe(self, audio, *, format=AudioFormat(), language=None):
        """递交完整 raw 音频，错误响应保留服务的稳定分类。"""
        params = asdict(format)
        if language is not None:
            params["language"] = language
        response = self.http.post(
            "/api/v1/transcribe",
            params=params,
            content=audio,
            headers={"Content-Type": "application/octet-stream"},
        )
        value = response.json()
        if not isinstance(value, dict) or value.get("protocol_version") != 1:
            raise ASRError("INVALID_RESPONSE", "transcribe", "HTTP 响应协议无效")
        if response.is_error:
            raise remote_error(value)
        return value

    def transcribe_file(self, path, *, language=None):
        """文件解码沿用 audio 功能；服务端进行唯一一次格式转换。"""
        with closing(audio_file_chunks(path)) as chunks:
            format, first = next(chunks)
            dtype = "<i2" if format.sample_type == "int16" else "<f4"
            audio = first.astype(dtype, copy=False).tobytes() + b"".join(
                chunk.astype(dtype, copy=False).tobytes() for _, chunk in chunks
            )
        return self.transcribe(audio, format=format, language=language)

    def connect(self, *, mode="manual", format=AudioFormat(), realtime=True, **options):
        """鉴权并等待 open 成功，再交出自有远程会话；失败关闭已取得的 Socket。"""
        import websocket

        parts = urlsplit(self.url)
        url = urlunsplit(
            (
                "wss" if parts.scheme == "https" else "ws",
                parts.netloc,
                parts.path + "/api/v1/ws",
                "",
                "",
            )
        )
        sslopt = {"context": self.verify} if isinstance(self.verify, ssl.SSLContext) else {}
        if self.verify is False:
            sslopt = {"cert_reqs": ssl.CERT_NONE}
        socket = websocket.create_connection(
            url,
            timeout=self.timeout,
            sslopt=sslopt,
            header={"Authorization": f"Bearer {self.token}"} if self.token else {},
            suppress_origin=True,
        )
        try:
            socket.send(
                json.dumps(
                    {
                        "protocol_version": 1,
                        "type": "open",
                        "request_id": uuid4().hex,
                        "mode": mode,
                        "format": asdict(format),
                        "realtime": realtime,
                        **options,
                    }
                )
            )
            ready = read_message(socket.recv())
            if ready["type"] == "error":
                raise remote_error(ready)
            if ready["type"] != "ready":
                raise ASRError("INVALID_RESPONSE", "open", "服务没有确认会话")
            socket.settimeout(None)
            return RemoteSession(socket, ready, format, self.timeout)
        except BaseException:
            socket.close()
            raise

    def close(self):
        """HTTP 连接池关闭不改投或重连已有 WebSocket。"""
        self.http.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class RemoteSession:
    """串行输入操作与独立接收线程；等待应答时不占用发送锁。"""

    def __init__(self, socket, ready, format, timeout):
        self._socket, self._format, self._timeout = socket, format, timeout
        self._session_id, self._generation = ready["session_id"], ready["generation"]
        self._mode = ready["mode"]
        self._unit = None
        self._sequence = 0
        self._tail = b""
        self._error = None
        self._messages = Queue(maxsize=256)
        self._messages.put_nowait(ready)
        self._responses = {}
        self._response_lock = Lock()
        self._send_lock = Lock()
        self._operation_lock = Lock()
        self._closed = Event()
        self._stopping = Event()
        self._reader = Thread(target=self._receive, name="ASRRemoteReceive", daemon=True)
        self._reader.start()

    @property
    def session_id(self):
        """服务分配的连接身份，只读；不能绑定其他会话。"""
        return self._session_id

    @property
    def generation(self):
        """最近成功 reset 的代次，只由控制应答更新。"""
        return self._generation

    def _receive(self):
        try:
            while not self._stopping.is_set():
                message = read_message(self._socket.recv())
                self._messages.put_nowait(message)
                with self._response_lock:
                    pending = self._responses.get(message.get("request_id"))
                    if pending is not None:
                        pending[1].append(message)
                        pending[0].set()
                if message["type"] == "closed":
                    return
                if message["type"] == "error" and not message.get("request_id"):
                    raise remote_error(message)
        except Full:
            self._error = ASRError("CLIENT_BACKPRESSURE", "receive", "客户端消息未被及时消费")
        except Exception as exc:
            if not self._stopping.is_set():
                self._error = (
                    exc
                    if isinstance(exc, ASRError)
                    else ASRError("CONNECTION_LOST", "receive", str(exc))
                )
        finally:
            self._closed.set()
            with self._response_lock:
                for event, _ in self._responses.values():
                    event.set()
            if self._error is not None:
                self._socket.abort()
                self._socket.close()

    def check(self):
        """供空闲 stdin 或采集循环及时发现接收端失败。"""
        if self._error is not None:
            raise self._error

    def _command(self, kind, **fields):
        """登记应答交接，短锁发送后等待；接收线程始终可以推进。"""
        self.check()
        if self._closed.is_set():
            raise ASRError("CONNECTION_CLOSED", kind, "远程会话已经关闭")
        request_id = uuid4().hex
        event, response = Event(), []
        with self._response_lock:
            self._responses[request_id] = event, response
        try:
            message = {
                "protocol_version": 1,
                "type": kind,
                "request_id": request_id,
                "session_id": self._session_id,
                "generation": self._generation,
                **fields,
            }
            with self._send_lock:
                self._socket.send(json.dumps(message))
            if not event.wait(self._timeout):
                raise ASRError("CONTROL_TIMEOUT", kind, "等待控制应答超时")
            self.check()
            if not response:
                raise ASRError("CONNECTION_LOST", kind, "应答前连接已经关闭")
            answer = response[0]
            if answer["type"] == "error":
                raise remote_error(answer)
            return answer
        finally:
            with self._response_lock:
                self._responses.pop(request_id)

    def start(self):
        """等待手动单元归属后开放音频输入，不等待上一单元的最终推理。"""
        with self._operation_lock:
            answer = self._command("start")
            self._unit, self._sequence = dict(answer["unit"]), 0
            return answer

    def feed_audio(self, audio):
        """合并跨调用的源帧尾部；只计数本次真正发送的源帧。"""
        with self._operation_lock:
            self.check()
            frame_bytes = self._format.channels * (2 if self._format.sample_type == "int16" else 4)
            joined = self._tail + audio
            size = len(joined) // frame_bytes * frame_bytes
            self._tail = joined[size:]
            block_bytes = 32768 // frame_bytes * frame_bytes
            for offset in range(0, size, block_bytes):
                block = joined[offset : min(offset + block_bytes, size)]
                metadata = {
                    "protocol_version": 1,
                    "session_id": self._session_id,
                    "generation": self._generation,
                    "unit_id": self._unit["unit_id"] if self._unit else None,
                    "audio_sequence": self._sequence,
                    "format": asdict(self._format),
                    "frame_count": len(block) // frame_bytes,
                }
                with self._send_lock:
                    self._socket.send_binary(encode_audio_packet(metadata, block))
                self._sequence += 1

    def _check_tail(self):
        if self._tail:
            raise ASRError("INVALID_AUDIO", "finish", "输入结束于不完整源帧")

    def finish(self):
        """封口当前单元或自动段；应答仅表示提交，最终仍通过 get_message 读取。"""
        with self._operation_lock:
            self._check_tail()
            answer = self._command("finish")
            self._unit = None
            return answer

    def end_input(self):
        """结束自动持续源并刷新短尾，继续输入前须 reset。"""
        with self._operation_lock:
            self._check_tail()
            return self._command("end_input")

    def cancel(self, unit=None):
        """取消本连接目标；自动模式保留持续源，手动模式丢弃本单元余量。"""
        with self._operation_lock:
            answer = self._command("cancel", **({"unit": unit} if unit else {}))
            if answer["cancelled"] and (unit is None or unit == self._unit):
                self._unit = None
                # 自动取消只删除录音段，不中断持续源的字节帧对齐。
                if self._mode == "manual":
                    self._tail = b""
            return answer

    def reset(self):
        """等待新代次应答，再清除本地输入序号与余量，不重放旧音频。"""
        with self._operation_lock:
            answer = self._command("reset")
            self._generation, self._sequence = answer["generation"], 0
            self._unit, self._tail = None, b""
            return answer

    def ping(self):
        """检查当前连接响应，不改变音频或模型状态。"""
        with self._operation_lock:
            return self._command("ping")

    def get_message(self, timeout=None):
        """返回包括应答与终态的完整消息；空等待为 None，断开明确失败。"""
        try:
            return self._messages.get(timeout=0 if self._closed.is_set() else timeout)
        except Empty:
            self.check()
            return None

    def close(self, *, graceful=True):
        """正常排空或立即断开；线程和 Socket 均由本连接释放。"""
        try:
            if graceful and not self._closed.is_set():
                with self._operation_lock:
                    self._check_tail()
                    self._command("close")
                if not self._closed.wait(self._timeout):
                    raise ASRError("CLOSE_TIMEOUT", "close", "远程结束未完成")
            if graceful:
                self.check()
        finally:
            self._stopping.set()
            if not graceful or self._error is not None:
                # 失败和立即断开不能等待对端握手，先唤醒可能等待 I/O 的线程。
                self._socket.abort()
            self._socket.close()
            self._reader.join(min(5, self._timeout))
            if self._reader.is_alive():
                raise ASRError("CLOSE_TIMEOUT", "close", "接收线程尚未退出")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, *args):
        self.close(graceful=exc_type is None)
