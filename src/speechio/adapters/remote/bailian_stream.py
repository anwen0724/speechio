"""百炼单个 WebSocket 任务；管理累计文本、停止确认和连接释放。"""

import json
import logging
import socket
from threading import Event, Timer
from time import monotonic
from uuid import uuid4

from websockets.sync.client import reconnect

from speechio._async import check_cancelled, current_cancellation
from speechio.adapters.contracts import STTStream
from speechio.errors import STTError
from speechio.stt.results import Transcript


class _DirectConnection(reconnect):
    """只连接配置地址，不跟随重定向；仅作为上下文使用，不迭代重连。"""

    def process_redirect(self, exc):
        return exc


class BailianStream(STTStream):
    """每个音频单元拥有独立连接；仅在 task-finished 后交付排空结果。"""

    def __init__(self, config, language, on_close):
        from websockets.exceptions import InvalidStatus, WebSocketException

        self.config = config
        self._on_close = on_close
        self._task_id = uuid4().hex
        self._ws = None
        self._connection = None
        self._cancelled = Event()
        self._started = self._ending = self._finished = self._closed = False
        self._sentences = {}
        self._progress_version = 0
        self._audio_samples = 0
        self._stash = None
        self._result = Transcript()
        self._unsubscribe_cancel = None
        root = (config.base_url or "https://dashscope.aliyuncs.com").rstrip("/")
        url = root.replace("https://", "wss://", 1).replace("http://", "ws://", 1)
        try:
            self._connection = _DirectConnection(
                url + "/api-ws/v1/inference",
                additional_headers={"Authorization": f"Bearer {config.api_key}"},
                open_timeout=config.timeout_seconds,
                close_timeout=min(2, config.timeout_seconds),
                # WebSocket DEBUG 日志包含完整请求头，使用独立静默传输日志。
                logger=logging.Logger(__name__, level=logging.CRITICAL),
            )
            self._ws = self._connection.__enter__()
            token = current_cancellation()
            if token is not None:
                self._unsubscribe_cancel = token._subscribe(self.cancel)
            check_cancelled()
            parameters = {"format": "pcm", "sample_rate": 16000}
            if language is not None:
                parameters["language_hints"] = [
                    {"zh-CN": "zh", "en-US": "en"}.get(language, language)
                ]
            self._send_command(
                "run-task",
                {
                    "task_group": "audio",
                    "task": "asr",
                    "function": "recognition",
                    "model": config.model,
                    "parameters": parameters,
                    "input": {},
                },
            )
            deadline = monotonic() + config.timeout_seconds
            while not self._started:
                self._receive(max(0, deadline - monotonic()))
        except BaseException as exc:
            self.close()
            if isinstance(exc, InvalidStatus):
                code = {401: "REMOTE_AUTH", 403: "REMOTE_AUTH", 429: "REMOTE_RATE_LIMIT"}.get(
                    exc.response.status_code,
                    "REMOTE_SERVICE",
                )
                raise STTError(code, "create_stream", "百炼流式连接被服务端拒绝") from None
            if isinstance(exc, TimeoutError):
                raise STTError("REMOTE_TIMEOUT", "create_stream", "建立百炼流式任务超时") from None
            if isinstance(exc, (OSError, WebSocketException)):
                raise STTError("REMOTE_CONNECTION", "create_stream", "百炼流式连接失败") from None
            raise

    def _send(self, message):
        """给阻塞发送设置截止时间；超时关闭读写通道，不重放音频。"""
        from websockets.exceptions import WebSocketException

        if self._cancelled.is_set():
            raise STTError("CANCELLED", "stream", "百炼识别已取消")
        expired = Event()

        def expire():
            expired.set()
            self._abort_transport()

        timer = Timer(self.config.timeout_seconds, expire)
        timer.daemon = True
        timer.start()
        try:
            self._ws.send(message)
            if expired.is_set() or self._cancelled.is_set():
                raise OSError
        except (OSError, WebSocketException):
            code = (
                "CANCELLED"
                if self._cancelled.is_set()
                else "REMOTE_TIMEOUT"
                if expired.is_set()
                else "REMOTE_CONNECTION"
            )
            raise STTError(code, "stream", "百炼音频流发送未完成，未自动重试") from None
        finally:
            timer.cancel()
            timer.join()

    def _abort_transport(self):
        """不等待 WebSocket 协议锁，以便解除持锁中的 sendall。"""
        if self._ws is not None:
            try:
                self._ws.socket.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            # Windows 上 shutdown 不一定立即解除另一线程的阻塞发送，
            # 同时关闭原始 socket；不能调用会等待协议锁的 WebSocket.close。
            self._ws.socket.close()

    def cancel(self):
        """只发送中断信号；执行线程退出后统一关闭连接。"""
        self._cancelled.set()
        self._abort_transport()

    def _send_command(self, action, payload):
        self._send(
            json.dumps(
                {
                    "header": {"action": action, "task_id": self._task_id, "streaming": "duplex"},
                    "payload": payload,
                }
            )
        )

    def accept_audio(self, audio, sample_rate=None):
        """只发送本次新增 PCM；拆成 100 毫秒帧不改变音频单元边界。"""
        from speechio.adapters.remote.bailian_stt import pcm_bytes

        if self._closed or self._ending or self._finished:
            raise STTError("STREAM_CLOSED", "feed_audio", "百炼流已结束，不能继续输入")
        if sample_rate not in (None, 16000):
            raise STTError("INVALID_AUDIO", "feed_audio", "百炼适配器需要 16000 Hz 标准音频")
        pcm = pcm_bytes(audio)
        for start in range(0, len(pcm), 3200):
            frame = pcm[start : start + 3200]
            self._send(frame)
            self._audio_samples += len(frame) // 2
            self.decode()

    def _receive(self, timeout, *, polling=False):
        from websockets.exceptions import WebSocketException

        try:
            raw = self._ws.recv(timeout=timeout)
        except TimeoutError:
            if polling:
                return False
            raise STTError("REMOTE_TIMEOUT", "stream", "等待百炼任务确认超时，未自动重试") from None
        except (OSError, WebSocketException):
            raise STTError("REMOTE_CONNECTION", "stream", "百炼流在确认结束前断开") from None
        try:
            message = json.loads(raw)
            header = message["header"]
            if header["task_id"] != self._task_id:
                raise ValueError
            event = header["event"]
            if event == "task-failed":
                code = {
                    "InvalidApiKey": "REMOTE_AUTH",
                    "AccessDenied": "REMOTE_AUTH",
                    "Throttling": "REMOTE_RATE_LIMIT",
                    "Throttling.RateQuota": "REMOTE_RATE_LIMIT",
                }.get(header.get("error_code"), "REMOTE_SERVICE")
                raise STTError(code, "stream", "百炼流式任务失败，未自动重试")
            if event == "task-started" and not self._started:
                self._started = True
            elif event == "result-generated" and self._started:
                sentence = message["payload"]["output"]["sentence"]
                if sentence.get("heartbeat"):
                    return True
                sentence_id = sentence["sentence_id"]
                if not isinstance(sentence_id, int) or not isinstance(sentence["text"], str):
                    raise ValueError
                # 新句、文本修订、时间戳或句子结束状态变化才算识别推进。
                if self._sentences.get(sentence_id) != sentence:
                    self._progress_version += 1
                self._sentences[sentence_id] = sentence
                self._stash = sentence.get("stash")
                self._update_result()
            elif event == "task-finished" and self._ending:
                if any(not s.get("sentence_end") for s in self._sentences.values()):
                    raise ValueError
                self._stash = None
                self._update_result()
                self._finished = True
            else:
                raise ValueError
        except (KeyError, TypeError, ValueError, AttributeError):
            raise STTError("REMOTE_PROTOCOL", "stream", "百炼流式事件格式或顺序无效") from None
        return True

    def _update_result(self):
        """同句覆盖修订，句间累积；未确认的 stash 仅作为预览。"""
        from speechio.adapters.remote.bailian_stt import transcript_from_sentences

        sentences = [self._sentences[key] for key in sorted(self._sentences)]
        if self._stash and self._stash.get("sentence_id") not in self._sentences:
            sentences.append(self._stash)
        text = "".join(sentence["text"] for sentence in sentences)
        self._result = transcript_from_sentences(
            text,
            sentences,
            word_timestamps=True,
            audio_samples=self._audio_samples,
        )

    def decode(self):
        """读取已到达的事件，不等待新音频；单轮有界以免占住运行时。"""
        if self._closed or self._finished:
            return
        for _ in range(128):
            if not self._receive(0, polling=True) or self._finished:
                break

    def get_result(self):
        return self._result

    def finish(self):
        """发送一次结束指令；持续收到识别进展时继续等待结束确认。"""
        if self._finished:
            return self._result
        if self._closed:
            raise STTError("STREAM_CLOSED", "finish_stream", "百炼流已关闭，未获得结束确认")
        try:
            if not self._ending:
                self._ending = True
                self._send_command("finish-task", {"input": {}})
            deadline = monotonic() + self.config.timeout_seconds
            while not self._finished:
                remaining = deadline - monotonic()
                if remaining <= 0:
                    raise STTError("REMOTE_TIMEOUT", "finish_stream", "等待百炼结束确认超时")
                progress = self._progress_version
                self._receive(remaining)
                # 心跳或重复事件不延长等待，避免未完成的任务永久占用连接。
                if self._progress_version != progress:
                    deadline = monotonic() + self.config.timeout_seconds
            return self._result
        finally:
            self.close()

    def close(self):
        if self._unsubscribe_cancel is not None:
            self._unsubscribe_cancel()
        """关闭本连接并解除引擎登记；不会将未确认预览当作最终结果。"""
        if self._closed:
            return
        self._closed = True
        try:
            if self._ws is not None:
                self._connection.__exit__(None, None, None)
        finally:
            self._on_close(self)
