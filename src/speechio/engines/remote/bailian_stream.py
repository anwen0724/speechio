"""百炼单个 WebSocket 任务；管理累计文本、停止确认和连接释放。"""

import json
import logging
from time import monotonic
from uuid import uuid4

from websockets.sync.client import reconnect

from speechio.engines.base import StreamingSession
from speechio.errors import ASRError
from speechio.results import Transcript


class _DirectConnection(reconnect):
    """只连接配置地址，不跟随重定向；仅作为上下文使用，不迭代重连。"""

    def process_redirect(self, exc):
        return exc


class BailianStream(StreamingSession):
    """每个音频单元拥有独立连接；仅在 task-finished 后交付排空结果。"""

    def __init__(self, config, language, on_close):
        from websockets.exceptions import InvalidStatus, WebSocketException

        self.config = config
        self._on_close = on_close
        self._task_id = uuid4().hex
        self._ws = None
        self._connection = None
        self._started = self._ending = self._finished = self._closed = False
        self._sentences = {}
        self._audio_samples = 0
        self._stash = None
        self._result = Transcript()
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
                raise ASRError(code, "create_stream", "百炼流式连接被服务端拒绝") from None
            if isinstance(exc, TimeoutError):
                raise ASRError("REMOTE_TIMEOUT", "create_stream", "建立百炼流式任务超时") from None
            if isinstance(exc, (OSError, WebSocketException)):
                raise ASRError("REMOTE_CONNECTION", "create_stream", "百炼流式连接失败") from None
            raise

    def _send(self, message):
        """网络错误只暴露稳定分类，不回传可能含密钥的响应或异常内容。"""
        from websockets.exceptions import WebSocketException

        try:
            self._ws.send(message)
        except (OSError, WebSocketException):
            raise ASRError("REMOTE_CONNECTION", "stream", "百炼音频流发送失败") from None

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
        from speechio.engines.remote.bailian import pcm_bytes

        if self._closed or self._ending or self._finished:
            raise ASRError("STREAM_CLOSED", "feed_audio", "百炼流已结束，不能继续输入")
        if sample_rate not in (None, 16000):
            raise ASRError("INVALID_AUDIO", "feed_audio", "百炼适配器需要 16000 Hz 标准音频")
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
            raise ASRError("REMOTE_TIMEOUT", "stream", "等待百炼任务确认超时，未自动重试") from None
        except (OSError, WebSocketException):
            raise ASRError("REMOTE_CONNECTION", "stream", "百炼流在确认结束前断开") from None
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
                raise ASRError(code, "stream", "百炼流式任务失败，未自动重试")
            if event == "task-started" and not self._started:
                self._started = True
            elif event == "result-generated" and self._started:
                sentence = message["payload"]["output"]["sentence"]
                if sentence.get("heartbeat"):
                    return True
                sentence_id = sentence["sentence_id"]
                if not isinstance(sentence_id, int) or not isinstance(sentence["text"], str):
                    raise ValueError
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
            raise ASRError("REMOTE_PROTOCOL", "stream", "百炼流式事件格式或顺序无效") from None
        return True

    def _update_result(self):
        """同句覆盖修订，句间累积；未确认的 stash 仅作为预览。"""
        from speechio.engines.remote.bailian import transcript_from_sentences

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
        """发送一次结束指令，并在固定期限内等待全部尾部结果和结束确认。"""
        if self._finished:
            return self._result
        if self._closed:
            raise ASRError("STREAM_CLOSED", "finish_stream", "百炼流已关闭，未获得结束确认")
        try:
            if not self._ending:
                self._ending = True
                self._send_command("finish-task", {"input": {}})
            deadline = monotonic() + self.config.timeout_seconds
            while not self._finished:
                remaining = deadline - monotonic()
                if remaining <= 0:
                    raise ASRError("REMOTE_TIMEOUT", "finish_stream", "等待百炼结束确认超时")
                self._receive(remaining)
            return self._result
        finally:
            self.close()

    def close(self):
        """关闭本连接并解除引擎登记；不会将未确认预览当作最终结果。"""
        if self._closed:
            return
        self._closed = True
        try:
            if self._ws is not None:
                self._connection.__exit__(None, None, None)
        finally:
            self._on_close(self)
