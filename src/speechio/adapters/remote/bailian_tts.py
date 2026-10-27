"""百炼 Qwen TTS；每次任务使用一条独立连接。"""

import base64
import io
import json
import logging
import socket
import wave
from dataclasses import replace
from threading import Event, Lock, Timer
from time import monotonic, sleep
from uuid import uuid4

from speechio.adapters.contracts import TTSAdapter, TTSCapabilities
from speechio.audio.formats import AudioFormat
from speechio.audio.results import AudioChunk, AudioResult
from speechio.errors import TTSError
from speechio.resources import CloseResult
from speechio.tts.timing import TextTiming, WordTiming
from speechio.tts.voices import Voice


class BailianSynthesisEngine(TTSAdapter):
    """统一完整文本、音频流与增量文本的服务协议。"""

    capabilities = TTSCapabilities(
        default_sample_rate=24000,
        synthesis=True,
        audio_streaming=True,
        text_streaming=True,
        language_selection=True,
        speed_control=True,
        instruction_control=True,
        voice_cloning=True,
        sample_rates=(8000, 16000, 22050, 24000, 44100, 48000),
    )

    def __init__(self, config):
        self.config = config
        self.capabilities = replace(
            type(self).capabilities,
            text_timestamps=("sentence", "word")
            if config.model == "qwen-audio-3.1-tts-flash"
            else (),
        )
        self._streams = set()
        self._lock = Lock()

    def prepare(self):
        """仅检查凭据和运行库；不建立空合成任务，不读取环境变量。"""
        if not self.config.api_key:
            raise TTSError("REMOTE_AUTH", "prepare", "需要显式提供 api_key")
        try:
            import websockets.sync.client  # noqa: F401
        except ImportError:
            raise TTSError(
                "OPTIONAL_DEPENDENCY_MISSING", "prepare", "运行库缺失，请运行 pip install speechio"
            ) from None

    def open_stream(self, *, options):
        options = self._voice_options(options)
        if not 0.5 <= options.speed <= 2.0:
            raise TTSError("INVALID_INPUT", "synthesize", "百炼语速应在 0.5 到 2.0 之间")
        stream = _BailianSynthesisStream(self.config, options, self._forget)
        with self._lock:
            self._streams.add(stream)
        return stream

    def _voice_options(self, options):
        """将保存的云端音色转成服务 ID，不重新创建音色或读取其中的地址。"""
        voice = options.voice
        if voice is None:
            raise TTSError("INVALID_INPUT", "synthesize", "百炼当前合成接口需要指定音色")
        if not isinstance(voice, Voice):
            return options
        if (
            voice.adapter != "bailian"
            or voice.model != self.config.model
            or voice.data_version != 1
        ):
            raise TTSError("VOICE_INCOMPATIBLE", "synthesize", "音色与百炼模型不兼容")
        try:
            value = json.loads(voice.data)
            voice_id = value["voice_id"]
            if (
                not isinstance(voice_id, str)
                or not voice_id.strip()
                or value["base_url"] != self._root_url()
            ):
                raise ValueError
        except (ValueError, KeyError, TypeError):
            raise TTSError(
                "VOICE_INCOMPATIBLE", "synthesize", "音色数据无效或服务地址不匹配"
            ) from None
        return replace(options, voice=voice_id)

    def _root_url(self):
        return (self.config.base_url or "https://dashscope.aliyuncs.com").rstrip("/")

    def clone_voice(self, reference_audio, *, reference_text=None):
        """创建云端音色并等到可用；不执行试听合成，不保存密钥或原录音。"""
        if reference_text is not None:
            raise TTSError("CAPABILITY_UNSUPPORTED", "clone_voice", "百炼音色创建不接收参考文字")
        if (
            reference_audio.format.sample_rate < 16000
            or not 5 <= reference_audio.duration_seconds <= 60
            or len(reference_audio.data) + 44 > 10 * 1024 * 1024
        ):
            raise TTSError(
                "INVALID_INPUT",
                "clone_voice",
                "百炼参考录音须为 5～60 秒、至少 16 kHz、WAV 不超过 10 MB",
            )
        try:
            import httpx
        except ImportError:
            raise TTSError(
                "OPTIONAL_DEPENDENCY_MISSING",
                "clone_voice",
                "运行库缺失，请运行 pip install speechio",
            ) from None

        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(reference_audio.format.sample_rate)
            output.writeframes(reference_audio.data)
        audio = "data:audio/wav;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
        with httpx.Client(follow_redirects=False) as client:
            output = self._enrollment_request(
                client,
                {
                    "action": "create_voice",
                    "target_model": self.config.model,
                    "prefix": "speechio",
                    "audio": audio,
                },
                timeout=self.config.timeout_seconds,
            )
            voice_id = output.get("voice_id")
            if not isinstance(voice_id, str) or not voice_id.strip():
                raise TTSError("PROTOCOL_ERROR", "clone_voice", "百炼没有返回有效音色 ID")
            deadline = monotonic() + self.config.timeout_seconds
            try:
                while True:
                    remaining = deadline - monotonic()
                    if remaining <= 0:
                        raise TTSError("REMOTE_TIMEOUT", "clone_voice", "等待云端音色可用超时")
                    output = self._enrollment_request(
                        client, {"action": "query_voice", "voice_id": voice_id}, timeout=remaining
                    )
                    status = output.get("status")
                    if status == "OK":
                        break
                    if status != "DEPLOYING":
                        raise TTSError("CLONING_FAILED", "clone_voice", "云端音色未能创建成功")
                    sleep(min(0.5, max(0, deadline - monotonic())))
            except TTSError as exc:
                # 创建后查询失败不代表音色不存在；保留 ID 供调用方查询，禁止重建或删除。
                exc.add_note(f"云端已返回音色 ID：{voice_id}；未自动重建或删除")
                raise
        return Voice(
            adapter="bailian",
            model=self.config.model,
            data=json.dumps({"voice_id": voice_id, "base_url": self._root_url()}).encode("utf-8"),
        )

    def _enrollment_request(self, client, payload, *, timeout):
        """一次创建或查询请求；不重试、不重定向，也不回显服务原文中的敏感数据。"""
        import httpx

        try:
            response = client.post(
                self._root_url() + "/api/v1/services/audio/tts/customization",
                headers={"Authorization": f"Bearer {self.config.api_key}"},
                json={"model": "voice-enrollment", "input": payload},
                timeout=timeout,
            )
        except httpx.TimeoutException:
            raise TTSError(
                "REMOTE_TIMEOUT", "clone_voice", "百炼音色请求超时，未自动重试"
            ) from None
        except httpx.HTTPError:
            raise TTSError(
                "REMOTE_CONNECTION", "clone_voice", "百炼音色请求失败，未自动重试"
            ) from None
        if not response.is_success:
            code = {401: "REMOTE_AUTH", 403: "REMOTE_AUTH", 429: "REMOTE_RATE_LIMIT"}.get(
                response.status_code, "REMOTE_SERVICE"
            )
            raise TTSError(code, "clone_voice", "百炼拒绝音色请求，未自动重试")
        try:
            output = response.json()["output"]
            if not isinstance(output, dict):
                raise ValueError
            return output
        except (ValueError, TypeError, KeyError):
            raise TTSError("PROTOCOL_ERROR", "clone_voice", "百炼音色响应格式无效") from None

    def _forget(self, stream):
        with self._lock:
            self._streams.discard(stream)

    def stream(self, text, *, options):
        stream = self.open_stream(options=options)
        try:
            stream.send_text(text)
            stream.finish_input()
            return stream
        except BaseException:
            stream.close()
            raise

    def synthesize(self, text, *, options):
        """收集同一流任务的音频，不另发一次完整文本请求。"""
        stream = self.stream(text, options=options)
        try:
            chunks = []
            timings = []
            while (event := stream.read_event()) is not None:
                if isinstance(event, AudioChunk):
                    chunks.append(event.data)
                elif event.end_seconds is not None:
                    timings.append(event)
            return AudioResult(b"".join(chunks), stream.format, tuple(timings))
        finally:
            stream.close()

    def close(self):
        with self._lock:
            streams = tuple(self._streams)
        for stream in streams:
            stream.close()
        return CloseResult()


class _BailianSynthesisStream:
    """隐藏服务事件、PCM 帧边界和网络等待，不拥有公共任务状态。"""

    def __init__(self, config, options, on_close):
        self._config = config
        self.format = AudioFormat(sample_rate=options.sample_rate or 24000)
        self._on_close = on_close
        self._task_id = str(uuid4())
        self._ws = None
        self._connection = None
        self._send_lock = Lock()
        self._close_lock = Lock()
        self._state_lock = Lock()
        self._closed = False
        self._cancelled = Event()
        self._ending = False
        self._finished = False
        self._started = False
        self._characters = 0
        self._remainder = b""
        self._finish_deadline = None
        self._progress_index = -1
        self._progress = set()
        self._text_timestamps = options.text_timestamps
        self._audio_bytes = 0
        self._segment = None
        self._next_segment = 0
        self._awaiting_audio = False
        try:
            self._connect()
            parameters = {
                "text_type": "PlainText",
                "voice": options.voice,
                "format": "pcm",
                "sample_rate": self.format.sample_rate,
                "rate": options.speed,
            }
            if options.language is not None:
                parameters["language_hints"] = [options.language]
            if options.text_timestamps == "word":
                parameters["word_timestamp_enabled"] = True
            if options.instruction is not None:
                parameters["instruction"] = options.instruction
            self._send(
                "run-task",
                {
                    "task_group": "audio",
                    "task": "tts",
                    "function": "SpeechSynthesizer",
                    "model": config.model,
                    "parameters": parameters,
                    "input": {},
                },
            )
            deadline = monotonic() + config.timeout_seconds
            while not self._started:
                raw = self._recv(max(0, deadline - monotonic()))
                if isinstance(raw, bytes):
                    raise TTSError("PROTOCOL_ERROR", "open_stream", "启动确认前收到音频")
                self._event(raw)
        except BaseException as exc:
            try:
                self.close()
            except Exception:
                exc.add_note("启动失败后，合成连接清理未成功")
            if isinstance(exc, TimeoutError):
                raise TTSError("REMOTE_TIMEOUT", "open_stream", "等待任务启动确认超时") from None
            raise

    def _connect(self):
        from websockets.exceptions import InvalidStatus
        from websockets.sync.client import reconnect

        class DirectConnection(reconnect):
            """禁用重定向；仅作一次连接上下文，不执行自动重试。"""

            def process_redirect(self, exc):
                return exc

        root = (self._config.base_url or "https://dashscope.aliyuncs.com").rstrip("/")
        url = root.replace("https://", "wss://", 1).replace("http://", "ws://", 1)
        try:
            self._connection = DirectConnection(
                url + "/api-ws/v1/inference",
                additional_headers={"Authorization": f"Bearer {self._config.api_key}"},
                open_timeout=self._config.timeout_seconds,
                close_timeout=min(2, self._config.timeout_seconds),
                logger=logging.Logger(__name__, level=logging.CRITICAL),
                max_queue=4,
            )
            self._ws = self._connection.__enter__()
        except InvalidStatus as exc:
            code = {401: "REMOTE_AUTH", 403: "REMOTE_AUTH", 429: "REMOTE_RATE_LIMIT"}.get(
                exc.response.status_code, "REMOTE_SERVICE"
            )
            raise TTSError(code, "open_stream", "百炼拒绝建立合成连接") from None
        except TimeoutError:
            raise TTSError("REMOTE_TIMEOUT", "open_stream", "建立合成连接超时") from None
        except Exception:
            raise TTSError("REMOTE_CONNECTION", "open_stream", "无法建立合成连接") from None

    def _send(self, action, payload):
        """同步库没有发送超时参数；截止时中断 socket，解除阻塞且不重放。"""
        expired = Event()

        def abort_send():
            expired.set()
            try:
                self._ws.socket.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

        timer = Timer(self._config.timeout_seconds, abort_send)
        timer.daemon = True
        timer.start()
        try:
            self._ws.send(
                json.dumps(
                    {
                        "header": {
                            "action": action,
                            "task_id": self._task_id,
                            "streaming": "duplex",
                        },
                        "payload": payload,
                    }
                )
            )
            if expired.is_set():
                raise TimeoutError
        except Exception:
            code = "REMOTE_TIMEOUT" if expired.is_set() else "REMOTE_CONNECTION"
            raise TTSError(code, "send_text", "百炼合成消息发送失败，未自动重试") from None
        finally:
            timer.cancel()
            timer.join()

    def send_text(self, text):
        with self._send_lock:
            if self._ending or self._cancelled.is_set() or self._closed:
                raise TTSError("INPUT_CLOSED", "send_text", "百炼文本输入已结束")
            if self._characters + len(text) > 200000:
                suffix = "；此前提交的文本可能已产生费用" if self._characters else ""
                raise TTSError("INVALID_INPUT", "send_text", "百炼单任务最多 200000 字符" + suffix)
            for offset in range(0, len(text), 20000):
                part = text[offset : offset + 20000]
                self._send("continue-task", {"input": {"text": part}})
                self._characters += len(part)

    def finish_input(self):
        with self._send_lock:
            if self._ending:
                return
            if self._cancelled.is_set():
                raise TTSError("CANCELLED", "finish_input", "合成已取消")
            with self._state_lock:
                self._ending = True
                self._finish_deadline = monotonic() + self._config.timeout_seconds
            self._send("finish-task", {"input": {}})

    def _recv(self, timeout):
        try:
            return self._ws.recv(timeout=timeout)
        except TimeoutError:
            raise
        except Exception:
            raise TTSError("REMOTE_CONNECTION", "read", "百炼在完成确认前断开连接") from None

    def _mark_progress(self):
        with self._state_lock:
            if self._ending:
                self._finish_deadline = monotonic() + self._config.timeout_seconds

    def _event(self, raw):
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
                raise TTSError(code, "synthesize", "百炼合成任务失败，未自动重试")
            if event == "task-started" and not self._started:
                self._started = True
            elif event == "task-finished" and self._started and self._ending:
                if self._remainder or (self._text_timestamps and self._segment is not None):
                    raise ValueError
                self._finished = True
            elif event == "result-generated" and self._started:
                output = message["payload"]["output"]
                kind = output.get("type")
                if kind in ("sentence-begin", "sentence-synthesis", "sentence-end"):
                    index = output.get("sentence", {}).get("index")
                    if type(index) is not int:
                        raise ValueError
                    # 只保留当前句子的进展标记；重复或旧句事件不延长截止时间。
                    if index > self._progress_index:
                        self._progress_index = index
                        self._progress.clear()
                    if index == self._progress_index and kind not in self._progress:
                        self._progress.add(kind)
                        self._mark_progress()
                    if self._text_timestamps is not None:
                        return self._sentence_event(output)
                # 未知扩展和心跳不延长结束等待。
            else:
                raise ValueError
        except (ValueError, TypeError, KeyError, AttributeError):
            raise TTSError("PROTOCOL_ERROR", "read", "百炼任务事件或 PCM 尾部无效") from None

    def _sentence_event(self, output):
        """根据服务分句和实际 PCM 长度定位句子，不按发送文本片段估算。"""
        kind = output["type"]
        index = output["sentence"]["index"]
        words = self._word_timings(output["sentence"].get("words"))
        if self._awaiting_audio:
            raise ValueError
        if kind == "sentence-begin":
            text = output["original_text"]
            if (
                self._segment is not None
                or index != self._next_segment
                or not isinstance(text, str)
            ):
                raise ValueError
            self._segment = TextTiming(
                index,
                text,
                self._audio_bytes / (2 * self.format.sample_rate),
                words=words,
            )
            return self._segment
        if self._segment is None or index != self._segment.segment_index:
            raise ValueError
        if kind == "sentence-synthesis":
            self._awaiting_audio = True
            if words is not None and words != self._segment.words:
                self._segment = replace(self._segment, words=words)
                return self._segment
            return None
        if self._remainder:
            raise ValueError
        text = output.get("original_text", self._segment.text)
        if not isinstance(text, str):
            raise ValueError
        result = replace(
            self._segment,
            text=text,
            end_seconds=self._audio_bytes / (2 * self.format.sample_rate),
            words=words,
        )
        self._segment = None
        self._next_segment += 1
        return result

    def _word_timings(self, words):
        """实测 Qwen 3.1 flash 的毫秒值从整段音频开头计算，只转换单位。"""
        if self._text_timestamps != "word" or words is None or words == []:
            return None
        if not isinstance(words, list):
            raise ValueError
        result = []
        for word in words:
            text, begin, end = word["text"], word["begin_time"], word["end_time"]
            if (
                not isinstance(text, str)
                or type(begin) is not int
                or type(end) is not int
                or not 0 <= begin <= end
            ):
                raise ValueError
            result.append(WordTiming(text, begin / 1000, end / 1000))
        return tuple(result)

    def read(self, *, timeout=None):
        """旧适配器入口继续只交付音频。"""
        deadline = None if timeout is None else monotonic() + timeout
        while True:
            event = self.read_event(
                timeout=None if deadline is None else max(0, deadline - monotonic())
            )
            if event is None or isinstance(event, AudioChunk):
                return event

    def read_event(self, *, timeout=None):
        """服务元信息与紧随其后的 PCM 保持同一读取顺序。"""
        deadline = None if timeout is None else monotonic() + timeout
        while not self._finished:
            if self._cancelled.is_set():
                raise TTSError("CANCELLED", "read", "百炼合成已取消")
            with self._state_lock:
                finish_deadline = self._finish_deadline
            now = monotonic()
            if finish_deadline is not None and now >= finish_deadline:
                raise TTSError("REMOTE_TIMEOUT", "read", "结束输入后等待合成进展超时")
            if deadline is not None and now >= deadline:
                raise TTSError("READ_TIMEOUT", "read", "本次读取等待超时")
            remaining = min(
                [0.1] + [limit - now for limit in (deadline, finish_deadline) if limit is not None]
            )
            try:
                raw = self._recv(remaining)
            except TimeoutError:
                continue
            if isinstance(raw, bytes):
                if self._text_timestamps is not None:
                    if not self._awaiting_audio:
                        raise TTSError("PROTOCOL_ERROR", "read", "音频缺少对应的句子事件")
                    self._awaiting_audio = False
                if raw:
                    self._mark_progress()
                data = self._remainder + raw
                length = len(data) // 2 * 2
                self._remainder = data[length:]
                if length:
                    self._audio_bytes += length
                    return AudioChunk(data[:length], self.format)
            else:
                event = self._event(raw)
                if event is not None:
                    return event
        return None

    def cancel(self):
        if self._finished or self._closed or self._cancelled.is_set():
            return
        self._cancelled.set()
        with self._send_lock:
            if not self._closed:
                self._send("finish-task", {"input": {"directive": "cancel"}})

    def close(self, *, timeout=5.0):
        """中断协议读取并关闭连接；公共拥有者负责等待线程退出。"""
        with self._close_lock:
            if self._closed:
                return CloseResult()
            try:
                if self._ws is not None:
                    self._ws.close()
                if self._connection is not None:
                    self._connection.__exit__(None, None, None)
            finally:
                self._closed = True
                self._on_close(self)
        return CloseResult()
