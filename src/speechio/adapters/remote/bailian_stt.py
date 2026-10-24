"""百炼短音频与流式识别；隐藏网络协议，使用统一的标准音频和结果。"""

import base64
import io
import wave
from dataclasses import replace
from math import isfinite

import numpy as np

from speechio.adapters.contracts import STTAdapter
from speechio.errors import STTError
from speechio.resources import CloseError, CloseResult
from speechio.stt.results import Transcript, WordTimestamp

HTTP_PATH = "/api/v1/services/aigc/multimodal-generation/generation"


def pcm_bytes(audio):
    """标准 float32 还原为小端 PCM16，供两种协议共用。"""
    return np.clip(np.rint(audio * 32768), -32768, 32767).astype("<i2").tobytes()


def transcript_from_sentences(text, sentences, *, word_timestamps, audio_samples, metadata=None):
    """仅在词范围完整覆盖正文时报告时间戳，不补造服务端没有提供的位置。"""
    words = []
    if word_timestamps:
        try:
            for sentence in sentences:
                for word in sentence["words"]:
                    start, end = word["begin_time"], word["end_time"]
                    if (
                        word.get("fixed") is False
                        or not isfinite(start)
                        or not isfinite(end)
                        or not 0 <= start <= end
                        or round(end * 16) > audio_samples
                        or (words and round(start * 16) < words[-1].end_sample)
                    ):
                        raise ValueError
                    words.append(
                        WordTimestamp(
                            word["text"] + word.get("punctuation", ""),
                            round(start * 16),
                            round(end * 16),
                        )
                    )
            # HTTP 最终响应有时只含最后一句的词，不能冒充全段时间戳。
            if "".join("".join(w.text.split()) for w in words) != "".join(text.split()):
                words.clear()
        except (KeyError, TypeError, ValueError):
            words.clear()
    return Transcript(
        text=text,
        backend_metadata=metadata or {},
        words=tuple(words) if words else None,
        timestamp_status=("reported" if words else "unavailable")
        if word_timestamps
        else "not_requested",
    )


class BailianEngine(STTAdapter):
    """拥有请求客户端及本引擎创建的流；模型名始终使用调用方明确选择。"""

    def __init__(self, config):
        self.config = config
        if not config.api_key or any(not 33 <= ord(char) <= 126 for char in config.api_key):
            raise STTError("INVALID_CONFIGURATION", "configure", "百炼需要有效的 api_key")
        self._http = None
        self._prepared = False
        self._streams = set()

    @property
    def capabilities(self):
        from speechio.models.factory import engine_capabilities

        return engine_capabilities(self.config)

    def prepare(self):
        """创建客户端但不提交音频；缺依赖时只提示安装，不自行安装。"""
        if self._prepared:
            return
        try:
            import httpx
            from websockets.sync.client import connect  # noqa: F401
        except ImportError:
            raise STTError(
                "DEPENDENCY_MISSING", "prepare", '运行库缺失，请运行 pip install speechio'
            ) from None
        if not self.capabilities.streaming:
            self._http = httpx.Client(timeout=self.config.timeout_seconds, follow_redirects=False)
        self._prepared = True

    def transcribe(self, audio, language=None, *, role="final", word_timestamps=False):
        """按所选模型通过流式任务或短音频 HTTP 请求提交完整输入。"""
        import httpx

        if self.capabilities.streaming:
            # 流式模型接收完整音频，不套用 HTTP 短音频模型的时长上限。
            stream = self.create_streaming_session(language)
            try:
                stream.accept_audio(audio)
                result = stream.finish()
                return (
                    result
                    if word_timestamps
                    else replace(
                        result,
                        words=None,
                        timestamp_status="not_requested",
                    )
                )
            finally:
                stream.close()
        if len(audio) > 300 * 16000:
            raise STTError("AUDIO_TOO_LONG", "transcribe", "百炼短音频最多 300 秒，不自动切分")
        pcm = pcm_bytes(audio)
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as output:
            output.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
            output.writeframes(pcm)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        if len(encoded) > 10 * 1024 * 1024:
            raise STTError(
                "AUDIO_TOO_LARGE", "transcribe", "音频 Base64 超过百炼直接提交的 10 MB 限制"
            )
        parameters = {"format": "wav", "sample_rate": "16000"}
        if language is not None:
            parameters["language_hints"] = [{"zh-CN": "zh", "en-US": "en"}.get(language, language)]
        body = {
            "model": self.config.model,
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_audio",
                                "input_audio": {"data": "data:audio/wav;base64," + encoded},
                            }
                        ],
                    }
                ]
            },
            "parameters": parameters,
        }
        root = (self.config.base_url or "https://dashscope.aliyuncs.com").rstrip("/")
        try:
            response = self._http.post(
                root + HTTP_PATH,
                json=body,
                headers={
                    "Authorization": f"Bearer {self.config.api_key}",
                    "X-DashScope-SSE": "disable",
                },
            )
        except httpx.TimeoutException:
            raise STTError("REMOTE_TIMEOUT", "transcribe", "等待百炼响应超时，未自动重试") from None
        except httpx.HTTPError:
            raise STTError(
                "REMOTE_CONNECTION", "transcribe", "百炼请求连接失败，未自动重试"
            ) from None
        if response.status_code != 200:
            code = {401: "REMOTE_AUTH", 403: "REMOTE_AUTH", 429: "REMOTE_RATE_LIMIT"}.get(
                response.status_code, "REMOTE_SERVICE"
            )
            raise STTError(code, "transcribe", f"百炼请求失败，HTTP {response.status_code}")
        try:
            data = response.json()
            text = data["output"]["text"]
            if not isinstance(text, str):
                raise ValueError
        except (ValueError, KeyError, TypeError):
            raise STTError("REMOTE_PROTOCOL", "transcribe", "百炼响应缺少有效的识别文本") from None
        output = data["output"]
        sentences = output.get("sentences") or (
            [output["sentence"]] if output.get("sentence") else []
        )
        # 不保留原始响应、鉴权头或服务错误文本。
        return transcript_from_sentences(
            text,
            sentences,
            word_timestamps=word_timestamps,
            audio_samples=len(audio),
            metadata={"request_id": data.get("request_id")},
        )

    def create_streaming_session(self, language=None):
        """创建独立的远程任务；引擎负责兜住调用方遗留的连接。"""
        if not self.capabilities.streaming:
            return super().create_streaming_session(language)
        self.prepare()
        from speechio.adapters.remote.bailian_stream import BailianStream

        stream = BailianStream(self.config, language, self._streams.discard)
        self._streams.add(stream)
        return stream

    def close(self):
        """关闭自有客户端；重复关闭不会再次释放。"""
        client, self._http = self._http, None
        resources = [*self._streams, *([client] if client is not None else [])]
        self._prepared = False
        errors = []
        for resource in resources:
            try:
                resource.close()
            except Exception:
                errors.append(CloseError("CLEANUP_FAILED", "close_remote", "远程连接关闭失败"))
        return CloseResult(status="failed" if errors else "ok", errors=tuple(errors))
