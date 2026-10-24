"""百炼长录音文件适配：隐藏上传、异步任务查询及结果读取协议。"""

from math import isfinite
from pathlib import Path
from time import monotonic, sleep
from urllib.parse import quote, urlsplit
from uuid import uuid4

from speechio.adapters.contracts import STTAdapter
from speechio.adapters.remote.bailian_stt import transcript_from_sentences
from speechio.audio.files import audio_file_chunks, read_audio_info
from speechio.errors import STTError
from speechio.resources import CloseResult

_FILE_SUFFIXES = {
    ".aac",
    ".amr",
    ".avi",
    ".flac",
    ".flv",
    ".m4a",
    ".mkv",
    ".mov",
    ".mp3",
    ".mp4",
    ".mpeg",
    ".ogg",
    ".opus",
    ".wav",
    ".webm",
    ".wma",
    ".wmv",
}


def validate_file_url(url, *, operation="transcribe_url"):
    """检查地址形式；远程文件由百炼读取，不在本机探测或下载。"""
    try:
        parsed = urlsplit(url)
        parsed.port
        valid = (
            isinstance(url, str)
            and parsed.scheme in ("http", "https")
            and parsed.hostname
            and not parsed.username
            and not parsed.password
            and not parsed.fragment
            and not any(char.isspace() for char in url)
        )
    except (TypeError, ValueError, AttributeError):
        valid = False
    if not valid:
        raise STTError("INVALID_CONFIGURATION", operation, "文件地址必须为有效的 HTTP 或 HTTPS URL")


class BailianFileEngine(STTAdapter):
    """拥有 HTTP 客户端；一次调用提交一个任务，不自动重试计费请求。"""

    def __init__(self, config):
        self.config = config
        if not config.api_key or any(not 33 <= ord(char) <= 126 for char in config.api_key):
            raise STTError("INVALID_CONFIGURATION", "configure", "百炼需要有效的 api_key")
        self._http = None
        self._root = (config.base_url or "https://dashscope.aliyuncs.com").rstrip("/")

    @property
    def capabilities(self):
        from speechio.models.factory import engine_capabilities

        return engine_capabilities(self.config)

    def prepare(self):
        """仅创建传输客户端；不上传音频、不提交任务。"""
        if self._http is not None:
            return
        try:
            import httpx
        except ImportError:
            raise STTError(
                "DEPENDENCY_MISSING", "prepare", '运行库缺失，请运行 pip install speechio'
            ) from None
        self._http = httpx.Client(timeout=self.config.timeout_seconds, follow_redirects=False)

    def _request(self, method, url, operation, *, deadline=None, **options):
        """网络错误统一分类；上传与结果地址不继承百炼 API 鉴权头。"""
        import httpx

        if deadline is not None:
            options["timeout"] = min(self.config.timeout_seconds, self._remaining(deadline))
        try:
            response = self._http.request(method, url, **options)
        except httpx.TimeoutException:
            raise STTError("REMOTE_TIMEOUT", operation, "百炼文件请求超时，未自动重试") from None
        except httpx.HTTPError:
            raise STTError(
                "REMOTE_CONNECTION", operation, "百炼文件请求连接失败，未自动重试"
            ) from None
        if deadline is not None:
            self._remaining(deadline)
        if response.status_code != 200:
            code = {401: "REMOTE_AUTH", 403: "REMOTE_AUTH", 429: "REMOTE_RATE_LIMIT"}.get(
                response.status_code, "REMOTE_SERVICE"
            )
            raise STTError(code, operation, f"百炼文件请求失败，HTTP {response.status_code}")
        return response

    @staticmethod
    def _json(response):
        """只接受对象响应，错误信息不包含原始正文或签名链接。"""
        try:
            value = response.json()
            if not isinstance(value, dict):
                raise ValueError
            return value
        except ValueError:
            raise STTError(
                "REMOTE_PROTOCOL", "transcribe_file", "百炼文件响应不是有效对象"
            ) from None

    def _headers(self):
        return {"Authorization": f"Bearer {self.config.api_key}"}

    def transcribe_file(
        self, path, language=None, *, word_timestamps=False, task_timeout_seconds=3600.0
    ):
        """校验并分块上传原始文件；不构造整份 PCM 或 Base64 副本。"""
        path = Path(path)
        try:
            if not path.is_file():
                raise OSError
            size = path.stat().st_size
        except OSError:
            raise STTError("AUDIO_READ_FAILED", "read_file", "无法读取本地音频文件") from None
        if size > 1024**3:
            raise STTError("AUDIO_TOO_LARGE", "upload_file", "百炼临时上传最多支持 1 GB")
        if not size or path.suffix.lower() not in _FILE_SUFFIXES:
            raise STTError("INVALID_AUDIO", "read_file", "文件为空或格式不受百炼长音频模型支持")
        info = read_audio_info(path)
        duration = info.duration_seconds
        if duration is None:
            duration = 0.0
            for format, chunk in audio_file_chunks(path, max_audio_seconds=12 * 3600):
                duration += len(chunk) / format.sample_rate
        if duration <= 0:
            raise STTError("INVALID_AUDIO", "read_file", "文件没有音频采样")
        if duration > 12 * 3600:
            raise STTError("AUDIO_TOO_LONG", "transcribe_file", "百炼长音频最多支持 12 小时")
        self.prepare()
        url = self._upload(path)
        return self._transcribe(
            url,
            language,
            word_timestamps,
            temporary=True,
            task_timeout_seconds=task_timeout_seconds,
        )

    def transcribe_url(
        self, url, language=None, *, word_timestamps=False, task_timeout_seconds=3600.0
    ):
        """直接提交已有 HTTP(S) 地址，不下载或重新上传音频。"""
        validate_file_url(url)
        self.prepare()
        return self._transcribe(
            url, language, word_timestamps, task_timeout_seconds=task_timeout_seconds
        )

    def _upload(self, path):
        """使用模型绑定的临时凭证上传；凭证不写入配置、日志或结果。"""
        response = self._request(
            "GET",
            self._root + "/api/v1/uploads",
            "upload_file",
            params={"action": "getPolicy", "model": self.config.model},
            headers=self._headers(),
        )
        try:
            policy = self._json(response)["data"]
            fields = {
                "OSSAccessKeyId": policy["oss_access_key_id"],
                "Signature": policy["signature"],
                "policy": policy["policy"],
                "x-oss-object-acl": policy["x_oss_object_acl"],
                "x-oss-forbid-overwrite": policy["x_oss_forbid_overwrite"],
                "key": policy["upload_dir"] + "/" + uuid4().hex + path.suffix.lower(),
                "success_action_status": "200",
            }
            host = policy["upload_host"]
            if any(not isinstance(value, str) or not value for value in fields.values()):
                raise ValueError
            validate_file_url(host, operation="upload_file")
        except (KeyError, TypeError, ValueError, STTError):
            raise STTError("REMOTE_PROTOCOL", "upload_file", "百炼上传凭证格式无效") from None
        try:
            with path.open("rb") as source:
                self._request(
                    "POST",
                    host,
                    "upload_file",
                    data=fields,
                    files={"file": (path.name, source, "application/octet-stream")},
                )
        except OSError:
            raise STTError("AUDIO_READ_FAILED", "upload_file", "读取上传文件失败") from None
        return "oss://" + fields["key"]

    def _remaining(self, deadline):
        remaining = deadline - monotonic()
        if remaining <= 0:
            raise STTError("REMOTE_TIMEOUT", "wait_task", "等待百炼文件任务超时，云端任务未取消")
        return remaining

    def _transcribe(
        self, url, language, word_timestamps, *, temporary=False, task_timeout_seconds=3600.0
    ):
        """提交一次任务，在总等待时限内查询并读取结果。"""
        headers = {**self._headers(), "X-DashScope-Async": "enable"}
        if temporary:
            headers["X-DashScope-OssResourceResolve"] = "enable"
        parameters = {"channel_id": [0]}
        if language is not None:
            parameters["language_hints"] = [{"zh-CN": "zh", "en-US": "en"}.get(language, language)]
        response = self._request(
            "POST",
            self._root + "/api/v1/services/audio/asr/transcription",
            "submit_task",
            headers=headers,
            json={
                "model": self.config.model,
                "input": {"file_urls": [url]},
                "parameters": parameters,
            },
        )
        try:
            output = self._json(response)["output"]
            task_id = output["task_id"]
            if not isinstance(task_id, str) or not task_id:
                raise ValueError
        except (KeyError, TypeError, ValueError):
            raise STTError("REMOTE_PROTOCOL", "submit_task", "百炼响应缺少任务编号") from None
        deadline = monotonic() + task_timeout_seconds
        while True:
            response = self._request(
                "GET",
                self._root + "/api/v1/tasks/" + quote(task_id, safe=""),
                "wait_task",
                headers=self._headers(),
                deadline=deadline,
            )
            try:
                output = self._json(response)["output"]
                if output["task_id"] != task_id:
                    raise ValueError
                status = output["task_status"]
                if status in ("FAILED", "CANCELED"):
                    raise STTError("REMOTE_SERVICE", "wait_task", "百炼文件任务失败或已取消")
                if status == "SUCCEEDED":
                    results = output["results"]
                    if not isinstance(results, list) or len(results) != 1:
                        raise ValueError
                    if results[0]["subtask_status"] != "SUCCEEDED":
                        raise STTError("REMOTE_SERVICE", "wait_task", "百炼文件识别子任务失败")
                    result_url = results[0]["transcription_url"]
                    break
                if status not in ("PENDING", "RUNNING"):
                    raise ValueError
            except (KeyError, TypeError, ValueError):
                raise STTError("REMOTE_PROTOCOL", "wait_task", "百炼文件任务响应格式无效") from None
            sleep(min(2.0, self._remaining(deadline)))
        try:
            validate_file_url(result_url, operation="read_result")
        except STTError:
            raise STTError("REMOTE_PROTOCOL", "read_result", "百炼结果地址无效") from None
        response = self._request("GET", result_url, "read_result", deadline=deadline)
        data = self._json(response)
        try:
            transcripts = data["transcripts"]
            if not isinstance(transcripts, list) or len(transcripts) != 1:
                raise ValueError
            transcript = transcripts[0]
            if transcript["channel_id"] != 0 or not isinstance(transcript["text"], str):
                raise ValueError
            duration = data["properties"]["original_duration_in_milliseconds"]
            if isinstance(duration, bool) or not isfinite(duration) or duration < 0:
                raise ValueError
            sentences = transcript.get("sentences", [])
            if not isinstance(sentences, list):
                raise ValueError
        except (KeyError, TypeError, ValueError):
            raise STTError(
                "REMOTE_PROTOCOL", "read_result", "百炼文件结果缺少有效文本或时长"
            ) from None
        return transcript_from_sentences(
            transcript["text"],
            sentences,
            word_timestamps=word_timestamps,
            audio_samples=round(duration * 16),
            metadata={"task_id": task_id},
        )

    def close(self):
        """释放本引擎拥有的连接；重复关闭不重复释放。"""
        client, self._http = self._http, None
        if client is not None:
            client.close()
        return CloseResult(status="ok")
