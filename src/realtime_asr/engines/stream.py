"""Nemotron 独立流，封装原生状态与关闭后缓存。"""

import re
from uuid import uuid4

from realtime_asr.audio.conversion import as_audio, validate_sample_rate
from realtime_asr.errors import ASRError, error_record
from realtime_asr.results import Transcript

LANGUAGE_TAG = re.compile(r"^\s*<(?P<language>[a-zA-Z]{2,3}(?:-[a-zA-Z]{2,4})?)>\s*")


def native_result(raw_text, language):
    """分离起始语言标签，只使用后端真实语言或明确的固定提示。"""
    text = raw_text
    match = LANGUAGE_TAG.match(text)
    detected = match.group("language") if match else None
    if match:
        text = text[match.end() :].strip()
    reported = detected or (language if text and language != "auto" else None)
    source = "detected" if detected else ("hint" if reported else "unavailable")
    return Transcript(
        text=text,
        requested_language=language,
        reported_language=reported,
        language_source=source,
        backend_metadata={"raw_text": raw_text},
    )


class NativeStream:
    """每个对象拥有一份原生流，不拥有共享识别器。"""

    def __init__(self, recognizer, language=None, use_prompt=True):
        self.recognizer = recognizer
        self.language = "auto" if language is None else language
        if not isinstance(self.language, str) or not self.language:
            raise ASRError("BACKEND_INCOMPATIBLE", "create_streaming_session", "语言值无效")
        self.state = "open"
        self.failed = False
        self.has_final_result = False
        self.accepted_samples = 0
        self.stream_id = ""
        self._native = None
        self._cache = Transcript(requested_language=self.language)
        self.cleanup = {
            "method": "reference_drop",
            "status": "ok",
            "completed_ns": None,
            "errors": [],
        }
        self.reset()

    def _require_input(self):
        """状态错误在进入原生运行库前返回，失败试次不再推进。"""
        if self.state == "closed":
            raise ASRError("STREAM_CLOSED", "accept_audio", "流已经关闭")
        if self.state == "finished":
            raise ASRError("STREAM_FINISHED", "accept_audio", "流已经结束")
        if self.failed:
            raise ASRError("INFERENCE_FAILED", "accept_audio", "失败流只允许读取缓存和关闭")

    def accept_audio(self, audio, sample_rate=None):
        """只递交新增样本；原生接受成功之后才推进确认范围。"""
        self._require_input()
        validate_sample_rate(sample_rate)
        array = as_audio(audio, allow_empty=True, allow_column=True)
        if array.size == 0:
            return
        try:
            self._native.accept_waveform(16000, array)
        except Exception as exc:
            self.failed = True
            raise ASRError("INFERENCE_FAILED", "accept_audio", str(exc)) from exc
        self.accepted_samples += array.size

    def decode(self):
        """排空当前可处理内容并更新缓存；关闭后保持缓存不变。"""
        if self.state == "closed":
            return
        if self.failed:
            raise ASRError("INFERENCE_FAILED", "decode", "失败流不能继续解码")
        try:
            while self.recognizer.is_ready(self._native):
                self.recognizer.decode_stream(self._native)
        except Exception as exc:
            self.failed = True
            raise ASRError("INFERENCE_FAILED", "decode", str(exc)) from exc
        self.get_result()

    def get_result(self):
        """读取当前结果，不追加音频，也不升级为结束结果。"""
        if self.state == "closed" or self.failed:
            return self._cache
        try:
            # 固定的 sherpa-onnx 入口返回结果对象；语言标签来自其 text。
            value = self.recognizer.get_result_all(self._native)
            self._cache = native_result(value.text, self.language)
        except Exception as exc:
            self.failed = True
            raise ASRError("INFERENCE_FAILED", "get_result", str(exc)) from exc
        return self._cache

    def finish(self):
        """显式结束并排空；重复调用不重发原生结束通知。"""
        if self.state in ("closed", "finished"):
            return self._cache
        if self.failed:
            raise ASRError("INFERENCE_FAILED", "finish", "失败流不能正常结束")
        finish = getattr(self._native, "input_finished", None)
        if not callable(finish):
            self.failed = True
            raise ASRError("BACKEND_INCOMPATIBLE", "finish", "原生流缺少 input_finished")
        try:
            finish()
        except Exception as exc:
            self.failed = True
            raise ASRError("INFERENCE_FAILED", "finish", str(exc)) from exc
        self.decode()
        self.state = "finished"
        self.has_final_result = True
        return self._cache

    def reset(self):
        """释放旧流并建立新标识，保持原语言，重置范围与缓存。"""
        if self.state == "closed":
            raise ASRError("STREAM_CLOSED", "reset", "关闭流不能重置")
        if self.failed:
            raise ASRError("INFERENCE_FAILED", "reset", "失败试次应创建新流")
        try:
            self._release()
            self._native = self.recognizer.create_stream()
            setter = getattr(self._native, "set_option", None)
            if not callable(setter):
                raise ASRError("BACKEND_INCOMPATIBLE", "reset", "原生流缺少语言选项")
            setter("language", self.language)
        except Exception as exc:
            self.failed = True
            try:
                self.close()
            except ASRError:
                pass
            if isinstance(exc, ASRError):
                raise
            raise ASRError("BACKEND_INCOMPATIBLE", "reset", str(exc)) from exc
        self.stream_id = uuid4().hex
        self.accepted_samples = 0
        self.has_final_result = False
        self.state = "open"
        self._cache = Transcript(requested_language=self.language)

    def _release(self):
        """按运行库公开能力释放；清理异常保留，引用清除不代表显存回收。"""
        native, self._native = self._native, None
        if native is None:
            return
        self.cleanup["method"] = "reference_drop"
        for name in ("close", "release"):
            operation = getattr(native, name, None)
            if callable(operation):
                self.cleanup["method"] = f"native_{name}"
                try:
                    operation()
                except Exception as exc:
                    self.cleanup["status"] = "failed"
                    error = ASRError("CLEANUP_FAILED", "close", str(exc))
                    self.cleanup["errors"].append(error_record(error, "close"))
                    raise error from exc
                break

    def close(self):
        """关闭不隐含 finish；重复关闭不覆盖首次清理结论。"""
        if self.state == "closed":
            return
        try:
            self._release()
        finally:
            self.state = "closed"
