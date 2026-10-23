"""同步整段识别运行时，拥有模型资源和每次调用的输入生命周期。"""

import logging
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
from threading import Lock

import numpy as np

from speechio.adapters.contracts import validate_engine
from speechio.audio.files import audio_file_chunks, read_audio_file
from speechio.audio.formats import AudioFormat
from speechio.errors import STTError
from speechio.models.config import LocalModelConfig, RemoteModelConfig
from speechio.models.factory import build_engine
from speechio.resources import CloseError, CloseResult
from speechio.stt.input import AudioConverter, validate_max_audio_seconds
from speechio.stt.options import resolve_options, validate_task_timeout
from speechio.stt.results import Transcript

LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler())


class Transcriber:
    """通过小接口隐藏模型准备、输入转换和资源管理，不持有设备或服务。"""

    def __init__(self, config: LocalModelConfig | RemoteModelConfig, *, max_audio_seconds=None):
        """通过模型配置创建内置适配器；首次调用时准备模型。"""
        self._initialize(build_engine(config), config, max_audio_seconds)

    @classmethod
    def from_adapter(cls, adapter, *, max_audio_seconds=None):
        """接管调用方构造的适配器，不需要模型配置。"""
        instance = cls.__new__(cls)
        instance._initialize(adapter, None, max_audio_seconds)
        return instance

    def _initialize(self, adapter, config, max_audio_seconds):
        validate_max_audio_seconds(max_audio_seconds)
        self._capabilities = validate_engine(adapter)
        self._max_audio_seconds = max_audio_seconds
        self._engine = adapter
        self._config = config
        self._state = "unprepared"
        self._close_result = None
        self._operation_lock = Lock()

    def __enter__(self):
        """进入时准备引擎；内置模型失败可重试，注入实例失败后关闭。"""
        self.prepare()
        return self

    def __exit__(self, exc_type, exc, tb):
        """关闭自有资源，正文异常优先于关闭异常。"""
        result = self.close()
        if exc_type is None and result.status == "failed":
            raise STTError("CLEANUP_FAILED", "close", result.errors[0].message)
        return False

    @property
    def config(self) -> LocalModelConfig | RemoteModelConfig | None:
        return self._config

    @property
    def state(self) -> str:
        return self._state

    @property
    def close_result(self) -> CloseResult | None:
        return self._close_result

    @property
    def capabilities(self):
        """准备前可查询的实际引擎能力，不触发模型或网络调用。"""
        return self._capabilities

    @property
    def _adapter(self):
        """供角色组装识别同一实例；调用仍应经由本拥有者。"""
        return self._engine

    def prepare(self) -> None:
        """显式准备所选模型或远程客户端；输入调用也可延迟至转换完成后准备。"""
        with self._operation():
            self._prepare()

    @contextmanager
    def _operation(self, *, closing=False):
        """忙时立即拒绝，不排队；守卫在任何输入或资源副作用之前执行。"""
        if not self._operation_lock.acquire(blocking=False):
            raise STTError("RUNTIME_BUSY", "close" if closing else "transcribe", "运行时正在使用")
        try:
            if self.state == "closed" and not closing:
                raise STTError("RUNTIME_CLOSED", "transcribe", "运行时已经关闭")
            yield
        except STTError as exc:
            LOGGER.error(
                "ASR 操作失败：code=%s operation=%s %s", exc.code, exc.operation, exc.message
            )
            raise
        finally:
            self._operation_lock.release()

    def _prepare(self):
        if self.state == "ready":
            return
        try:
            self._engine.prepare()
            self._state = "ready"
        except BaseException as exc:
            cleanup_errors = self._release_resources()
            if cleanup_errors:
                exc.cleanup_errors = cleanup_errors
            if self.config is not None:
                self._engine = build_engine(self.config)
            else:
                # 已关闭的注入实例不能凭空重建，失败仍保留原始原因。
                self._state = "closed"
                self._close_result = CloseResult(
                    status="failed" if cleanup_errors else "ok", errors=cleanup_errors
                )
            if isinstance(exc, (STTError, KeyboardInterrupt, SystemExit)):
                raise
            raise STTError("MODEL_LOAD", "load_model", str(exc)) from exc

    def transcribe(self, audio, *, format: AudioFormat, options=None):
        """识别一段完整字节或数组。"""
        selected = resolve_options(options)
        language, word_timestamps = selected.language, selected.word_timestamps
        with self._operation():
            self._validate_options(language, word_timestamps)
            return self._transcribe_chunks((audio,), format, language, word_timestamps)

    def transcribe_chunks(self, chunks, *, format: AudioFormat, options=None):
        """有限块序列结束后返回一个完整结果。"""
        selected = resolve_options(options)
        language, word_timestamps = selected.language, selected.word_timestamps
        with self._operation():
            self._validate_options(language, word_timestamps)
            return self._transcribe_chunks(chunks, format, language, word_timestamps)

    def _transcribe_chunks(self, chunks, format, language, word_timestamps):
        """输入迭代失败与转换失败分别保留；全部转换完成后才准备模型。"""
        converter = AudioConverter(format, max_audio_seconds=self._max_audio_seconds)
        output = bytearray()
        try:
            try:
                iterator = iter(chunks)
            except Exception as exc:
                raise STTError("AUDIO_READ_FAILED", "read_chunks", str(exc)) from exc
            while True:
                try:
                    chunk = next(iterator)
                except StopIteration:
                    break
                except Exception as exc:
                    raise STTError("AUDIO_READ_FAILED", "read_chunks", str(exc)) from exc
                for frame in converter.feed(chunk):
                    output.extend(frame.pcm)
            for frame in converter.finish():
                output.extend(frame.pcm)
            return self._infer(bytes(output), language, word_timestamps=word_timestamps)
        finally:
            converter.close()

    def transcribe_file(self, path, *, options=None, task_timeout_seconds=3600.0):
        """按能力直接提交原始文件；其余引擎沿用标准音频解码。"""
        selected = resolve_options(options)
        language, word_timestamps = selected.language, selected.word_timestamps
        validate_task_timeout(task_timeout_seconds)
        with self._operation():
            if self.capabilities.file_transcription:
                self._validate_options(language, word_timestamps, capability="file_transcription")
                if self._max_audio_seconds is not None:
                    # 有界解码检查真实时长，不把完整长音频保存在内存中。
                    for _ in audio_file_chunks(path, max_audio_seconds=self._max_audio_seconds):
                        pass
                return self._infer_source(
                    "transcribe_file", path, language, word_timestamps, task_timeout_seconds
                )
            self._validate_options(language, word_timestamps)
            return self._infer(
                read_audio_file(path, max_audio_seconds=self._max_audio_seconds),
                language,
                word_timestamps=word_timestamps,
            )

    def transcribe_url(self, url, *, options=None, task_timeout_seconds=3600.0):
        """按声明能力识别文件地址；不能对未下载的音频承诺本地时长检查。"""
        selected = resolve_options(options)
        language, word_timestamps = selected.language, selected.word_timestamps
        validate_task_timeout(task_timeout_seconds)
        with self._operation():
            self._validate_options(language, word_timestamps, capability="url_transcription")
            if self._max_audio_seconds is not None:
                raise STTError(
                    "INVALID_CONFIGURATION",
                    "transcribe_url",
                    "URL 输入不支持本地 max_audio_seconds 检查；请使用本地文件或移除此限制",
                )
            return self._infer_source(
                "transcribe_url", url, language, word_timestamps, task_timeout_seconds
            )

    def _validate_options(self, language, word_timestamps, *, capability="transcription"):
        """调用边界先检查元数据，错误选项不能消费源或加载模型。"""
        if language is not None and (not isinstance(language, str) or not language.strip()):
            raise STTError("INVALID_CONFIGURATION", "transcribe", "语言应为非空代码或 None")
        if not getattr(self.capabilities, capability):
            raise STTError("BACKEND_INCOMPATIBLE", "transcribe", "引擎不支持该输入方式")
        if type(word_timestamps) is not bool:
            raise STTError("INVALID_CONFIGURATION", "transcribe", "词时间戳开关应为布尔值")

    def _candidate(self, pcm, language, *, word_timestamps=False):
        """候选仍是整段输入，使用通用候选角色而不读取任何会话状态。"""
        with self._operation():
            self._validate_options(language, word_timestamps)
            return self._infer(pcm, language, role="candidate", word_timestamps=word_timestamps)

    def _infer_source(self, method, source, language, word_timestamps, task_timeout_seconds):
        """文件输入沿用资源、错误和结果契约，厂商传输由适配器处理。"""
        self._prepare()
        try:
            result = getattr(self._engine, method)(
                source,
                task_timeout_seconds=task_timeout_seconds,
                language=language,
                word_timestamps=word_timestamps and self.capabilities.word_timestamps,
            )
        except STTError:
            raise
        except Exception as exc:
            raise STTError("INFERENCE_FAILED", method, str(exc)) from exc
        return self._normalize_result(result, language, word_timestamps)

    def _snapshot(self, pcm, language):
        """识别固定的标准音频快照，使用实时角色参数。"""
        with self._operation():
            if not self.capabilities.snapshot:
                raise STTError("BACKEND_INCOMPATIBLE", "snapshot", "引擎不支持周期快照")
            return self._infer(pcm, language, role="snapshot")

    def _create_stream(self, language):
        """创建独立流，语言及网络协议映射由具体引擎负责。"""
        with self._operation():
            if not self.capabilities.streaming:
                raise STTError("BACKEND_INCOMPATIBLE", "create_stream", "引擎不支持原生流式")
            self._prepare()
            try:
                return self._engine.create_streaming_session(language=language)
            except STTError:
                raise
            except Exception as exc:
                raise STTError("INFERENCE_FAILED", "create_stream", str(exc)) from exc

    def _infer(self, pcm, language, *, role="final", word_timestamps=False):
        """固定标准输入经统一引擎调用，不读取模型名称或专有配置。"""
        if not pcm:
            raise STTError("INVALID_AUDIO", "transcribe", "完整输入没有标准采样")
        self._prepare()
        array = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        try:
            result = self._engine.transcribe(
                array,
                language=language,
                role=role,
                word_timestamps=word_timestamps and self.capabilities.word_timestamps,
            )
        except STTError:
            raise
        except Exception as exc:
            raise STTError("INFERENCE_FAILED", "transcribe", str(exc)) from exc
        return self._normalize_result(result, language, word_timestamps)

    def _normalize_result(self, result, language, word_timestamps):
        """所有输入方式交付隔离的统一结果，不泄露引擎持有的可变对象。"""
        if not isinstance(result, Transcript):
            raise STTError("BACKEND_INCOMPATIBLE", "transcribe", "引擎应返回 Transcript")
        unavailable = word_timestamps and (
            not self.capabilities.word_timestamps or result.words is None
        )
        return replace(
            result,
            requested_language=language,
            segments=deepcopy(result.segments),
            backend_metadata=deepcopy(result.backend_metadata),
            words=deepcopy(result.words),
            timestamp_status="unavailable" if unavailable else result.timestamp_status,
        )

    def _release_resources(self):
        """引擎关闭错误独立记录；本地或网络资源都由该引擎负责。"""
        engine, self._engine = self._engine, None
        if engine is None:
            return ()
        try:
            result = engine.close()
            return result.errors if isinstance(result, CloseResult) else ()
        except Exception as exc:
            LOGGER.error("引擎资源关闭失败：%s", exc)
            return (CloseError("CLEANUP_FAILED", "close", str(exc)),)

    def close(self) -> CloseResult:
        """保留首次关闭结果；不把引用清除当作宿主资源回到初始值。"""
        with self._operation(closing=True):
            if self.close_result is None:
                errors = self._release_resources()
                self._close_result = CloseResult(status="failed" if errors else "ok", errors=errors)
                self._state = "closed"
            return self.close_result
