"""同步整段识别运行时，拥有模型资源和每次调用的输入生命周期。"""

import logging
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
from threading import Event, Lock, Thread

import numpy as np

from speechio._async import AdapterCalls, cancellable, check_cancelled, run_blocking, run_operation
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
        self._calls = AdapterCalls(adapter)
        self._config = config
        self._state = "unprepared"
        self._close_result = None
        self._operation_lock = Lock()
        self._close_lock = Lock()
        self._closed = Event()
        self._cleanup = None

    def __enter__(self):
        """进入时准备引擎；内置模型失败可重试，注入实例失败后关闭。"""
        self.prepare()
        return self

    def __exit__(self, exc_type, exc, tb):
        """关闭自有资源，正文异常优先于关闭异常。"""
        result = self.close()
        if result.status != "ok":
            code = "CLOSE_TIMEOUT" if result.status == "pending" else "CLEANUP_FAILED"
            message = result.errors[0].message if result.errors else "识别器关闭尚未完成"
            if exc_type is None:
                raise STTError(code, "close", message)
            exc.add_note(f"{code}: {message}")
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

    @cancellable(STTError)
    def prepare(self, *, cancellation=None) -> None:
        """显式准备所选模型或远程客户端；输入调用也可延迟至转换完成后准备。"""
        with self._operation():
            self._prepare()

    @contextmanager
    def _operation(self, *, closing=False):
        """忙时立即拒绝，不排队；守卫在任何输入或资源副作用之前执行。"""
        if not self._operation_lock.acquire(blocking=False):
            raise STTError("RUNTIME_BUSY", "close" if closing else "transcribe", "运行时正在使用")
        try:
            if (self.state in ("closing", "closed") or self._cleanup is not None) and not closing:
                raise STTError("RUNTIME_CLOSED", "transcribe", "运行时正在关闭或已经关闭")
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
            self._calls.call("prepare")
            self._state = "ready"
        except BaseException as exc:
            cleanup = self._release_resources()
            if cleanup.errors:
                exc.cleanup_errors = cleanup.errors
            if self.config is not None and cleanup.status == "ok":
                self._engine = build_engine(self.config)
                self._calls = AdapterCalls(self._engine)
            else:
                # 清理未完成时不能覆盖旧引擎；注入实例也不能凭空重建。
                self._state = "closing" if cleanup.status == "pending" else "closed"
                self._close_result = cleanup
            if isinstance(exc, (STTError, KeyboardInterrupt, SystemExit)):
                raise
            raise STTError("MODEL_LOAD", "load_model", str(exc)) from exc

    @cancellable(STTError)
    def transcribe(self, audio, *, format: AudioFormat, options=None, cancellation=None):
        """识别一段完整字节或数组。"""
        selected = resolve_options(options)
        language, word_timestamps = selected.language, selected.word_timestamps
        with self._operation():
            self._validate_options(language, word_timestamps)
            return self._transcribe_chunks((audio,), format, language, word_timestamps)

    @cancellable(STTError)
    def transcribe_chunks(self, chunks, *, format: AudioFormat, options=None, cancellation=None):
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
                check_cancelled()
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

    @cancellable(STTError)
    def transcribe_file(
        self, path, *, options=None, task_timeout_seconds=3600.0, cancellation=None
    ):
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
                        check_cancelled()
                return self._infer_source(
                    "transcribe_file", path, language, word_timestamps, task_timeout_seconds
                )
            self._validate_options(language, word_timestamps)
            return self._infer(
                read_audio_file(path, max_audio_seconds=self._max_audio_seconds),
                language,
                word_timestamps=word_timestamps,
            )

    @cancellable(STTError)
    def transcribe_url(self, url, *, options=None, task_timeout_seconds=3600.0, cancellation=None):
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

    @cancellable(STTError)
    def _candidate(self, pcm, language, *, word_timestamps=False, cancellation=None):
        """候选仍是整段输入，使用通用候选角色而不读取任何会话状态。"""
        with self._operation():
            self._validate_options(language, word_timestamps)
            return self._infer(pcm, language, role="candidate", word_timestamps=word_timestamps)

    def _infer_source(self, method, source, language, word_timestamps, task_timeout_seconds):
        """文件输入沿用资源、错误和结果契约，厂商传输由适配器处理。"""
        self._prepare()
        try:
            result = self._calls.call(
                method,
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

    @cancellable(STTError)
    def _snapshot(self, pcm, language, *, cancellation=None):
        """识别固定的标准音频快照，使用实时角色参数。"""
        with self._operation():
            if not self.capabilities.snapshot:
                raise STTError("BACKEND_INCOMPATIBLE", "snapshot", "引擎不支持周期快照")
            return self._infer(pcm, language, role="snapshot")

    @cancellable(STTError)
    def _create_stream(self, language, *, cancellation=None):
        """创建独立流，语言及网络协议映射由具体引擎负责。"""
        with self._operation():
            if not self.capabilities.streaming:
                raise STTError("BACKEND_INCOMPATIBLE", "create_stream", "引擎不支持原生流式")
            self._prepare()
            try:
                stream = self._calls.call("create_streaming_session", language=language)
                from speechio._async import current_cancellation

                token = current_cancellation()
                if token is not None:
                    cancel = getattr(stream, "cancel", None)
                    if callable(cancel):
                        token._subscribe(cancel)
                    if token.cancel_requested:
                        stream.close()
                        check_cancelled()
                return stream
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
            result = self._calls.call(
                "transcribe",
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
        """保留完整关闭状态；pending 引擎仍归当前对象所有。"""
        engine = self._engine
        if engine is None:
            return self._close_result or CloseResult()
        try:
            result = engine.close()
            if result is None:
                result = CloseResult()
            if not isinstance(result, CloseResult) or result.status not in (
                "ok",
                "pending",
                "failed",
            ):
                raise ValueError("适配器 close 应返回 None 或有效的 CloseResult")
        except Exception as exc:
            LOGGER.error("引擎资源关闭失败：%s", exc)
            result = CloseResult(
                status="failed", errors=(CloseError("CLEANUP_FAILED", "close", str(exc)),)
            )
        errors = tuple(
            dict.fromkeys(
                (*self._close_result.errors, *result.errors)
                if self._close_result
                else result.errors
            )
        )
        if result.status == "failed" and not errors:
            errors = (CloseError("CLEANUP_FAILED", "close", "适配器报告关闭失败，未提供详情"),)
        status = "failed" if result.status == "ok" and errors else result.status
        result = replace(result, status=status, errors=errors)
        if status != "pending":
            self._engine = None
            self._calls = None
        return result

    def close(self, *, timeout=5.0) -> CloseResult:
        """有界等待；后台拥有者等执行退出后才释放模型。"""
        from speechio.stt.session.session import _validate_timeout

        _validate_timeout(timeout)
        with self._close_lock:
            if self._state == "closed" and self._close_result is not None:
                return self._close_result
            if self._cleanup is None:
                self._state = "closing"
                self._close_result = CloseResult(
                    status="pending",
                    cleanup_method="deferred",
                    errors=self._close_result.errors if self._close_result else (),
                )
                self._cleanup = Thread(
                    target=self._release_when_idle, name="speechio-stt-close", daemon=True
                )
                self._cleanup.start()
        if not self._closed.wait(timeout):
            return CloseResult(status="pending", cleanup_method="deferred")
        return self._close_result

    def _release_when_idle(self):
        with self._operation_lock:
            self._state = "closing"
            self._close_result = self._release_resources()
            while self._close_result.status == "pending":
                self._closed.wait(0.05)
                self._close_result = self._release_resources()
            self._state = "closed"
        self._closed.set()

    @property
    def busy(self):
        return self._operation_lock.locked()

    def wait_idle(self, *, timeout=5.0):
        from speechio.stt.session.session import _validate_timeout

        _validate_timeout(timeout)
        if self._state in ("closing", "closed"):
            raise STTError("RUNTIME_CLOSED", "wait_idle", "识别器正在关闭或已经关闭")
        acquired = self._operation_lock.acquire(timeout=timeout)
        if acquired:
            self._operation_lock.release()
        return acquired

    async def prepare_async(self, *, cancellation=None):
        return await run_operation(self.prepare, cancellation=cancellation)

    async def transcribe_async(self, audio, *, format, options=None, cancellation=None):
        return await run_operation(
            self.transcribe, audio, format=format, options=options, cancellation=cancellation
        )

    async def transcribe_chunks_async(self, chunks, *, format, options=None, cancellation=None):
        return await run_operation(
            self.transcribe_chunks,
            chunks,
            format=format,
            options=options,
            cancellation=cancellation,
        )

    async def transcribe_file_async(
        self, path, *, options=None, task_timeout_seconds=3600.0, cancellation=None
    ):
        return await run_operation(
            self.transcribe_file,
            path,
            options=options,
            task_timeout_seconds=task_timeout_seconds,
            cancellation=cancellation,
        )

    async def transcribe_url_async(
        self, url, *, options=None, task_timeout_seconds=3600.0, cancellation=None
    ):
        return await run_operation(
            self.transcribe_url,
            url,
            options=options,
            task_timeout_seconds=task_timeout_seconds,
            cancellation=cancellation,
        )

    async def wait_idle_async(self, *, timeout=5.0):
        return await run_blocking(self.wait_idle, timeout=timeout)

    async def close_async(self, *, timeout=5.0):
        return await run_blocking(self.close, timeout=timeout)

    async def __aenter__(self):
        try:
            await self.prepare_async()
        except BaseException:
            await self.close_async(timeout=0)
            raise
        return self

    async def __aexit__(self, exc_type, exc, tb):
        result = await self.close_async()
        if result.status != "ok":
            if exc is not None:
                exc.add_note(f"识别器关闭状态：{result.status}")
            else:
                raise STTError("CLEANUP_FAILED", "close", f"识别器关闭状态：{result.status}")
