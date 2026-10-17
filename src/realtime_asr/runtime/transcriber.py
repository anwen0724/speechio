"""同步整段识别运行时，拥有模型资源和每次调用的输入生命周期。"""

import logging
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
from threading import Lock

import numpy as np

from realtime_asr.audio.conversion import AudioConverter, AudioFormat
from realtime_asr.audio.files import read_audio_file
from realtime_asr.engines.base import validate_engine
from realtime_asr.engines.local import LocalEngine, ModelConfig
from realtime_asr.errors import ASRError
from realtime_asr.results import CloseError, CloseResult, Transcript

LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler())


class Transcriber:
    """通过小接口隐藏模型准备、输入转换和资源管理，不持有设备或服务。"""

    def __init__(self, config: ModelConfig | None = None, *, engine=None, model=None, **options):
        if config is not None and (engine is not None or model is not None or options):
            raise ASRError("INVALID_CONFIGURATION", "configure", "配置对象与选项应且仅应提供一个")
        if engine is not None and options:
            raise ASRError("INVALID_CONFIGURATION", "configure", "注入引擎不接受本地选项")
        if config is None and engine is None:
            config = ModelConfig(model=model or "tiny", **options)
        self._engine = LocalEngine(config) if engine is None else engine
        self._config = self._engine.config if engine is None else None
        self._capabilities = validate_engine(self._engine)
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
            raise ASRError("CLEANUP_FAILED", "close", result.errors[0].message)
        return False

    @property
    def config(self) -> ModelConfig | None:
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
    def engine(self):
        """供角色组装识别同一实例；调用仍应经由本拥有者。"""
        return self._engine

    def prepare(self) -> None:
        """显式离线准备所选模型；输入调用也可延迟至转换完成后准备。"""
        with self._operation():
            self._prepare()

    @contextmanager
    def _operation(self, *, closing=False):
        """忙时立即拒绝，不排队；守卫在任何输入或资源副作用之前执行。"""
        if not self._operation_lock.acquire(blocking=False):
            raise ASRError("RUNTIME_BUSY", "close" if closing else "transcribe", "运行时正在使用")
        try:
            if self.state == "closed" and not closing:
                raise ASRError("RUNTIME_CLOSED", "transcribe", "运行时已经关闭")
            yield
        except ASRError as exc:
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
                self._engine = LocalEngine(self.config)
            else:
                # 已关闭的注入实例不能凭空重建，失败仍保留原始原因。
                self._state = "closed"
                self._close_result = CloseResult(
                    status="failed" if cleanup_errors else "ok", errors=cleanup_errors
                )
            if isinstance(exc, (ASRError, KeyboardInterrupt, SystemExit)):
                raise
            raise ASRError("MODEL_LOAD", "load_model", str(exc)) from exc

    def transcribe(self, audio, *, format: AudioFormat, language=None, word_timestamps=False):
        """识别一段完整字节或数组。"""
        with self._operation():
            self._validate_options(language, word_timestamps)
            return self._transcribe_chunks((audio,), format, language, word_timestamps)

    def transcribe_chunks(
        self, chunks, *, format: AudioFormat, language=None, word_timestamps=False
    ):
        """有限块序列结束后返回一个完整结果。"""
        with self._operation():
            self._validate_options(language, word_timestamps)
            return self._transcribe_chunks(chunks, format, language, word_timestamps)

    def _transcribe_chunks(self, chunks, format, language, word_timestamps):
        """输入迭代失败与转换失败分别保留；全部转换完成后才准备模型。"""
        converter = AudioConverter(format)
        output = bytearray()
        try:
            try:
                iterator = iter(chunks)
            except Exception as exc:
                raise ASRError("AUDIO_READ_FAILED", "read_chunks", str(exc)) from exc
            while True:
                try:
                    chunk = next(iterator)
                except StopIteration:
                    break
                except Exception as exc:
                    raise ASRError("AUDIO_READ_FAILED", "read_chunks", str(exc)) from exc
                for frame in converter.feed(chunk):
                    output.extend(frame.pcm)
            for frame in converter.finish():
                output.extend(frame.pcm)
            return self._infer(bytes(output), language, word_timestamps=word_timestamps)
        finally:
            converter.close()

    def transcribe_file(self, path, *, language=None, word_timestamps=False):
        """从本地文件头解码，不接受外部格式覆盖。"""
        with self._operation():
            self._validate_options(language, word_timestamps)
            return self._infer(read_audio_file(path), language, word_timestamps=word_timestamps)

    def _validate_options(self, language, word_timestamps):
        """调用边界先检查元数据，错误选项不能消费源或加载模型。"""
        if language is not None and (not isinstance(language, str) or not language.strip()):
            raise ASRError("INVALID_CONFIGURATION", "transcribe", "语言应为非空代码或 None")
        if not self.capabilities.transcription:
            raise ASRError("BACKEND_INCOMPATIBLE", "transcribe", "引擎不支持整段识别")
        if type(word_timestamps) is not bool:
            raise ASRError("INVALID_CONFIGURATION", "transcribe", "词时间戳开关应为布尔值")

    def _candidate(self, pcm, language, *, word_timestamps=False):
        """候选仍是整段输入，使用通用候选角色而不读取任何会话状态。"""
        with self._operation():
            self._validate_options(language, word_timestamps)
            return self._infer(pcm, language, role="candidate", word_timestamps=word_timestamps)

    def _snapshot(self, pcm, language):
        """识别固定的标准音频快照，使用实时角色参数。"""
        with self._operation():
            if not self.capabilities.snapshot:
                raise ASRError("BACKEND_INCOMPATIBLE", "snapshot", "引擎不支持周期快照")
            return self._infer(pcm, language, role="snapshot")

    def _create_stream(self, language):
        """创建独立流，语言及网络协议映射由具体引擎负责。"""
        with self._operation():
            if not self.capabilities.streaming:
                raise ASRError("BACKEND_INCOMPATIBLE", "create_stream", "引擎不支持原生流式")
            self._prepare()
            try:
                return self._engine.create_streaming_session(language=language)
            except ASRError:
                raise
            except Exception as exc:
                raise ASRError("INFERENCE_FAILED", "create_stream", str(exc)) from exc

    def _infer(self, pcm, language, *, role="final", word_timestamps=False):
        """固定标准输入经统一引擎调用，不读取模型名称或专有配置。"""
        if not pcm:
            raise ASRError("INVALID_AUDIO", "transcribe", "完整输入没有标准采样")
        self._prepare()
        array = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        try:
            result = self._engine.transcribe(
                array,
                language=language,
                role=role,
                word_timestamps=word_timestamps and self.capabilities.word_timestamps,
            )
        except ASRError:
            raise
        except Exception as exc:
            raise ASRError("INFERENCE_FAILED", "transcribe", str(exc)) from exc
        if not isinstance(result, Transcript):
            raise ASRError("BACKEND_INCOMPATIBLE", "transcribe", "引擎应返回 Transcript")
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
