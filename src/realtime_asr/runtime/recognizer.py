"""同步整段识别运行时，拥有模型资源和每次调用的输入生命周期。"""

import logging
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, replace
from os import PathLike
from pathlib import Path
from threading import Lock

import numpy as np

from realtime_asr.audio.conversion import AudioConverter, AudioFormat
from realtime_asr.audio.files import read_audio_file
from realtime_asr.errors import ASRError
from realtime_asr.results import CloseError, CloseResult
from realtime_asr.runtime.native.gpu import CUDA_DIRECTORY, GPUContext

LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler())


@dataclass(frozen=True, kw_only=True)
class RecognizerConfig:
    """选择显式本地模型和运行库，不读取隐藏环境配置。"""

    model_path: str | PathLike
    backend: str = "whisper"
    device: str = "cpu"
    compute_type: str | None = None
    device_index: int = 0
    cudnn_folder: str | PathLike | None = None
    cuda_directory: str | PathLike | None = None


class Recognizer:
    """通过小接口隐藏模型准备、输入转换和资源管理，不持有设备或服务。"""

    def __init__(self, config: RecognizerConfig):
        self._config = self._resolve_config(config)
        self._state = "unprepared"
        self._engine = None
        self._gpu = None
        self._close_result = None
        self._operation_lock = Lock()

    def __enter__(self):
        """进入时显式准备模型，失败保留可重新准备的状态。"""
        self.prepare()
        return self

    def __exit__(self, exc_type, exc, tb):
        """关闭自有资源，正文异常优先于关闭异常。"""
        result = self.close()
        if exc_type is None and result.status == "failed":
            raise ASRError("CLEANUP_FAILED", "close", result.errors[0].message)
        return False

    @property
    def config(self) -> RecognizerConfig:
        return self._config

    @property
    def state(self) -> str:
        return self._state

    @property
    def close_result(self) -> CloseResult | None:
        return self._close_result

    @staticmethod
    def _resolve_config(config):
        """固定路径和当前支持组合；不触碰模型文件或加载 DLL。"""
        if not isinstance(config, RecognizerConfig):
            raise ASRError("INVALID_CONFIGURATION", "configure", "需要 RecognizerConfig")
        valid_index = isinstance(config.device_index, int) and not isinstance(
            config.device_index, bool
        )
        if (
            config.backend not in ("whisper", "nemotron")
            or config.device not in ("cpu", "cuda")
            or not valid_index
            or config.device_index < 0
            or (config.device == "cpu" and config.device_index != 0)
            or (config.backend == "nemotron" and config.device != "cpu")
        ):
            raise ASRError("INVALID_CONFIGURATION", "configure", "后端或设备组合无效")
        compute_type = "float16" if config.device == "cuda" else "int8"
        if config.compute_type not in (None, compute_type):
            raise ASRError("INVALID_CONFIGURATION", "configure", "本阶段计算类型与设备不匹配")
        if config.device == "cpu" and (
            config.cudnn_folder is not None or config.cuda_directory is not None
        ):
            raise ASRError("INVALID_CONFIGURATION", "configure", "CPU 不使用 GPU 路径")
        if config.device == "cuda" and config.cudnn_folder is None:
            raise ASRError("INVALID_CONFIGURATION", "configure", "CUDA 需要显式 cuDNN 目录")
        try:
            model_path = Path(config.model_path).resolve()
            cudnn_folder = None
            cuda_directory = None
            if config.device == "cuda":
                cudnn_folder = Path(config.cudnn_folder).resolve()
                cuda_directory = (
                    Path(config.cuda_directory)
                    if config.cuda_directory is not None
                    else CUDA_DIRECTORY
                ).resolve()
            return replace(
                config,
                model_path=model_path,
                compute_type=compute_type,
                cudnn_folder=cudnn_folder,
                cuda_directory=cuda_directory,
            )
        except (TypeError, ValueError, OSError) as exc:
            raise ASRError("INVALID_CONFIGURATION", "configure", "模型或运行库路径无效") from exc

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
            if self.config.device == "cuda":
                self._gpu = GPUContext(self.config.cudnn_folder, self.config.cuda_directory)
                self._gpu.__enter__()
            if self.config.backend == "whisper":
                from realtime_asr.backends.whisper import WhisperEngine

                self._engine = WhisperEngine(
                    self.config.model_path,
                    device=self.config.device,
                    compute_type=self.config.compute_type,
                    device_index=self.config.device_index,
                )
            else:
                from realtime_asr.backends.nemotron import NemotronEngine

                self._engine = NemotronEngine(self.config.model_path)
            self._state = "ready"
        except BaseException as exc:
            cleanup_errors = self._release_resources()
            if cleanup_errors:
                exc.cleanup_errors = cleanup_errors
            if isinstance(exc, (ASRError, KeyboardInterrupt, SystemExit)):
                raise
            code = "DEPENDENCY_MISSING" if isinstance(exc, (ImportError, OSError)) else "MODEL_LOAD"
            raise ASRError(code, "load_model", str(exc)) from exc

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

    @staticmethod
    def _validate_options(language, word_timestamps):
        """调用边界先检查元数据，错误选项不能消费源或加载模型。"""
        if language is not None and (not isinstance(language, str) or not language.strip()):
            raise ASRError("INVALID_CONFIGURATION", "transcribe", "语言应为非空代码或 None")
        if type(word_timestamps) is not bool:
            raise ASRError("INVALID_CONFIGURATION", "transcribe", "词时间戳开关应为布尔值")

    def _snapshot(self, pcm, language):
        """识别固定的标准音频快照，使用实时角色参数。"""
        with self._operation():
            return self._infer(pcm, language, role="snapshot")

    def _create_stream(self, language):
        """为运行时创建独立原生流；后续操作由模型拥有者串行调度。"""
        with self._operation():
            self._prepare()
            backend_language = {None: "auto", "zh": "zh-CN", "en": "en-US"}.get(language, language)
            return self._engine.create_streaming_session(language=backend_language)

    def _infer(self, pcm, language, *, role="final", word_timestamps=False):
        """完整标准 PCM 只进入一次普通识别，不对输入块分别转写。"""
        if not pcm:
            raise ASRError("INVALID_AUDIO", "transcribe", "完整输入没有标准采样")
        self._prepare()
        if self.config.backend == "whisper":
            backend_language = {"zh-CN": "zh", "en-US": "en"}.get(language, language)
        else:
            backend_language = {None: "auto", "zh": "zh-CN", "en": "en-US"}.get(language, language)
        array = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        try:
            parameters = {"language": backend_language}
            if self.config.backend == "whisper":
                parameters["role"] = role
                parameters["word_timestamps"] = word_timestamps
            result = self._engine.transcribe(array, **parameters)
        except ASRError:
            raise
        except Exception as exc:
            raise ASRError("INFERENCE_FAILED", "transcribe", str(exc)) from exc
        language_source = result.language_source
        if self.config.backend == "whisper" and backend_language is not None:
            language_source = "hint" if result.reported_language else "unavailable"
        return replace(
            result,
            requested_language=language,
            language_source=language_source,
            segments=deepcopy(result.segments),
            backend_metadata=deepcopy(result.backend_metadata),
            timestamp_status=(
                "unavailable"
                if word_timestamps and self.config.backend != "whisper"
                else result.timestamp_status
            ),
        )

    def _release_resources(self):
        """尝试释放全部自有资源，每个失败保留独立记录，引用始终清除。"""
        errors = []
        engine, self._engine = self._engine, None
        gpu, self._gpu = self._gpu, None
        if engine is not None:
            try:
                engine.close()
            except Exception as exc:
                LOGGER.error("模型资源关闭失败：%s", exc)
                errors.append(CloseError("CLEANUP_FAILED", "close", str(exc)))
        if gpu is not None:
            try:
                gpu.__exit__(None, None, None)
            except Exception as exc:
                LOGGER.error("GPU 资源关闭失败：%s", exc)
                errors.append(CloseError("CLEANUP_FAILED", "close", str(exc)))
        return tuple(errors)

    def close(self) -> CloseResult:
        """保留首次关闭结果；不把引用清除当作宿主资源回到初始值。"""
        with self._operation(closing=True):
            if self.close_result is None:
                errors = self._release_resources()
                self._close_result = CloseResult(status="failed" if errors else "ok", errors=errors)
                self._state = "closed"
            return self.close_result
