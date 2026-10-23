"""独立唤醒检测及样本时钟门控，不拥有 ASR 单元或模型任务。"""

import os
from dataclasses import dataclass, field
from math import isfinite
from pathlib import Path
from threading import RLock
from typing import Protocol

import numpy as np

from speechio.errors import STTError
from speechio.resources import CloseError


@dataclass(frozen=True, kw_only=True)
class WakeWordConfig:
    """宿主检测参数和门控秒数；密钥不参与日志表示。"""

    backend: str = "openwakeword"
    model_paths: tuple[Path, ...] = ()
    keywords: tuple[str, ...] = ()
    model_path: Path | None = None
    feature_model_dir: Path | None = None
    access_key: str | None = field(default=None, repr=False)
    sensitivity: float = 0.5
    activation_delay: float = 0
    speech_timeout: float = 5
    trim_duration: float = 0.1
    followup_window: float = 0

    def __post_init__(self):
        seconds = (
            self.activation_delay,
            self.speech_timeout,
            self.trim_duration,
            self.followup_window,
        )
        if (
            self.backend not in ("porcupine", "openwakeword")
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not isfinite(value)
                or value < 0
                for value in seconds
            )
            or self.speech_timeout <= 0
            or not 0 <= self.sensitivity <= 1
            or not isinstance(self.model_paths, tuple)
            or not isinstance(self.keywords, tuple)
        ):
            raise STTError("INVALID_CONFIGURATION", "configure", "唤醒后端、阈值或时间无效")


class WakeDetector(Protocol):
    """单个输入代次的原生检测上下文。"""

    sample_rate: int
    frame_samples: int

    def detect(self, pcm: bytes) -> str | None: ...
    def reset(self): ...
    def close(self): ...


class LocalWakeDetector:
    """隐藏两个运行库的准备、固定帧和释放差异，准备时不下载模型。"""

    sample_rate = 16000

    def __init__(self, config):
        self.config = config
        self._model = None
        try:
            if config.backend == "porcupine":
                import pvporcupine

                key = config.access_key or os.environ.get("PICOVOICE_ACCESS_KEY")
                if not key:
                    raise STTError(
                        "WAKE_ACCESS_KEY_REQUIRED", "prepare_wake", "需要 PICOVOICE_ACCESS_KEY"
                    )
                if bool(config.keywords) == bool(config.model_paths):
                    raise STTError(
                        "INVALID_CONFIGURATION",
                        "prepare_wake",
                        "关键词或自定义模型应且仅应提供一个",
                    )
                selected = (
                    {"keywords": list(config.keywords)}
                    if config.keywords
                    else {"keyword_paths": [str(path) for path in config.model_paths]}
                )
                self._names = config.keywords or tuple(
                    Path(path).stem for path in config.model_paths
                )
                self._model = pvporcupine.create(
                    access_key=key,
                    sensitivities=[config.sensitivity] * len(self._names),
                    **selected,
                    **({"model_path": str(config.model_path)} if config.model_path else {}),
                )
                self.sample_rate = self._model.sample_rate
                self.frame_samples = self._model.frame_length
            else:
                from openwakeword.model import Model

                directory = Path(config.feature_model_dir) if config.feature_model_dir else None
                if not config.model_paths or directory is None:
                    raise STTError(
                        "WAKE_MODEL_INVALID", "prepare_wake", "需要关键词与基础特征模型路径"
                    )
                paths = [Path(path) for path in config.model_paths]
                features = [directory / "melspectrogram.onnx", directory / "embedding_model.onnx"]
                if any(path.suffix != ".onnx" or not path.is_file() for path in paths + features):
                    raise STTError(
                        "WAKE_MODEL_INVALID", "prepare_wake", "唤醒 ONNX 文件不存在或类型无效"
                    )
                self._model = Model(
                    wakeword_models=[str(path) for path in paths],
                    inference_framework="onnx",
                    melspec_model_path=str(features[0]),
                    embedding_model_path=str(features[1]),
                )
                self._names = tuple(self._model.models)
                self.frame_samples = 1280
            if self.sample_rate != 16000:
                self.close()
                raise STTError("WAKE_MODEL_INVALID", "prepare_wake", "唤醒检测采样率应为 16000")
        except ImportError:
            extra = "wake-porcupine" if config.backend == "porcupine" else "wake-openwakeword"
            raise STTError(
                "OPTIONAL_DEPENDENCY_MISSING", "prepare_wake", f"请安装 speechio[{extra}]"
            ) from None
        except STTError:
            raise
        except Exception as exc:
            # 原生异常可能包含密钥；向调用方只公开异常类型。
            self.close()
            raise STTError(
                "WAKE_MODEL_INVALID", "prepare_wake", f"唤醒准备失败：{type(exc).__name__}"
            ) from None

    def detect(self, pcm):
        audio = np.frombuffer(pcm, dtype="<i2")
        if self.config.backend == "porcupine":
            index = self._model.process(audio.tolist())
            return self._names[index] if index >= 0 else None
        prediction = self._model.predict(audio)
        name = max(prediction, key=prediction.get)
        return name if prediction[name] >= self.config.sensitivity else None

    def reset(self):
        if self.config.backend == "openwakeword":
            self._model.reset()
        else:
            # 运行库没有公开 reset；重建原生上下文，避免带入上一段特征。
            self.close()
            self.__init__(self.config)

    def close(self):
        model, self._model = self._model, None
        if model is not None and self.config.backend == "porcupine":
            model.delete()


class WakeGate:
    """拥有有界检测余量和门控时间；关闭延后至正在执行的检测返回。"""

    def __init__(self, config, pre_roll_samples, factory=None, *, on_release):
        if (
            not isinstance(config, WakeWordConfig)
            or int(config.trim_duration * 16000) > pre_roll_samples
        ):
            raise STTError("INVALID_CONFIGURATION", "configure", "唤醒裁剪不能超过前置缓冲")
        try:
            detector = factory() if factory else LocalWakeDetector(config)
        except STTError:
            raise
        except Exception as exc:
            raise STTError("WAKE_MODEL_INVALID", "prepare_wake", type(exc).__name__) from None
        if (
            getattr(detector, "sample_rate", None) != 16000
            or type(getattr(detector, "frame_samples", None)) is not int
            or detector.frame_samples <= 0
            or any(
                not callable(getattr(detector, name, None)) for name in ("detect", "reset", "close")
            )
        ):
            if callable(getattr(detector, "close", None)):
                detector.close()
            raise STTError("WAKE_MODEL_INVALID", "prepare_wake", "检测器不符合 WakeDetector 契约")
        self.config, self.detector, self.pre_roll = config, detector, pre_roll_samples
        self._lock = RLock()
        self._closing = self.released = self.busy = False
        self._on_release = on_release
        self.errors = ()
        self._pending = bytearray()
        self._reset_needed = False
        self.position = self.detections = self.timeouts = 0
        self.keyword = self.trigger = None
        self.state = "grace" if config.activation_delay else "awaiting_wake"
        self.deadline = int(config.activation_delay * 16000) if config.activation_delay else None
        self.recording_context = None

    @property
    def status(self):
        return {
            "enabled": True,
            "backend": self.config.backend,
            "state": self.state,
            "position_samples": self.position,
            "keyword": self.keyword,
            "trigger_sample": self.trigger,
            "deadline_sample": self.deadline,
            "detections": self.detections,
            "timeouts": self.timeouts,
            "released": self.released,
        }

    def process(self, frame, *, recording):
        """返回是否开放 VAD 与需清除的历史边界，检测调用在全局锁外。"""
        with self._lock:
            self.busy = True
            try:
                if self._reset_needed:
                    self.detector.reset()
                    self._reset_needed = False
                return self._process(frame, recording)
            except STTError:
                raise
            except Exception as exc:
                raise STTError("WAKE_DETECTION_FAILED", "detect_wake", type(exc).__name__) from exc
            finally:
                self.busy = False
                if self._closing:
                    self._release()

    def _process(self, frame, recording):
        self.position = frame.end_sample
        if recording:
            if self.state != "recording":
                self.recording_context = dict(self.status)
                self._pending.clear()
                self.detector.reset()
            self.state = "recording"
            return True, None
        if self.state == "recording":
            self.state = "followup" if self.config.followup_window else "awaiting_wake"
            self.deadline = (
                frame.start_sample + int(self.config.followup_window * 16000)
                if self.config.followup_window
                else None
            )
        expired = self.deadline is not None and frame.start_sample >= self.deadline
        clear_before = None
        if expired:
            self.timeouts += int(self.state == "awaiting_speech")
            self.state, self.deadline = "awaiting_wake", None
            self._pending.clear()
            self.detector.reset()
            clear_before = frame.start_sample
        if self.state != "awaiting_wake":
            return True, clear_before
        self._pending.extend(frame.pcm)
        size = self.detector.frame_samples * 2
        while len(self._pending) >= size:
            pcm = bytes(self._pending[:size])
            del self._pending[:size]
            name = self.detector.detect(pcm)
            if self._closing:
                return False, None
            if name is not None:
                if not isinstance(name, str) or not name:
                    raise STTError("WAKE_DETECTION_FAILED", "detect_wake", "命中名称无效")
                self.trigger = frame.end_sample - len(self._pending) // 2
                self.keyword = name
                self.detections += 1
                self.state = "awaiting_speech"
                self.deadline = self.trigger + int(self.config.speech_timeout * 16000)
                self._pending.clear()
                start = max(0, self.trigger - self.pre_roll)
                return True, min(self.trigger, start + int(self.config.trim_duration * 16000))
        return False, clear_before

    def cancel(self):
        self.state, self.deadline = "awaiting_wake", None
        self.keyword = self.trigger = self.recording_context = None
        self._pending.clear()
        self._reset_needed = True

    def close(self):
        """控制立即失效，原生释放不会与 detect 同时发生。"""
        self._closing = True
        if self._lock.acquire(blocking=False):
            try:
                self._release()
            finally:
                self._lock.release()

    def recording_started(self):
        """首次语音开口固定触发上下文，正文期间不检测唤醒。"""
        self.recording_context = dict(self.status)
        self.state = "recording"
        self._pending.clear()
        # 状态交接可能持有会话锁；原生重置放到下一次锁外检测流程。
        self._reset_needed = True

    def recording_finished(self, sample):
        self.state = "followup" if self.config.followup_window else "awaiting_wake"
        self.deadline = (
            sample + int(self.config.followup_window * 16000)
            if self.config.followup_window
            else None
        )

    def _release(self):
        if self.released:
            return
        try:
            self.detector.close()
        except Exception as exc:
            self.errors = (CloseError("WAKE_CLOSE_FAILED", "close_wake", type(exc).__name__),)
        self._pending.clear()
        self.released = True
        self._on_release(self)
