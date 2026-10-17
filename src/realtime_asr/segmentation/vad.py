"""本地双 VAD：隐藏 ONNX 图、合法检测帧及每会话递归状态。"""

import hashlib
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Protocol

import numpy as np

from realtime_asr.errors import ASRError


@dataclass(frozen=True, kw_only=True)
class VadConfig:
    """检测器的显式文件与开口、句尾选择，不包含识别配置。"""

    model_path: Path
    webrtc_mode: int = 3
    silero_sensitivity: float = 0.4
    silero_end_detection: bool = False
    intra_op_threads: int = 2
    warmup: bool = True

    def __post_init__(self):
        if (
            type(self.webrtc_mode) is not int
            or not 0 <= self.webrtc_mode <= 3
            or not isinstance(self.silero_sensitivity, (int, float))
            or not isfinite(self.silero_sensitivity)
            or not 0 <= self.silero_sensitivity <= 1
            or type(self.silero_end_detection) is not bool
            or type(self.intra_op_threads) is not int
            or self.intra_op_threads <= 0
            or type(self.warmup) is not bool
        ):
            raise ASRError("INVALID_CONFIGURATION", "configure", "VAD 配置无效")


@dataclass(frozen=True)
class VoiceActivity:
    """同次正向检测的开口、句尾和保守裁剪依据。"""

    onset: bool
    speech: bool
    webrtc: bool
    silero: bool
    rms: float


class VoiceDetector(Protocol):
    """一个输入代次的检测上下文，与 ASR 转写引擎具有不同职责。"""

    info: dict

    def detect(self, pcm: bytes, start_sample: int, *, recording=False) -> VoiceActivity:
        """提供语音活动记录，不创建任务或发布会话事件。"""

    def reset(self):
        """取消当前录音时重置检测上下文，保留连续源时钟。"""


def local_detector_factory(config):
    """只组装默认检测路线；ONNX 与 WebRTC 选择不泄漏到自动会话。"""
    if not isinstance(config, VadConfig):
        raise ASRError("INVALID_CONFIGURATION", "configure", "需要 VadConfig")
    return lambda: VoiceActivityDetector(config)


def create_detector(factory):
    """工厂属于模型外部边界，只在取得新上下文时检查契约。"""
    try:
        detector = factory()
    except ASRError:
        raise
    except Exception as exc:
        raise ASRError("VAD_MODEL_INVALID", "prepare_vad", str(exc)) from exc
    if (
        not isinstance(getattr(detector, "info", None), dict)
        or not callable(getattr(detector, "detect", None))
        or not callable(getattr(detector, "reset", None))
    ):
        raise ASRError("VAD_MODEL_INVALID", "prepare_vad", "检测器不符合 VoiceDetector 契约")
    return detector


class VoiceActivityDetector:
    """一个会话的检测拥有者；不共享上下文，不下载模型。"""

    def __init__(self, config: VadConfig):
        self.config = config
        try:
            import onnxruntime as ort
            import webrtcvad
        except ImportError as exc:
            raise ASRError(
                "OPTIONAL_DEPENDENCY_MISSING", "prepare_vad", "请安装 realtime-asr[vad]"
            ) from exc
        path = Path(config.model_path).resolve()
        try:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            options = ort.SessionOptions()
            options.intra_op_num_threads = config.intra_op_threads
            options.inter_op_num_threads = 1
            self._model = ort.InferenceSession(
                str(path), sess_options=options, providers=["CPUExecutionProvider"]
            )
            inputs = {item.name: item for item in self._model.get_inputs()}
            outputs = [item.name for item in self._model.get_outputs()]
            if (
                set(inputs) != {"input", "h", "c"}
                or inputs["input"].shape[-1] != 576
                or inputs["h"].shape != [1, 1, 128]
                or inputs["c"].shape != [1, 1, 128]
                or any(item.type != "tensor(float)" for item in inputs.values())
                or outputs != ["speech_probs", "hn", "cn"]
            ):
                raise ValueError("需要 16 kHz Silero v6 的 input/h/c 图")
        except Exception as exc:
            raise ASRError("VAD_MODEL_INVALID", "prepare_vad", str(exc)) from exc
        self._webrtc = webrtcvad.Vad(config.webrtc_mode)
        self.reset()
        if config.warmup:
            self.detect(bytes(1024), 0)
            tone = (np.sin(np.arange(512) * 2 * np.pi * 220 / 16000) * 983).astype("<i2")
            try:
                self._silero(np.zeros(512, np.float32))
                self._silero(tone.astype(np.float32) / 32768)
            except Exception as exc:
                raise ASRError("VAD_FAILED", "warmup_vad", str(exc)) from exc
            self.reset()
        self.info = {
            "backends": ("webrtc", "silero-v6"),
            "model_path": str(path),
            "sha256": digest,
            "provider": "CPUExecutionProvider",
            "warmed_up": config.warmup,
        }

    def reset(self):
        """清除一个代次的检测历史；模型文件与图仍由本对象拥有。"""
        self._context = np.zeros(64, np.float32)
        self._h = np.zeros((1, 1, 128), np.float32)
        self._c = np.zeros((1, 1, 128), np.float32)
        self._web_tail = b""
        self._last_web_end = None
        self._previous_web = False

    def detect(self, pcm: bytes, start_sample: int, *, recording=False) -> VoiceActivity:
        """检测副本可以补零；所有时序仍按有效音频位置计算。"""
        try:
            return self._detect(pcm, start_sample, recording=recording)
        except Exception as exc:
            raise ASRError("VAD_FAILED", "detect_vad", str(exc)) from exc

    def _silero(self, audio):
        """固定图输入与输出；递归状态只属于本会话。"""
        model_input = np.concatenate((self._context, audio)).reshape(1, 576)
        probability, self._h, self._c = self._model.run(
            None, {"input": model_input, "h": self._h, "c": self._c}
        )
        probability = float(np.asarray(probability).reshape(-1)[0])
        if not isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("Silero 返回无效概率")
        self._context = audio[-64:].copy()
        return probability > 1 - self.config.silero_sensitivity

    def _detect(self, pcm, start_sample, *, recording):
        samples = np.frombuffer(pcm, dtype="<i2")
        web = False
        combined = self._web_tail + pcm
        usable = len(combined) // 320 * 320
        if len(samples) < 512 and usable < len(combined):
            combined += bytes(320 - len(combined) % 320)
            usable = len(combined)
        for offset in range(0, usable, 320):
            web = self._webrtc.is_speech(combined[offset : offset + 320], 16000) or web
        self._web_tail = combined[usable:]
        if web:
            self._last_web_end = start_sample + len(samples)
            if not self._previous_web:
                self._context.fill(0)
                self._h.fill(0)
                self._c.fill(0)
        self._previous_web = web
        silero = False
        # 监听阶段由廉价检测触发；预热和 Silero 句尾要实际调用第二条路径。
        if web or (recording and self.config.silero_end_detection):
            audio = np.zeros(512, np.float32)
            audio[: len(samples)] = samples.astype(np.float32) / 32768
            silero = self._silero(audio)
        recent_web = (
            self._last_web_end is not None
            and start_sample + len(samples) - self._last_web_end <= 16000
        )
        rms = float(np.sqrt(np.mean(samples.astype(np.float64) ** 2))) if len(samples) else 0
        speech = silero if self.config.silero_end_detection else web
        return VoiceActivity(bool(silero and recent_web), bool(speech), web, silero, rms)
