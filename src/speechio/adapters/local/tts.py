"""本地合成共享的设备选择与标准音频交付。"""

from collections.abc import Mapping
from os import PathLike

import numpy as np

from speechio.audio.formats import AudioFormat
from speechio.audio.results import AudioResult
from speechio.errors import TTSError


def validate_local_config(config):
    """只在模型配置边界检查通用字段；设备名称对外使用 cpu / gpu。"""
    if (
        config.device not in ("cpu", "gpu")
        or type(config.device_index) is not int
        or config.device_index < 0
    ):
        raise TTSError(
            "INVALID_CONFIGURATION", "configure", "设备应为 cpu / gpu，设备序号应为非负整数"
        )
    if config.device == "cpu" and config.device_index != 0:
        raise TTSError("INVALID_CONFIGURATION", "configure", "CPU 不使用 GPU 序号")
    if config.compute_type not in (None, "float32", "float16", "bfloat16"):
        raise TTSError(
            "INVALID_CONFIGURATION", "configure", "本地合成支持 float32 / float16 / bfloat16"
        )
    if not isinstance(config.model_path, (str, PathLike)) or not str(config.model_path).strip():
        raise TTSError("INVALID_CONFIGURATION", "configure", "model_path 必须是已有模型目录")
    if config.adapter_options is not None and (
        not isinstance(config.adapter_options, Mapping) or config.adapter_options
    ):
        raise TTSError("INVALID_CONFIGURATION", "configure", "本地合成首版没有额外 adapter_options")


def torch_device(config, torch):
    """仅在适配器内转换 GPU 名称；不修改系统运行库。"""
    if config.device == "gpu":
        if not torch.cuda.is_available():
            raise TTSError("DEVICE_UNAVAILABLE", "prepare", "当前 PyTorch 环境无法使用 GPU")
        return f"cuda:{config.device_index}"
    return "cpu"


def waveform_result(waveform, sample_rate):
    """第三方浮点波形转换成公开的单声道 PCM16 结果。"""
    values = np.asarray(waveform, dtype=np.float32)
    if values.ndim != 1 or not values.size or not np.isfinite(values).all():
        raise TTSError("PROTOCOL_ERROR", "synthesize", "模型未返回有效的单声道音频")
    data = (np.clip(values, -1, 32767 / 32768) * 32768).astype("<i2").tobytes()
    return AudioResult(data, AudioFormat(sample_rate=int(sample_rate)))
