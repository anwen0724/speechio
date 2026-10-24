"""浏览器输入边界与事件序列化；音频使用独立的二进制消息。"""

from dataclasses import asdict
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class RunConfig(BaseModel):
    """一次运行的固定参数，不接受浏览器提供的文件系统路径。"""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    model: str
    device: Literal["cpu", "gpu"] | None = None
    device_index: int | None = Field(default=None, ge=0, strict=True)
    compute_type: str | None = Field(default=None, min_length=1)
    timeout_seconds: float | None = Field(default=None, gt=0)
    task_timeout_seconds: float | None = Field(default=None, gt=0)
    language: str | None = "zh"
    input_mode: Literal["live", "file", "chunks"] = "live"
    segmentation: Literal["auto", "manual"] = "auto"
    sample_rate: int = Field(default=16000, ge=8000, le=192000)
    channels: int = Field(default=1, ge=1, le=2)
    realtime: bool = True
    completion_mode: Literal["auto", "final"] = "auto"
    realtime_mode: Literal["snapshot", "streaming"] = "snapshot"
    realtime_model: str | None = None
    realtime_device: Literal["cpu", "gpu"] | None = None
    realtime_device_index: int | None = Field(default=None, ge=0, strict=True)
    realtime_compute_type: str | None = Field(default=None, min_length=1)
    realtime_timeout_seconds: float | None = Field(default=None, gt=0)
    interval: float = Field(default=0.2, gt=0)
    stabilize: bool = True
    word_timestamps: bool = False
    silence: float = Field(default=0.8, ge=0.16)
    vad_sensitivity: float = Field(default=0.4, ge=0, le=1)
    wake: str | None = None
    wake_sensitivity: float = Field(default=0.5, ge=0, le=1)
    wake_timeout: float = Field(default=5, gt=0)
    candidates: bool = False
    correction: bool = False
    max_audio_samples: int = Field(default=480000, gt=0)
    max_pending_units: int = Field(default=8, gt=0)
    max_audio_seconds: float | None = Field(default=None, gt=0)
    chunk_ms: float = Field(default=100, ge=10, le=1000)
    pace: Literal["realtime", "fast"] = "realtime"

    @model_validator(mode="after")
    def validate_combination(self):
        if self.segmentation == "manual" and self.wake:
            raise ValueError("手动分段不启用唤醒")
        if self.correction and not self.candidates:
            raise ValueError("修正需要启用候选结果")
        return self


class SynthesisConfig(BaseModel):
    """本次合成覆盖项；未传字段沿用服务端预设。"""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    model: str
    device: Literal["cpu", "gpu"] | None = None
    mode: Literal["full", "stream", "incremental"] = "stream"
    voice: str | None = None
    saved_voice_id: str | None = None
    reference_audio_token: str | None = None
    reference_text: str | None = None
    language: str | None = None
    speed: float | None = Field(default=None, gt=0)
    instruction: str | None = None
    sample_rate: int | None = Field(default=None, gt=0)
    timeout_seconds: float | None = Field(default=None, gt=0)
    max_buffer_seconds: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def validate_voice(self):
        if (
            sum(
                bool(value)
                for value in (self.voice, self.saved_voice_id, self.reference_audio_token)
            )
            > 1
        ):
            raise ValueError("预设音色、已保存音色和参考音频只能选择一项")
        if self.reference_audio_token and self.mode != "full":
            raise ValueError("参考音频直接合成只支持完整生成；流式请先克隆并保存音色")
        if self.reference_text is not None and not self.reference_audio_token:
            raise ValueError("参考文字需要同时上传参考音频")
        return self


class CloneVoiceRequest(BaseModel):
    """独立克隆的模型与录音；名称只用于应用保存记录。"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, allow_inf_nan=False)
    model: str
    device: Literal["cpu", "gpu"] | None = None
    name: str = Field(min_length=1, max_length=100)
    reference_audio_token: str
    reference_text: str | None = None
    timeout_seconds: float | None = Field(default=None, gt=0)
    save_dir: str | None = Field(default=None, min_length=1)


class ImportVoiceRequest(BaseModel):
    """导入已有核心音色文件，并绑定本次选择的模型配置。"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    model: str
    name: str = Field(min_length=1, max_length=100)
    file_token: str


class SynthesisCommand(BaseModel):
    """完整文本与增量文本共用一条连接；音频以二进制返回。"""

    model_config = ConfigDict(extra="forbid")
    type: Literal["start", "text", "finish_input", "cancel"]
    config: SynthesisConfig | None = None
    text: str = Field(default="", max_length=200_000)


class Command(BaseModel):
    """控制消息与音频消息按同一连接的接收顺序执行。"""

    model_config = ConfigDict(extra="forbid")
    type: Literal[
        "start", "end_input", "end_turn", "cancel_turn", "reset", "candidate", "correct", "close"
    ]
    config: RunConfig | None = None
    file_token: str | None = None


def event_message(event):
    """保留核心包事件的归属、错误和时间信息，方便调试与导出。"""
    return {"type": "event", **asdict(event)}
