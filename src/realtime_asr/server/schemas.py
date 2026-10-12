"""网络边界的严格字段约束；内部仍使用 SDK dataclass。"""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from realtime_asr import AudioFormat, UnitRef

PositiveInt = Annotated[int, Field(gt=0)]
NonnegativeInt = Annotated[int, Field(ge=0)]
PositiveSeconds = Annotated[float, Field(gt=0, allow_inf_nan=False)]
RequestId = Annotated[str, Field(min_length=1, max_length=128)]


class StrictModel(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")


class Format(StrictModel):
    sample_rate: Annotated[int, Field(ge=1, le=192000)] = 16000
    channels: Annotated[int, Field(ge=1, le=8)] = 1
    sample_type: Literal["int16", "float32"] = "int16"

    def to_audio_format(self):
        return AudioFormat(self.sample_rate, self.channels, self.sample_type)


class HttpInput(Format):
    language: Annotated[str, Field(min_length=1)] | None = None
    word_timestamps: bool = False


class Versioned(StrictModel):
    protocol_version: Annotated[int, Field(ge=1, le=1)]


class Open(Versioned):
    type: Literal["open"]
    request_id: RequestId
    token: str | None = None
    mode: Literal["manual", "auto"] = "manual"
    format: Format = Field(default_factory=Format)
    language: Annotated[str, Field(min_length=1)] | None = None
    realtime: bool = True
    dual_realtime: bool = False
    completion_mode: Literal["final", "candidate"] = "final"
    retain_for_correction: bool = False
    max_audio_seconds: PositiveSeconds | None = None
    max_pending_units: PositiveInt | None = None
    webrtc_mode: Annotated[int, Field(ge=0, le=3)] = 3
    silero_end_detection: bool = False
    trim_pre_roll: bool = False
    backdate_seconds: Annotated[float, Field(ge=0, allow_inf_nan=False)] = 0
    stable_text: bool = False
    word_timestamps: bool = False
    early_transcription_silence: Annotated[float, Field(ge=0, allow_inf_nan=False)] = 0
    dynamic_silence: bool = False
    punctuation_split: bool = False


class Unit(StrictModel):
    session_id: str
    generation: NonnegativeInt
    unit_id: PositiveInt

    def to_ref(self):
        return UnitRef(self.session_id, self.generation, self.unit_id)


class Command(Versioned):
    type: Literal[
        "start",
        "finish",
        "end_input",
        "cancel",
        "reset",
        "ping",
        "close",
        "candidate",
        "resume",
        "correct",
    ]
    request_id: RequestId
    session_id: str
    generation: NonnegativeInt
    unit: Unit | None = None
    candidate_kind: Literal["tail", "whole"] = "whole"
    hint: bool = False
    expected_audio_samples: NonnegativeInt | None = None

    @model_validator(mode="after")
    def command_fields(self):
        """只接纳该操作真正使用的字段，避免看似成功却被忽略的控制。"""
        if self.type != "candidate" and self.model_fields_set & {"candidate_kind", "hint"}:
            raise ValueError("候选字段只用于 candidate")
        if self.type != "resume" and "expected_audio_samples" in self.model_fields_set:
            raise ValueError("采样边界只用于 resume")
        if self.type == "correct" and self.unit is None:
            raise ValueError("correct 要求 unit")
        return self


class AudioMetadata(Versioned):
    session_id: str
    generation: NonnegativeInt
    unit_id: PositiveInt | None
    audio_sequence: NonnegativeInt
    format: Format
    frame_count: PositiveInt


class ConfigPatch(StrictModel):
    idle_timeout: PositiveSeconds | None = None
    drain_timeout: PositiveSeconds | None = None
    send_timeout: PositiveSeconds | None = None
    max_audio_seconds: Annotated[float, Field(gt=0, le=600, allow_inf_nan=False)] | None = None
    max_pending_units: PositiveInt | None = None


def validation_details(error):
    """不回传原始输入，避免鉴权失败附带客户端 Token。"""
    return [
        {"field": list(item["loc"]), "message": item["msg"]}
        for item in error.errors(include_input=False, include_url=False)
    ]
