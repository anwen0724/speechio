"""独立 ASR SDK；按音频、会话、分句、后端与运行时组织实现。"""

from speechio.api import StreamTranscriber, transcribe_audio, transcribe_stream
from speechio.audio.conversion import AudioFormat, AudioFrame
from speechio.audio.microphone import (
    InputDevice,
    MicrophoneConfig,
    MicrophoneInput,
    list_input_devices,
)
from speechio.config import LocalModelConfig, RemoteModelConfig
from speechio.engines.base import (
    BaseTranscriptionEngine,
    EngineCapabilities,
    StreamingSession,
)
from speechio.errors import ASRError
from speechio.results import CloseResult, Transcript, WordTimestamp
from speechio.runtime.asr import RuntimeLimits
from speechio.runtime.candidates import CandidateConfig
from speechio.runtime.realtime import RealtimeConfig
from speechio.runtime.transcriber import ModelConfig, Transcriber
from speechio.segmentation.gate import SegmentationConfig
from speechio.segmentation.vad import VadConfig
from speechio.segmentation.wakeword import WakeConfig
from speechio.session.automatic import LiveSession
from speechio.session.session import (
    AudioSnapshot,
    Session,
    SessionConfig,
    SessionEvent,
    TurnRef,
)
from speechio.text.endpoint import EndpointConfig
from speechio.text.realtime_merge import RealtimeMergeConfig
from speechio.text.stabilizer import StabilizationConfig

__all__ = [
    "transcribe_audio",
    "transcribe_stream",
    "Transcriber",
    "StreamTranscriber",
    "Session",
    "LiveSession",
    "LocalModelConfig",
    "RemoteModelConfig",
    "ModelConfig",
    "RealtimeConfig",
    "SessionConfig",
    "VadConfig",
    "SegmentationConfig",
    "EndpointConfig",
    "StabilizationConfig",
    "WakeConfig",
    "CandidateConfig",
    "RealtimeMergeConfig",
    "RuntimeLimits",
    "Transcript",
    "WordTimestamp",
    "CloseResult",
    "SessionEvent",
    "TurnRef",
    "AudioSnapshot",
    "AudioFormat",
    "AudioFrame",
    "MicrophoneConfig",
    "MicrophoneInput",
    "InputDevice",
    "list_input_devices",
    "BaseTranscriptionEngine",
    "EngineCapabilities",
    "StreamingSession",
    "ASRError",
]
