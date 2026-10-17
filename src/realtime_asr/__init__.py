"""独立 ASR SDK；按音频、会话、分句、后端与运行时组织实现。"""

from realtime_asr.api import StreamTranscriber, transcribe_audio, transcribe_stream
from realtime_asr.audio.conversion import AudioFormat, AudioFrame
from realtime_asr.audio.microphone import (
    InputDevice,
    MicrophoneConfig,
    MicrophoneInput,
    list_input_devices,
)
from realtime_asr.engines.base import (
    BaseTranscriptionEngine,
    EngineCapabilities,
    StreamingSession,
)
from realtime_asr.errors import ASRError
from realtime_asr.results import CloseResult, Transcript, WordTimestamp
from realtime_asr.runtime.asr import RuntimeLimits
from realtime_asr.runtime.candidates import CandidateConfig
from realtime_asr.runtime.realtime import RealtimeConfig
from realtime_asr.runtime.transcriber import ModelConfig, Transcriber
from realtime_asr.segmentation.gate import SegmentationConfig
from realtime_asr.segmentation.vad import VadConfig
from realtime_asr.segmentation.wakeword import WakeConfig
from realtime_asr.session.automatic import LiveSession
from realtime_asr.session.session import (
    AudioSnapshot,
    Session,
    SessionConfig,
    SessionEvent,
    TurnRef,
)
from realtime_asr.text.endpoint import EndpointConfig
from realtime_asr.text.realtime_merge import RealtimeMergeConfig
from realtime_asr.text.stabilizer import StabilizationConfig

__all__ = [
    "transcribe_audio",
    "transcribe_stream",
    "Transcriber",
    "StreamTranscriber",
    "Session",
    "LiveSession",
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
