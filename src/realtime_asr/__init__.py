"""独立 ASR SDK；按音频、会话、分句、后端与运行时组织实现。"""

from realtime_asr.audio.conversion import AudioConverter, AudioFormat, AudioFrame
from realtime_asr.audio.microphone import (
    InputDevice,
    MicrophoneConfig,
    MicrophoneInput,
    list_input_devices,
)
from realtime_asr.backends.base import BaseTranscriptionEngine, EngineCapabilities, StreamingSession
from realtime_asr.errors import ASRError
from realtime_asr.results import CloseResult, RecognitionResult, WordTimestamp
from realtime_asr.runtime.asr import (
    ASRRuntime,
    RuntimeCapacity,
    RuntimeCloseResult,
    RuntimeLimits,
)
from realtime_asr.runtime.candidates import CandidateConfig
from realtime_asr.runtime.realtime import RealtimeConfig
from realtime_asr.runtime.recognizer import Recognizer, RecognizerConfig
from realtime_asr.segmentation.acoustic import AcousticBoundaryDetector
from realtime_asr.segmentation.gate import SegmentationConfig, SegmentBoundary
from realtime_asr.segmentation.vad import VadConfig, VoiceActivity, VoiceDetector
from realtime_asr.session.automatic import AutomaticSession
from realtime_asr.session.candidates import CandidateRequest, CandidateResult, ResumeAck
from realtime_asr.session.session import (
    EventError,
    FinalTask,
    Session,
    SessionChange,
    SessionConfig,
    SessionEvent,
    UnitRef,
)
from realtime_asr.text.endpoint import EndpointConfig
from realtime_asr.text.realtime_merge import (
    RealtimeMergeConfig,
    RealtimeMerger,
    RealtimeMergeResult,
)
from realtime_asr.text.stabilizer import StabilizationConfig, TextObservation, TextStabilizer
from realtime_asr.text.tail_repair import TailRepairResult, repair_tail

__all__ = [
    "CandidateConfig",
    "CandidateRequest",
    "CandidateResult",
    "ResumeAck",
    "TailRepairResult",
    "repair_tail",
    "RealtimeMergeConfig",
    "RealtimeMergeResult",
    "RealtimeMerger",
    "BaseTranscriptionEngine",
    "EngineCapabilities",
    "StreamingSession",
    "VoiceActivity",
    "VoiceDetector",
    "EndpointConfig",
    "AcousticBoundaryDetector",
    "WordTimestamp",
    "StabilizationConfig",
    "TextObservation",
    "TextStabilizer",
    "InputDevice",
    "MicrophoneConfig",
    "MicrophoneInput",
    "list_input_devices",
    "ASRError",
    "ASRRuntime",
    "AudioConverter",
    "AudioFormat",
    "AudioFrame",
    "AutomaticSession",
    "CloseResult",
    "EventError",
    "FinalTask",
    "RecognitionResult",
    "RealtimeConfig",
    "Recognizer",
    "RecognizerConfig",
    "RuntimeCapacity",
    "RuntimeCloseResult",
    "RuntimeLimits",
    "Session",
    "SegmentBoundary",
    "SegmentationConfig",
    "SessionChange",
    "SessionConfig",
    "SessionEvent",
    "UnitRef",
    "VadConfig",
]
