"""独立 ASR SDK；按音频、会话、分句、后端与运行时组织实现。"""

from realtime_asr.core.audio.conversion import AudioConverter, AudioFormat, AudioFrame
from realtime_asr.core.audio.microphone import (
    InputDevice,
    MicrophoneConfig,
    MicrophoneInput,
    list_input_devices,
)
from realtime_asr.core.engines.base import (
    BaseTranscriptionEngine,
    EngineCapabilities,
    StreamingSession,
)
from realtime_asr.core.errors import ASRError
from realtime_asr.core.results import CloseResult, RecognitionResult, WordTimestamp
from realtime_asr.core.runtime.asr import (
    ASRRuntime,
    RuntimeCapacity,
    RuntimeCloseResult,
    RuntimeLimits,
)
from realtime_asr.core.runtime.candidates import CandidateConfig
from realtime_asr.core.runtime.realtime import RealtimeConfig
from realtime_asr.core.runtime.recognizer import Recognizer, RecognizerConfig
from realtime_asr.core.segmentation.acoustic import AcousticBoundaryDetector
from realtime_asr.core.segmentation.gate import SegmentationConfig, SegmentBoundary
from realtime_asr.core.segmentation.vad import VadConfig, VoiceActivity, VoiceDetector
from realtime_asr.core.segmentation.wakeword import WakeConfig, WakeDetector
from realtime_asr.core.session.automatic import AutomaticSession
from realtime_asr.core.session.candidates import CandidateRequest, CandidateResult, ResumeAck
from realtime_asr.core.session.session import (
    AudioSnapshot,
    EventError,
    FinalTask,
    Session,
    SessionChange,
    SessionConfig,
    SessionEvent,
    UnitRef,
)
from realtime_asr.core.text.endpoint import EndpointConfig
from realtime_asr.core.text.realtime_merge import (
    RealtimeMergeConfig,
    RealtimeMerger,
    RealtimeMergeResult,
)
from realtime_asr.core.text.stabilizer import StabilizationConfig, TextObservation, TextStabilizer
from realtime_asr.core.text.tail_repair import TailRepairResult, repair_tail

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
    "WakeConfig",
    "WakeDetector",
    "AudioSnapshot",
]
