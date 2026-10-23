"""语音识别与语音合成 SDK；不管理应用业务或播放设备。"""

from speechio.adapters.contracts import (
    BaseSynthesisEngine,
    BaseTranscriptionEngine,
    EngineCapabilities,
    StreamingSession,
    SynthesisCapabilities,
)
from speechio.audio.formats import AudioFormat
from speechio.audio.microphone import (
    InputDevice,
    MicrophoneConfig,
    MicrophoneInput,
    list_input_devices,
)
from speechio.audio.results import AudioChunk, AudioResult
from speechio.errors import ASRError, SpeechioError, TTSError
from speechio.models.config import LocalModelConfig, RemoteModelConfig
from speechio.resources import CloseResult
from speechio.stt.api import StreamTranscriber, transcribe_audio, transcribe_stream, transcribe_url
from speechio.stt.input import AudioFrame
from speechio.stt.results import Transcript, WordTimestamp
from speechio.stt.runtime.asr import RuntimeLimits
from speechio.stt.runtime.candidates import CandidateConfig
from speechio.stt.runtime.realtime import RealtimeConfig
from speechio.stt.segmentation.gate import SegmentationConfig
from speechio.stt.segmentation.vad import VadConfig
from speechio.stt.segmentation.wakeword import WakeConfig
from speechio.stt.session.automatic import LiveSession
from speechio.stt.session.session import (
    AudioSnapshot,
    Session,
    SessionConfig,
    SessionEvent,
    TurnRef,
)
from speechio.stt.text.endpoint import EndpointConfig
from speechio.stt.text.realtime_merge import RealtimeMergeConfig
from speechio.stt.text.stabilizer import StabilizationConfig
from speechio.stt.transcriber import ModelConfig, Transcriber
from speechio.tts.api import synthesize
from speechio.tts.options import SynthesisOptions
from speechio.tts.stream import SynthesisStream
from speechio.tts.synthesizer import Synthesizer

__all__ = [
    "synthesize",
    "Synthesizer",
    "SynthesisOptions",
    "SynthesisStream",
    "SynthesisCapabilities",
    "BaseSynthesisEngine",
    "AudioResult",
    "AudioChunk",
    "SpeechioError",
    "TTSError",
    "transcribe_audio",
    "transcribe_url",
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
