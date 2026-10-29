"""语音识别与语音合成 SDK；不管理应用业务或播放设备。"""

from speechio.adapters.contracts import (
    STTAdapter,
    STTCapabilities,
    STTStream,
    TTSAdapter,
    TTSCapabilities,
    VoiceCloningRequirements,
)
from speechio.audio.formats import AudioFormat
from speechio.audio.microphone import (
    InputDevice,
    MicrophoneConfig,
    MicrophoneInput,
    list_input_devices,
)
from speechio.audio.results import AudioChunk, AudioResult
from speechio.errors import SpeechioError, STTError, TTSError
from speechio.models.config import LocalModelConfig, RemoteModelConfig
from speechio.resources import CloseResult
from speechio.stt.api import (
    MicrophoneTranscriber,
    transcribe_audio,
    transcribe_microphone,
    transcribe_url,
)
from speechio.stt.input import AudioFrame
from speechio.stt.options import TranscriptionOptions
from speechio.stt.results import Transcript, WordTimestamp
from speechio.stt.runtime.asr import RuntimeLimits, STTRuntime
from speechio.stt.runtime.candidates import OnDemandConfig
from speechio.stt.runtime.realtime import RealtimeConfig
from speechio.stt.segmentation.gate import SegmentationConfig
from speechio.stt.segmentation.vad import VadConfig
from speechio.stt.segmentation.wakeword import WakeWordConfig
from speechio.stt.session.automatic import AutoSession
from speechio.stt.session.candidates import SessionTranscriptionRequest, SessionTranscriptionResult
from speechio.stt.session.session import (
    AudioSnapshot,
    Session,
    SessionConfig,
    SessionEvent,
    TurnRef,
)
from speechio.stt.text.endpoint import SpeechEndConfig
from speechio.stt.text.realtime_merge import RealtimeMergeConfig
from speechio.stt.text.stabilizer import StabilizationConfig
from speechio.stt.transcriber import Transcriber
from speechio.tts.api import clone_voice, synthesize
from speechio.tts.options import SynthesisOptions
from speechio.tts.stream import SynthesisStream
from speechio.tts.synthesizer import Synthesizer
from speechio.tts.timing import TextTiming, WordTiming
from speechio.tts.voices import Voice, VoiceInfo

__all__ = [
    "clone_voice",
    "Voice",
    "VoiceInfo",
    "VoiceCloningRequirements",
    "TranscriptionOptions",
    "STTRuntime",
    "SessionTranscriptionRequest",
    "SessionTranscriptionResult",
    "synthesize",
    "Synthesizer",
    "SynthesisOptions",
    "SynthesisStream",
    "TextTiming",
    "WordTiming",
    "TTSCapabilities",
    "TTSAdapter",
    "AudioResult",
    "AudioChunk",
    "SpeechioError",
    "TTSError",
    "transcribe_audio",
    "transcribe_url",
    "transcribe_microphone",
    "Transcriber",
    "MicrophoneTranscriber",
    "Session",
    "AutoSession",
    "LocalModelConfig",
    "RemoteModelConfig",
    "RealtimeConfig",
    "SessionConfig",
    "VadConfig",
    "SegmentationConfig",
    "SpeechEndConfig",
    "StabilizationConfig",
    "WakeWordConfig",
    "OnDemandConfig",
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
    "STTAdapter",
    "STTCapabilities",
    "STTStream",
    "STTError",
]
