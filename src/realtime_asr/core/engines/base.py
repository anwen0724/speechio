"""识别引擎的公开契约；执行位置与整段、流式能力相互独立。"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from realtime_asr.core.errors import ASRError
from realtime_asr.core.results import RecognitionResult


@dataclass(frozen=True, kw_only=True)
class EngineCapabilities:
    """一个引擎的固定能力；身份字段只用于诊断，不参与功能推导。"""

    engine: str
    model: str | None = None
    device: str | None = None
    transcription: bool = True
    snapshot: bool = False
    streaming: bool = False
    word_timestamps: bool = False

    def __post_init__(self):
        if (
            not isinstance(self.engine, str)
            or not self.engine.strip()
            or any(
                type(value) is not bool
                for value in (
                    self.transcription,
                    self.snapshot,
                    self.streaming,
                    self.word_timestamps,
                )
            )
            or (self.snapshot and not self.transcription)
        ):
            raise ASRError("INVALID_CONFIGURATION", "configure_engine", "引擎能力声明无效")

    @property
    def realtime_mode(self):
        """入口优先选择真正的流式能力；未声明实时能力时保持不可用。"""
        if self.streaming:
            return "native"
        if self.snapshot:
            return "snapshot"
        return None


class StreamingSession(ABC):
    """一个音频单元的独立流；远程适配器在此隐藏发送、接收与排空。"""

    @abstractmethod
    def accept_audio(self, audio, sample_rate=None):
        """接纳新增的 16 kHz 单声道 float32，不重复递交历史前缀。"""

    def decode(self):
        """推进已接纳输入；异步接收型引擎可以直接使用已有结果缓存。"""

    @abstractmethod
    def get_result(self) -> RecognitionResult:
        """返回本单元累计结果，不结束核心音频单元。"""

    @abstractmethod
    def finish(self) -> RecognitionResult:
        """结束并排空全部接纳输入，重复结束不得再次递交结束消息。"""

    def close(self):
        """释放本流资源，不影响共享引擎或其他音频单元。"""


class BaseTranscriptionEngine(ABC):
    """可由调用方继承或按相同契约实现的模型边界。"""

    @property
    @abstractmethod
    def capabilities(self) -> EngineCapabilities:
        """准备前可查询，生命周期内保持不变。"""

    def prepare(self):
        """准备自有模型或连接；默认适用于已经准备好的轻量实现。"""

    def transcribe(self, audio, language=None, *, role="final", word_timestamps=False):
        """识别固定标准音频；仅支持流式的实现无需提供整段调用。"""
        raise ASRError("BACKEND_INCOMPATIBLE", "transcribe", "引擎不支持整段识别")

    def create_streaming_session(self, language=None):
        """每个单元创建独立流，默认不声明流式能力。"""
        raise ASRError("BACKEND_INCOMPATIBLE", "create_stream", "引擎不支持原生流式")

    def close(self):
        """释放自有资源；没有额外资源的实现可以使用默认行为。"""


def validate_engine(engine):
    """只在注入边界检查契约，不在每个音频块重新探测方法。"""
    capabilities = getattr(engine, "capabilities", None)
    methods = ["prepare", "close"]
    if isinstance(capabilities, EngineCapabilities):
        if capabilities.transcription:
            methods.append("transcribe")
        if capabilities.streaming:
            methods.append("create_streaming_session")
    if not isinstance(capabilities, EngineCapabilities) or any(
        not callable(getattr(engine, method, None)) for method in methods
    ):
        raise ASRError("INVALID_CONFIGURATION", "configure_engine", "引擎不符合公开接入契约")
    return capabilities
