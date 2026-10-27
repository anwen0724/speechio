"""识别引擎的公开契约；执行位置与整段、流式能力相互独立。"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from speechio.errors import STTError, TTSError
from speechio.stt.results import Transcript


@dataclass(frozen=True, kw_only=True)
class STTCapabilities:
    """一个引擎的固定能力；身份字段只用于诊断，不参与功能推导。"""

    engine: str
    model: str | None = None
    device: str | None = None
    transcription: bool = True
    snapshot: bool = False
    streaming: bool = False
    word_timestamps: bool = False
    file_transcription: bool = False
    url_transcription: bool = False

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
                    self.file_transcription,
                    self.url_transcription,
                )
            )
            or (self.snapshot and not self.transcription)
        ):
            raise STTError("INVALID_CONFIGURATION", "configure_engine", "引擎能力声明无效")

    @property
    def realtime_mode(self):
        """入口优先选择真正的流式能力；未声明实时能力时保持不可用。"""
        if self.streaming:
            return "streaming"
        if self.snapshot:
            return "snapshot"
        return None


class STTStream(ABC):
    """一个音频单元的独立流；远程适配器在此隐藏发送、接收与排空。"""

    @abstractmethod
    def accept_audio(self, audio, sample_rate=None):
        """接纳新增的 16 kHz 单声道 float32，不重复递交历史前缀。"""

    def decode(self):
        """推进已接纳输入；异步接收型引擎可以直接使用已有结果缓存。"""

    @abstractmethod
    def get_result(self) -> Transcript:
        """返回本单元累计结果，不结束核心音频单元。"""

    @abstractmethod
    def finish(self) -> Transcript:
        """结束并排空全部接纳输入，重复结束不得再次递交结束消息。"""

    def close(self):
        """释放本流资源，不影响共享引擎或其他音频单元。"""

    def cancel(self):
        """非阻塞地请求中断；可与发送/接收并发，资源仍由 close 释放。"""


class STTAdapter(ABC):
    """可由调用方继承或按相同契约实现的模型边界。"""

    @property
    @abstractmethod
    def capabilities(self) -> STTCapabilities:
        """准备前可查询，生命周期内保持不变。"""

    def prepare(self):
        """准备自有模型或连接；默认适用于已经准备好的轻量实现。"""

    def transcribe(self, audio, language=None, *, role="final", word_timestamps=False):
        """识别固定标准音频；仅支持流式的实现无需提供整段调用。"""
        raise STTError("BACKEND_INCOMPATIBLE", "transcribe", "引擎不支持整段识别")

    def create_streaming_session(self, language=None):
        """每个单元创建独立流，默认不声明流式能力。"""
        raise STTError("BACKEND_INCOMPATIBLE", "create_stream", "引擎不支持原生流式")

    def transcribe_file(
        self, path, language=None, *, word_timestamps=False, task_timeout_seconds=3600.0
    ):
        """直接识别原始文件；未声明此能力的引擎仍由运行时解码后转写。"""
        raise STTError("BACKEND_INCOMPATIBLE", "transcribe_file", "引擎不支持直接文件识别")

    def transcribe_url(
        self, url, language=None, *, word_timestamps=False, task_timeout_seconds=3600.0
    ):
        """识别服务可访问的文件地址，不在通用运行时下载音频。"""
        raise STTError("BACKEND_INCOMPATIBLE", "transcribe_url", "引擎不支持文件地址识别")

    def close(self):
        """释放自有资源，返回 None 表示完成，也可返回 CloseResult。

        延迟释放应及时返回 pending；后续 close 继续确认释放进度。
        ok 和 failed 是最终结果，失败即使没有错误详情也不能视为成功。
        """


def validate_engine(engine):
    """只在注入边界检查契约，不在每个音频块重新探测方法。"""
    capabilities = getattr(engine, "capabilities", None)
    methods = ["prepare", "close"]
    if isinstance(capabilities, STTCapabilities):
        if capabilities.transcription:
            methods.append("transcribe")
        if capabilities.streaming:
            methods.append("create_streaming_session")
        if capabilities.file_transcription:
            methods.append("transcribe_file")
        if capabilities.url_transcription:
            methods.append("transcribe_url")
    if not isinstance(capabilities, STTCapabilities) or any(
        not callable(getattr(engine, method, None)) for method in methods
    ):
        raise STTError("INVALID_CONFIGURATION", "configure_engine", "引擎不符合公开接入契约")
    return capabilities


@dataclass(frozen=True, kw_only=True)
class TTSCapabilities:
    """准备前即可查询的合成能力，不根据模型名字猜测。"""

    default_sample_rate: int
    synthesis: bool = False
    audio_streaming: bool = False
    text_streaming: bool = False
    language_selection: bool = False
    speed_control: bool = False
    instruction_control: bool = False
    sample_rates: tuple[int, ...] | None = None
    voice_cloning: bool = False
    text_timestamps: tuple[str, ...] = ()

    def __post_init__(self):
        if (
            type(self.default_sample_rate) is not int
            or self.default_sample_rate <= 0
            or any(
                type(getattr(self, name)) is not bool
                for name in (
                    "synthesis",
                    "audio_streaming",
                    "text_streaming",
                    "language_selection",
                    "speed_control",
                    "instruction_control",
                    "voice_cloning",
                )
            )
            or (self.text_streaming and not self.audio_streaming)
            or not isinstance(self.text_timestamps, tuple)
            or any(level not in ("sentence", "word") for level in self.text_timestamps)
            or (
                self.sample_rates is not None
                and (
                    not isinstance(self.sample_rates, tuple)
                    or any(type(rate) is not int or rate <= 0 for rate in self.sample_rates)
                    or self.default_sample_rate not in self.sample_rates
                )
            )
        ):
            raise TTSError("INVALID_CONFIGURATION", "configure_engine", "合成能力声明无效")


class TTSAdapter:
    """可继承或按相同契约实现；不支持的方法直接报告能力错误。"""

    @property
    def capabilities(self) -> TTSCapabilities:
        raise NotImplementedError

    def prepare(self):
        """准备自有资源；轻量适配器可直接使用默认实现。"""

    def synthesize(self, text, *, options):
        raise TTSError("CAPABILITY_UNSUPPORTED", "synthesize", "后端不支持完整合成")

    def stream(self, text, *, options):
        raise TTSError("CAPABILITY_UNSUPPORTED", "stream", "后端不支持音频流")

    def open_stream(self, *, options):
        raise TTSError("CAPABILITY_UNSUPPORTED", "open_stream", "后端不支持增量文本")

    def clone_voice(self, reference_audio, *, reference_text=None):
        """从参考录音提取可独立保存的声音数据。"""
        raise TTSError("CAPABILITY_UNSUPPORTED", "clone_voice", "适配器不支持声音克隆")

    def close(self):
        """释放引擎资源；不由单个流释放共享引擎。"""
