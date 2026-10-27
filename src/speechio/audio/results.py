"""PCM 合成结果；不包含播放或应用存储策略。"""

import wave
from dataclasses import dataclass
from pathlib import Path

from speechio.audio.formats import AudioFormat
from speechio.errors import TTSError
from speechio.tts.timing import TextTiming


def validate_pcm(data, format):
    """合成输出边界只接受完整的小端单声道 PCM16 采样。"""
    if (
        not isinstance(format, AudioFormat)
        or format.channels != 1
        or format.sample_type != "int16"
        or not isinstance(data, bytes)
        or len(data) % 2
    ):
        raise TTSError("PROTOCOL_ERROR", "audio_result", "需要完整的单声道 PCM16 采样")


@dataclass(frozen=True)
class AudioChunk:
    """一块连续音频；按交付顺序拼接即为完整结果。"""

    data: bytes
    format: AudioFormat

    def __post_init__(self):
        validate_pcm(self.data, self.format)


@dataclass(frozen=True)
class AudioResult:
    """一次成功合成的完整音频，可保存为 WAV。"""

    data: bytes
    format: AudioFormat
    text_timings: tuple[TextTiming, ...] = ()

    def __post_init__(self):
        validate_pcm(self.data, self.format)

    @property
    def duration_seconds(self):
        return len(self.data) / (2 * self.format.sample_rate)

    def save(self, path, *, overwrite=False) -> Path:
        """以独占创建避免覆盖已有文件；由调用方创建目标目录。"""
        path = Path(path)
        if path.suffix.lower() != ".wav":
            raise TTSError("INVALID_INPUT", "save_audio", "首版只支持保存 .wav 文件")
        try:
            with path.open("wb" if overwrite else "xb") as file:
                with wave.open(file, "wb") as output:
                    output.setnchannels(1)
                    output.setsampwidth(2)
                    output.setframerate(self.format.sample_rate)
                    output.writeframes(self.data)
        except OSError:
            raise TTSError("AUDIO_WRITE_FAILED", "save_audio", "无法创建或写入 WAV 文件") from None
        return path
