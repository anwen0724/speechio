"""可复用的声音数据及文件读写；不持有模型、设备或应用资料。"""

import json
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from speechio._async import check_cancelled
from speechio.audio.decoding import open_audio_source
from speechio.audio.formats import AudioFormat
from speechio.audio.results import AudioResult
from speechio.errors import STTError, TTSError

_MAGIC = b"SPVOICE1"


@dataclass(frozen=True, kw_only=True)
class VoiceInfo:
    """可选音色的说明；id 可直接传给 SynthesisOptions.voice。"""

    id: str
    name: str
    languages: tuple[str, ...] | None = None
    instruction_control: bool | None = None
    text_timestamps: tuple[str, ...] | None = None


@dataclass(frozen=True, kw_only=True)
class Voice:
    """适配器生成的独立声音数据；使用者通过 clone_voice 或 load 获取。"""

    adapter: str
    model: str
    data: bytes
    data_version: int = 1

    def __post_init__(self):
        if (
            not isinstance(self.adapter, str)
            or not self.adapter.strip()
            or not isinstance(self.model, str)
            or not self.model.strip()
            or not isinstance(self.data, bytes)
            or not self.data
            or type(self.data_version) is not int
            or self.data_version < 1
        ):
            raise TTSError("INVALID_INPUT", "voice", "声音身份、数据或格式版本无效")

    def save(self, path, *, overwrite=False) -> Path:
        """保存通用身份说明及适配器数据，默认不覆盖已有文件。"""
        path = Path(path)
        metadata = json.dumps(
            {"adapter": self.adapter, "model": self.model, "data_version": self.data_version},
            ensure_ascii=False,
        ).encode("utf-8")
        try:
            with path.open("wb" if overwrite else "xb") as output:
                output.write(_MAGIC)
                output.write(struct.pack("<I", len(metadata)))
                output.write(metadata)
                output.write(self.data)
        except OSError as exc:
            raise TTSError("VOICE_WRITE_FAILED", "save_voice", "无法保存声音文件") from exc
        return path

    @classmethod
    def load(cls, path):
        """读取数据，不执行文件内代码，也不加载模型或访问网络。"""
        try:
            with Path(path).open("rb") as source:
                if source.read(8) != _MAGIC:
                    raise ValueError("未知声音文件格式")
                size = struct.unpack("<I", source.read(4))[0]
                if not 0 < size <= 65536:
                    raise ValueError("声音说明长度无效")
                metadata = json.loads(source.read(size))
                return cls(
                    adapter=metadata["adapter"],
                    model=metadata["model"],
                    data_version=metadata["data_version"],
                    data=source.read(),
                )
        except (OSError, ValueError, KeyError, TypeError, struct.error, TTSError) as exc:
            raise TTSError("VOICE_READ_FAILED", "load_voice", "声音文件无法读取或格式无效") from exc


def read_reference_audio(reference_audio):
    """解码参考录音并转为单声道 PCM16，保留原始采样率。"""
    if isinstance(reference_audio, AudioResult):
        audio = reference_audio
    else:
        try:
            with open_audio_source(reference_audio) as source:
                rate = source.info.format.sample_rate
                blocks = []
                for block in source.chunks(8192):
                    check_cancelled()
                    values = block.astype(np.float32)
                    if source.info.format.sample_type == "int16":
                        values /= 32768.0
                    mono = values.mean(axis=1)
                    blocks.append(
                        (np.clip(mono, -1, 32767 / 32768) * 32768).astype("<i2").tobytes()
                    )
                audio = AudioResult(b"".join(blocks), AudioFormat(sample_rate=rate))
        except STTError as exc:
            raise TTSError("INVALID_INPUT", "clone_voice", "无法读取参考录音") from exc
    if not audio.data:
        raise TTSError("INVALID_INPUT", "clone_voice", "参考录音没有音频采样")
    return audio
