"""网络消息的轻量编解码；不依赖服务框架、模型或设备。"""

import json
import struct

PROTOCOL_VERSION = 1
MAX_METADATA_BYTES = 64 * 1024


def encode_audio_packet(metadata, audio):
    """编码完整二进制消息；语义字段由各自接入边界检查。"""
    if not isinstance(metadata, dict) or not isinstance(audio, bytes):
        raise ValueError("元数据应为对象，音频应为 bytes")
    header = json.dumps(metadata, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    if len(header) > MAX_METADATA_BYTES:
        raise ValueError("音频元数据过大")
    return struct.pack("<I", len(header)) + header + audio


def decode_audio_packet(packet, *, max_bytes=1024 * 1024, max_metadata=MAX_METADATA_BYTES):
    """检查包边界后解析 UTF-8 元数据，不修补或补齐丢失字节。"""
    if len(packet) < 4 or len(packet) > max_bytes:
        raise ValueError("音频包长度无效")
    size = struct.unpack("<I", packet[:4])[0]
    if size > max_metadata or len(packet) < 4 + size:
        raise ValueError("音频元数据长度无效")
    metadata = json.loads(packet[4 : 4 + size].decode("utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError("音频元数据应为对象")
    return metadata, packet[4 + size :]
