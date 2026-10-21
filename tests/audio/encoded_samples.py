"""生成短音频测试文件；编码器只用于准备真实输入，不替代解码。"""

import io

import av
import numpy as np


def encoded_audio(container="mp4", codec="aac") -> bytes:
    """生成半秒 440 Hz 单声道音频，尾部也有声音以检查尾块丢失。"""
    buffer = io.BytesIO()
    with av.open(buffer, "w", format=container) as output:
        stream = output.add_stream(codec, rate=48000)
        stream.layout = "mono"
        samples = (8000 * np.sin(2 * np.pi * 440 * np.arange(24000) / 48000)).astype(np.int16)
        frame = av.AudioFrame.from_ndarray(samples.reshape(1, -1), format="s16", layout="mono")
        frame.sample_rate = 48000
        for packet in stream.encode(frame):
            output.mux(packet)
        for packet in stream.encode(None):
            output.mux(packet)
    return buffer.getvalue()


def video_with_optional_audio(*, audio=False) -> bytes:
    """生成一个短视频；可附带两条内容不同的音轨来验证音轨选择。"""
    buffer = io.BytesIO()
    with av.open(buffer, "w", format="mp4") as output:
        video = output.add_stream("mpeg4", rate=25)
        video.width = video.height = 16
        video.pix_fmt = "yuv420p"
        tracks = []
        if audio:
            for _ in range(2):
                track = output.add_stream("alac", rate=48000)
                track.layout = "mono"
                tracks.append(track)
        frame = av.VideoFrame.from_ndarray(np.zeros((16, 16, 3), np.uint8), format="rgb24")
        for packet in video.encode(frame):
            output.mux(packet)
        for packet in video.encode(None):
            output.mux(packet)
        for track, value in zip(tracks, (8000, -8000)):
            frame = av.AudioFrame.from_ndarray(
                np.full((1, 24000), value, np.int16), format="s16", layout="mono"
            )
            frame.sample_rate = 48000
            for packet in track.encode(frame):
                output.mux(packet)
            for packet in track.encode(None):
                output.mux(packet)
    return buffer.getvalue()
