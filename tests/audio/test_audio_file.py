"""实际写入 WAV 与 FLAC，从文件入口验证解码和转换组合。"""

import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import av
import numpy as np
import soundfile as sf

from speechio import Transcriber
from speechio.audio.files import audio_file_chunks, read_audio_file, read_audio_info
from speechio.errors import STTError
from tests.audio.encoded_samples import encoded_audio, video_with_optional_audio
from tests.runtime.test_engine_injection import RecordingEngine


class AudioFileTests(unittest.TestCase):
    def test_decoder_failure_after_first_frame_is_not_a_successful_prefix(self):
        real_open = av.open

        @contextmanager
        def failing_decoder(file, **options):
            with real_open(file, **options) as container:

                def decode(stream):
                    yield next(container.decode(stream))
                    raise OSError("解码器读取失败")

                yield SimpleNamespace(streams=container.streams, decode=decode)

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "input.m4a"
            path.write_bytes(encoded_audio())
            # 仅在 FFmpeg 解码边界注入读取失败，其他处理仍执行真实逻辑。
            with patch("av.open", side_effect=failing_decoder):
                with self.assertRaises(STTError) as caught:
                    read_audio_file(path)
            self.assertEqual(caught.exception.code, "AUDIO_READ_FAILED")
            path.unlink()

    def test_compressed_limit_uses_decoded_length_when_track_duration_is_unknown(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "input.webm"
            path.write_bytes(encoded_audio("webm", "libopus"))
            self.assertIsNone(read_audio_info(path).duration_seconds)
            for mode in ("batch", "chunks"):
                with self.subTest(mode=mode), self.assertRaises(STTError) as caught:
                    if mode == "batch":
                        read_audio_file(path, max_audio_seconds=0.1)
                    else:
                        list(audio_file_chunks(path, max_audio_seconds=0.1))
                self.assertEqual(caught.exception.code, "AUDIO_CAPACITY_EXCEEDED")
            path.unlink()

    def test_video_uses_first_audio_track_and_reports_missing_audio(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "input.mp4"
            path.write_bytes(video_with_optional_audio(audio=True))
            output = np.frombuffer(read_audio_file(path), dtype="<i2")
            self.assertEqual(len(output), 8000)
            self.assertLessEqual(np.max(np.abs(output[100:-100].astype(int) - 8000)), 2)
            path.write_bytes(video_with_optional_audio())
            with self.assertRaises(STTError) as caught:
                read_audio_file(path)
            self.assertEqual(caught.exception.code, "AUDIO_TRACK_MISSING")
            path.unlink()

    def test_early_close_and_invalid_file_release_the_input(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "input.m4a"
            path.write_bytes(encoded_audio())
            chunks = audio_file_chunks(path)
            next(chunks)
            chunks.close()
            # Windows 上未关闭的文件不能删除。
            path.unlink()
            path.write_bytes(b"not an audio file")
            with self.assertRaises(STTError) as caught:
                read_audio_file(path)
            self.assertEqual(caught.exception.code, "AUDIO_READ_FAILED")
            path.unlink()

    def test_common_compressed_files_reach_model_complete_in_batch_and_chunks(self):
        with tempfile.TemporaryDirectory() as folder:
            for container, codec in (
                ("mp4", "aac"),
                ("mp4", "alac"),
                ("adts", "aac"),
                ("webm", "libopus"),
            ):
                with self.subTest(container=container, codec=codec):
                    # Web 保存后没有原后缀，解码必须识别文件内容。
                    path = Path(folder) / "upload.audio"
                    path.write_bytes(encoded_audio(container, codec))
                    engine = RecordingEngine()
                    with Transcriber.from_adapter(adapter=engine) as transcriber:
                        transcriber.transcribe_file(path)
                        chunks = audio_file_chunks(path, chunk_frames=701)
                        try:
                            format, first = next(chunks)

                            def audio():
                                yield first
                                for _, chunk in chunks:
                                    self.assertLessEqual(len(chunk), 701)
                                    yield chunk

                            transcriber.transcribe_chunks(audio(), format=format)
                        finally:
                            chunks.close()
                    batch, streamed = engine.inputs[0][0], engine.inputs[1][0]
                    np.testing.assert_array_equal(batch, streamed)
                    # AAC 可有编码填充；不接受漏掉整帧或最后一段声音。
                    self.assertGreaterEqual(len(batch), 8000)
                    self.assertLessEqual(len(batch), 8700)
                    self.assertGreater(np.sqrt(np.mean(batch[7200:7900] ** 2)), 0.1)

    def test_pcm16_wav_and_flac_keep_exact_samples(self):
        source = np.array([-32768, -1, 0, 1, 32767], np.int16)
        with tempfile.TemporaryDirectory() as folder:
            for suffix in ("wav", "flac"):
                with self.subTest(suffix=suffix):
                    path = Path(folder) / f"samples.{suffix}"
                    sf.write(path, source, 16000, subtype="PCM_16")
                    self.assertEqual(read_audio_file(path), source.astype("<i2").tobytes())

    def test_stereo_nonstandard_pcm_and_float_wav_use_header_format(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "stereo.wav"
            sf.write(path, np.full((48000, 2), 1000, np.int16), 48000, subtype="PCM_16")
            output = np.frombuffer(read_audio_file(path), dtype="<i2")
            self.assertEqual(len(output), 16000)
            self.assertLessEqual(np.max(np.abs(output[100:-100].astype(int) - 1000)), 1)
            sf.write(path, np.array([-1, -0.5, 0, 0.5, 1], np.float32), 16000, subtype="FLOAT")
            self.assertEqual(
                read_audio_file(path),
                np.array([-32767, -16383, 0, 16383, 32767], dtype="<i2").tobytes(),
            )

    def test_read_failure_closes_owned_file_without_successful_prefix(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "input.wav"
            sf.write(path, np.arange(100, dtype=np.int16), 16000, subtype="PCM_16")
            source = sf.SoundFile(path)
            with (
                patch("soundfile.SoundFile", return_value=source),
                patch.object(
                    source, "read", side_effect=[np.ones((1, 1), np.int16), OSError("读取失败")]
                ),
            ):
                with self.assertRaises(STTError) as caught:
                    read_audio_file(path)
            self.assertEqual(caught.exception.code, "AUDIO_READ_FAILED")
            self.assertTrue(source.closed)
