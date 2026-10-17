"""实际写入 WAV 与 FLAC，从文件入口验证解码和转换组合。"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf

from realtime_asr.core.audio.files import read_audio_file
from realtime_asr.core.errors import ASRError


class AudioFileTests(unittest.TestCase):
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
                with self.assertRaises(ASRError) as caught:
                    read_audio_file(path)
            self.assertEqual(caught.exception.code, "AUDIO_READ_FAILED")
            self.assertTrue(source.closed)
