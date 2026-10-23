"""通过标准音频操作验证格式、解码值与失败分类。"""

import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

from speechio.audio.files import read_wav
from speechio.errors import STTError
from speechio.stt.input import as_audio


class AudioTests(unittest.TestCase):
    def test_empty_corrupt_and_truncated_wavs_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "input.wav"
            path.write_bytes(b"not-a-wav")
            with self.assertRaises(STTError) as caught:
                read_wav(path)
            self.assertEqual(caught.exception.code, "INVALID_AUDIO")
            for count in (0, 10):
                with wave.open(str(path), "wb") as wav:
                    wav.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                    wav.writeframes(bytes(count * 2))
                if count:
                    path.write_bytes(path.read_bytes()[:-1])
                with self.assertRaises(STTError) as caught:
                    read_wav(path)
                self.assertEqual(caught.exception.code, "INVALID_AUDIO")

    def test_pcm16_values_are_decoded_without_peak_normalization(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "input.wav"
            with wave.open(str(path), "wb") as wav:
                wav.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                wav.writeframes(np.array([-32768, 0, 16384, 32767], dtype="<i2").tobytes())
            audio = read_wav(path)
        np.testing.assert_array_equal(audio, [-1, 0, 0.5, 32767 / 32768])
        self.assertEqual(audio.dtype, np.float32)

    def test_stereo_and_empty_whole_input_fail_but_empty_stream_block_is_valid(self):
        for audio in [np.zeros((8, 2)), []]:
            with self.subTest(shape=np.shape(audio)):
                with self.assertRaises(STTError) as caught:
                    as_audio(audio)
                self.assertEqual(caught.exception.code, "INVALID_AUDIO")
        self.assertEqual(as_audio([], allow_empty=True).size, 0)
        self.assertEqual(as_audio([[0.2], [0.3]], allow_column=True).shape, (2,))

    def test_invalid_wav_format_fails_before_recognition(self):
        with tempfile.TemporaryDirectory() as folder:
            for rate, channels, width in [(8000, 1, 2), (16000, 2, 2), (16000, 1, 1)]:
                path = Path(folder) / "bad.wav"
                with wave.open(str(path), "wb") as wav:
                    wav.setparams((channels, width, rate, 0, "NONE", "not compressed"))
                    wav.writeframes(bytes(8 * channels * width))
                with self.subTest(rate=rate, channels=channels, width=width):
                    with self.assertRaises(STTError):
                        read_wav(path)
