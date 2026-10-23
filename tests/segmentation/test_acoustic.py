"""声学信号从公开 PCM 接口观察，不替换检测器或检查私有历史。"""

import unittest

import numpy as np

from speechio.stt.segmentation.acoustic import AcousticBoundaryDetector


class AcousticTests(unittest.TestCase):
    def test_voiced_valley_is_positioned_and_chunk_independent(self):
        voiced = np.sin(2 * np.pi * 180 * np.arange(3200) / 16000)
        audio = np.concatenate((np.zeros(1600), voiced * 0.2, voiced[:320] * 0.005, voiced * 0.2))
        pcm = (audio * 32767).astype("<i2").tobytes()
        whole = AcousticBoundaryDetector().feed(pcm)
        detector = AcousticBoundaryDetector()
        chunks = []
        for offset in range(0, len(pcm), 222):
            chunks.extend(detector.feed(pcm[offset : offset + 222]))
        self.assertEqual(whole, chunks)
        self.assertTrue(any(4800 <= event.boundary_sample <= 5120 for event in whole))
        self.assertEqual(AcousticBoundaryDetector().feed(bytes(32000)), [])
