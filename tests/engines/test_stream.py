"""原生运行库是外部边界；测试执行真实适配器状态规则。"""

import unittest
from types import SimpleNamespace

import numpy as np

from speechio.adapters.local.stream import NativeStream
from speechio.errors import STTError


class RuntimeStream:
    """小型外部运行库替身，接收样本并返回可区分的正文。"""

    def __init__(self):
        self.samples = []
        self.ended = False

    def set_option(self, name, value):
        self.language = value

    def accept_waveform(self, rate, audio):
        self.samples.extend(audio.tolist())

    def input_finished(self):
        if self.ended:
            raise RuntimeError("原生结束不允许重发")
        self.ended = True


class RuntimeRecognizer:
    def create_stream(self):
        return RuntimeStream()

    def is_ready(self, stream):
        return False

    def get_result_all(self, stream):
        value = "结束" if stream.ended else "部分"
        return SimpleNamespace(
            text=f"<zh-CN> {value}{len(stream.samples)}" if stream.samples else ""
        )


class StreamTests(unittest.TestCase):
    def test_rejected_audio_does_not_advance_range_and_legal_audio_can_continue(self):
        stream = NativeStream(RuntimeRecognizer(), language="zh-CN")
        for audio, rate in [(np.zeros((2, 2)), None), ([1], "16000"), ([1], 16000.0)]:
            with self.subTest(rate=rate, shape=np.shape(audio)):
                with self.assertRaises(STTError):
                    stream.accept_audio(audio, sample_rate=rate)
                self.assertEqual(stream.accepted_samples, 0)
        stream.accept_audio([])
        stream.accept_audio([[0.2], [0.3]])
        stream.decode()
        self.assertEqual(stream.accepted_samples, 2)
        self.assertEqual(stream.get_result().text, "部分2")

    def test_finish_is_repeatable_and_closed_final_cache_is_preserved(self):
        stream = NativeStream(RuntimeRecognizer(), language="zh-CN")
        stream.accept_audio([0.1])
        final = stream.finish()
        self.assertEqual(final.text, "结束1")
        self.assertEqual(final.reported_language, "zh-CN")
        self.assertEqual(final.language_source, "detected")
        self.assertIsNone(final.language_probability)
        self.assertEqual(stream.finish(), final)
        with self.assertRaises(STTError) as caught:
            stream.accept_audio([])
        self.assertEqual(caught.exception.code, "STREAM_FINISHED")
        stream.close()
        stream.close()
        stream.decode()
        self.assertEqual(stream.finish(), final)

    def test_reset_has_new_identity_and_direct_close_does_not_create_final(self):
        stream = NativeStream(RuntimeRecognizer())
        stream.accept_audio([0.1, 0.2])
        stream.decode()
        old = stream.stream_id
        stream.reset()
        self.assertNotEqual(old, stream.stream_id)
        self.assertEqual(stream.accepted_samples, 0)
        self.assertEqual(stream.get_result().text, "")
        stream.accept_audio([0.1])
        stream.decode()
        partial = stream.get_result()
        stream.close()
        self.assertEqual(stream.get_result(), partial)
        self.assertFalse(stream.has_final_result)
        with self.assertRaises(STTError) as caught:
            stream.reset()
        self.assertEqual(caught.exception.code, "STREAM_CLOSED")

    def test_result_read_failure_preserves_operation_and_last_successful_cache(self):
        class FailingRecognizer(RuntimeRecognizer):
            fail_read = False

            def get_result_all(self, stream):
                if self.fail_read:
                    raise RuntimeError("原生结果读取失败")
                return super().get_result_all(stream)

        recognizer = FailingRecognizer()
        stream = NativeStream(recognizer, language="zh-CN")
        stream.accept_audio([0.1])
        stream.decode()
        partial = stream.get_result()
        recognizer.fail_read = True
        with self.assertRaises(STTError) as caught:
            stream.decode()
        self.assertEqual(caught.exception.operation, "get_result")
        self.assertEqual(stream.get_result(), partial)
        with self.assertRaises(STTError):
            stream.accept_audio([0.2])
        stream.close()
