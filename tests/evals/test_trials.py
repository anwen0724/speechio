"""验证试次失败与取消；原生模型仅在外部边界替换。"""

import threading
import unittest

import numpy as np

from evals.p0.stream_run import run_stream
from evals.p0.whole_run import run_whole
from realtime_asr.core.engines.stream import NativeStream
from tests.core.test_stream import RuntimeRecognizer, RuntimeStream


class EngineBoundary:
    def __init__(self, recognizer=None):
        self.recognizer = recognizer or RuntimeRecognizer()

    def create_streaming_session(self, language):
        return NativeStream(self.recognizer, language)


class TrialTests(unittest.TestCase):
    def test_whole_native_finish_failure_preserves_accepted_body(self):
        class BrokenStream(RuntimeStream):
            def input_finished(self):
                raise RuntimeError("结束失败")

        class BrokenRecognizer(RuntimeRecognizer):
            def create_stream(self):
                return BrokenStream()

        trial = run_whole(EngineBoundary(BrokenRecognizer()), np.ones(10), "zh-CN")
        self.assertEqual(trial.status, "failed")
        self.assertEqual(trial.error["confirmed_range"], [0, 10])
        self.assertIsNone(trial.error["unconfirmed_range"])

    def test_decode_and_finish_failure_stop_trial_and_keep_confirmed_range(self):
        for operation in ("decode", "finish"):
            with self.subTest(operation=operation):

                class BrokenStream(RuntimeStream):
                    def input_finished(self):
                        raise RuntimeError("结束失败")

                class BrokenRecognizer(RuntimeRecognizer):
                    def create_stream(self):
                        return BrokenStream() if operation == "finish" else RuntimeStream()

                    def is_ready(self, stream):
                        return operation == "decode"

                    def decode_stream(self, stream):
                        raise RuntimeError("解码失败")

                trial = run_stream(EngineBoundary(BrokenRecognizer()), np.ones(10), "zh-CN")
                self.assertEqual(trial.status, "failed")
                self.assertEqual(trial.error["operation"], operation)
                self.assertEqual(trial.error["confirmed_range"], [0, 10])
                self.assertIsNone(trial.error["unconfirmed_range"])
                self.assertFalse(
                    any(r["output_kind"] == "stream_final" for r in trial.observations)
                )

    def test_cancel_during_accept_stops_new_native_decode(self):
        cancel = threading.Event()

        class CancellingStream(RuntimeStream):
            def accept_waveform(self, rate, audio):
                super().accept_waveform(rate, audio)
                cancel.set()

        class CancellingRecognizer(RuntimeRecognizer):
            def create_stream(self):
                return CancellingStream()

            def is_ready(self, stream):
                raise RuntimeError("取消后不应开始新解码")

        trial = run_stream(
            EngineBoundary(CancellingRecognizer()), np.ones(10), "zh-CN", cancel=cancel
        )
        self.assertEqual(trial.status, "cancelled")
        self.assertEqual(trial.confirmed_end_sample, 10)
        self.assertEqual(trial.cleanup["status"], "ok")

    def test_cleanup_failure_is_separate_from_original_inference_failure(self):
        class BrokenStream(RuntimeStream):
            def accept_waveform(self, rate, audio):
                raise RuntimeError("接受失败")

            def close(self):
                raise RuntimeError("关闭也失败")

        class BrokenRecognizer(RuntimeRecognizer):
            def create_stream(self):
                return BrokenStream()

        trial = run_stream(EngineBoundary(BrokenRecognizer()), np.ones(10), "zh-CN")
        self.assertEqual(trial.error["operation"], "accept_audio")
        self.assertEqual(trial.cleanup["status"], "failed")
        self.assertEqual(trial.cleanup["errors"][0]["error_code"], "CLEANUP_FAILED")

    def test_final_range_includes_short_last_block(self):
        trial = run_stream(EngineBoundary(), np.zeros(513, dtype=np.float32), "zh-CN")
        finals = [r for r in trial.observations if r["output_kind"] == "stream_final"]
        self.assertEqual(len(finals), 1)
        self.assertEqual(finals[0]["accepted_range"], [0, 513])
        self.assertEqual(trial.observations[-2]["block_range"], [512, 513])

    def test_cancelled_trial_has_no_successful_final(self):
        cancel = threading.Event()
        cancel.set()
        trial = run_stream(EngineBoundary(), np.ones(10, dtype=np.float32), "zh-CN", cancel=cancel)
        self.assertEqual(trial.status, "cancelled")
        self.assertFalse(any(r["output_kind"] == "stream_final" for r in trial.observations))
        self.assertIn("close_completed_ns", trial.timestamps)

    def test_native_accept_failure_keeps_confirmed_and_unconfirmed_ranges(self):
        class FailingStream(RuntimeStream):
            def accept_waveform(self, rate, audio):
                if self.samples:
                    raise RuntimeError("原生接受失败")
                super().accept_waveform(rate, audio)

        class FailingRecognizer(RuntimeRecognizer):
            def create_stream(self):
                return FailingStream()

        trial = run_stream(
            EngineBoundary(FailingRecognizer()), np.zeros(4), "zh-CN", chunk_samples=2
        )
        self.assertEqual(trial.status, "failed")
        self.assertEqual(trial.error["confirmed_range"], [0, 2])
        self.assertEqual(trial.error["unconfirmed_range"], [2, 4])
        self.assertEqual(trial.cleanup["status"], "ok")
