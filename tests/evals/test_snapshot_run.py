"""控制时钟和外部模型，验证真实调度的结果与跳过原因。"""

import unittest
from threading import Event

import numpy as np

from evals.p0.cancellation import CancellationSource
from evals.p0.snapshot_run import run_snapshots
from evals.p0.trial import Trial
from evals.p0.whole_run import run_whole
from realtime_asr.core.results import RecognitionResult


class ReplayClock:
    """测试只推进回放时间，不等待对应的真实时长。"""

    def __init__(self):
        self.elapsed_ns = 0

    def __call__(self):
        return self.elapsed_ns


class ReplayEvent(Event):
    """替换回放等待边界；模型线程仍使用真实的同步事件。"""

    def __init__(self, clock, on_wait=None):
        super().__init__()
        self.clock = clock
        self.on_wait = on_wait

    def wait(self, timeout=None):
        if not self.is_set():
            if self.on_wait is None:
                self.clock.elapsed_ns += round(timeout * 1_000_000_000)
            else:
                self.on_wait()
        return self.is_set()


class SnapshotRunTests(unittest.TestCase):
    def test_short_audio_and_end_at_opportunity_have_only_complete_result(self):
        class Model:
            def transcribe(self, audio, **parameters):
                return RecognitionResult(text=str(audio.size))

        # 短于一次快照与恰好结束于快照机会，分别验证两条结束规则。
        for count in (160, 3200):
            with self.subTest(sample_count=count):
                clock = ReplayClock()
                trial = run_snapshots(
                    Model(),
                    np.zeros(count, np.float32),
                    "zh",
                    cancel=ReplayEvent(clock),
                    trial=Trial(count, clock=clock),
                )
                self.assertEqual(trial.status, "ok")
                self.assertEqual(
                    [item["output_kind"] for item in trial.observations], ["snapshot_final"]
                )
                self.assertEqual(trial.observations[0]["accepted_range"], [0, count])
                self.assertEqual(trial.observations[0]["result"]["text"], str(count))
                self.assertEqual([call["parameter_role"] for call in trial.calls], ["final"])

    def test_busy_snapshot_keeps_prefix_and_records_replaced_and_discarded_requests(self):
        clock = ReplayClock()
        started, release = Event(), Event()

        class Model:
            def transcribe(self, audio, **parameters):
                if parameters["role"] == "snapshot":
                    started.set()
                    if not release.wait(5):
                        raise RuntimeError("测试未释放模型调用")
                return RecognitionResult(text=str(audio.size))

        def advance_replay():
            if clock.elapsed_ns < 200_000_000:
                clock.elapsed_ns = 200_000_000
                return
            # 先确认首个前缀进入模型，再推进至输入结束；超时仅防止测试挂起。
            try:
                self.assertTrue(started.wait(5), "模型调用没有开始")
                clock.elapsed_ns = 1_000_000_000
            finally:
                release.set()

        trial = run_snapshots(
            Model(),
            np.zeros(16000, np.float32),
            "zh",
            cancel=ReplayEvent(clock, advance_replay),
            trial=Trial(16000, clock=clock),
        )
        self.assertEqual(trial.status, "ok")
        results = [item for item in trial.observations if item["result"] is not None]
        self.assertEqual(
            [item["output_kind"] for item in results], ["snapshot_partial", "snapshot_final"]
        )
        self.assertEqual(results[0]["accepted_range"], [0, 3200])
        self.assertEqual(results[0]["result"]["text"], "3200")
        self.assertEqual(results[-1]["accepted_range"], [0, 16000])
        self.assertEqual(results[-1]["result"]["text"], "16000")
        skipped = [item for item in trial.observations if item["status"] == "skipped"]
        self.assertEqual(
            [(item["requested_end_sample"], item["reason"]) for item in skipped],
            [(6400, "superseded"), (9600, "superseded"), (12800, "discarded_at_end")],
        )
        self.assertEqual([call["parameter_role"] for call in trial.calls], ["snapshot", "final"])

    def test_whole_return_after_cancellation_is_late_and_keeps_request_time(self):
        cancel = CancellationSource()

        class Model:
            def transcribe(self, audio, **parameters):
                cancel.set()
                return RecognitionResult(text="迟到")

        trial = run_whole(Model(), np.zeros(160), "zh", cancel=cancel)
        self.assertEqual(trial.status, "cancelled")
        self.assertEqual(trial.observations[0]["output_kind"], "late_result")
        self.assertEqual(
            trial.timestamps["cancel_requested_ns"], cancel.requested_at_ns - trial.origin_ns
        )
        self.assertLessEqual(
            trial.timestamps["cancel_requested_ns"], trial.calls[0]["completed_ns"]
        )
