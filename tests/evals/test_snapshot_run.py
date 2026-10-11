"""替换模型外部边界，验证真实调度与结果归属，不作为识别效果证据。"""

import time
import unittest

import numpy as np

from evals.p0.cancellation import CancellationSource
from evals.p0.snapshot_run import run_snapshots
from evals.p0.whole_run import run_whole
from realtime_asr.core.results import RecognitionResult


class SnapshotRunTests(unittest.TestCase):
    def test_short_audio_has_only_independent_complete_call(self):
        class Model:
            def transcribe(self, audio, **parameters):
                self.call = (audio.size, parameters["role"])
                return RecognitionResult(text="完整")

        model = Model()
        trial = run_snapshots(model, np.zeros(160, np.float32), "zh")
        self.assertEqual(model.call, (160, "final"))
        self.assertEqual([r["output_kind"] for r in trial.observations], ["snapshot_final"])

    def test_busy_snapshot_stays_partial_after_audio_end_and_pending_is_discarded(self):
        class Model:
            def transcribe(self, audio, **parameters):
                if parameters["role"] == "snapshot":
                    time.sleep(0.4)
                return RecognitionResult(text=str(audio.size))

        trial = run_snapshots(Model(), np.zeros(8000, np.float32), "zh")
        results = [item for item in trial.observations if item["result"] is not None]
        self.assertEqual(
            [item["output_kind"] for item in results], ["snapshot_partial", "snapshot_final"]
        )
        self.assertLess(results[0]["accepted_range"][1], 8000)
        self.assertEqual(results[-1]["accepted_range"][1], 8000)
        self.assertTrue(
            any(item.get("reason") == "discarded_at_end" for item in trial.observations)
        )
        self.assertEqual([call["parameter_role"] for call in trial.calls], ["snapshot", "final"])

    def test_whole_return_after_cancellation_is_late_and_keeps_request_time(self):
        cancel = CancellationSource()

        class Model:
            def transcribe(self, audio, **parameters):
                cancel.set()
                time.sleep(0.01)
                return RecognitionResult(text="迟到")

        trial = run_whole(Model(), np.zeros(160), "zh", cancel=cancel)
        self.assertEqual(trial.status, "cancelled")
        self.assertEqual(trial.observations[0]["output_kind"], "late_result")
        self.assertLess(trial.timestamps["cancel_requested_ns"], trial.calls[0]["completed_ns"])
