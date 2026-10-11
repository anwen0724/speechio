"""用明确时间点检查快照机会，不依赖线程等待猜测时序。"""

import unittest

from evals.p0.snapshots import SnapshotSchedule


class SnapshotTests(unittest.TestCase):
    def test_busy_keeps_latest_prefix_and_end_discards_pending_request(self):
        schedule = SnapshotSchedule(16000)
        schedule.advance(200_000_000)
        first = schedule.start(250_000_000)
        self.assertEqual(first["accepted_range"], [0, 4000])
        schedule.advance(800_000_000)
        skipped = [item for item in schedule.requests if item["status"] == "skipped"]
        self.assertEqual([item["requested_end_sample"] for item in skipped], [6400, 9600])
        schedule.advance(1_000_000_000)
        self.assertTrue(schedule.ended)
        self.assertIsNone(schedule.start(1_000_000_000))
        self.assertEqual(schedule.requests[-1]["reason"], "discarded_at_end")

    def test_short_audio_and_end_at_opportunity_do_not_start_partial(self):
        for count in (100, 3200):
            schedule = SnapshotSchedule(count)
            schedule.advance(round(count * 1_000_000_000 / 16000))
            self.assertIsNone(schedule.start(300_000_000))
            self.assertEqual(schedule.requests, [])
