"""公开关闭入口保留适配器的未完成和失败结果。"""

import unittest
from threading import Event

from speechio import RealtimeConfig, STTError, Transcriber
from speechio.resources import CloseResult
from speechio.stt.runtime.asr import STTRuntime
from tests.runtime.test_engine_injection import RecordingEngine


class ClosingAdapter(RecordingEngine):
    def __init__(self, status="pending", *, prepare_error=False):
        super().__init__()
        self.status = status
        self.released = Event()
        self.prepare_error = prepare_error

    def prepare(self):
        if self.prepare_error:
            raise STTError("MODEL_LOAD", "prepare", "original preparation failure")

    def close(self):
        return CloseResult(status="ok" if self.released.is_set() else self.status)


class CloseStatusTests(unittest.TestCase):
    def test_transcriber_pending_close_can_complete_and_rejects_new_work(self):
        adapter = ClosingAdapter()
        transcriber = Transcriber.from_adapter(adapter)
        self.assertEqual(transcriber.close(timeout=0).status, "pending")
        self.assertEqual(transcriber.state, "closing")
        with self.assertRaises(STTError):
            transcriber.prepare()
        adapter.released.set()
        self.assertEqual(transcriber.close().status, "ok")
        self.assertEqual(transcriber.close().status, "ok")

    def test_failed_close_without_details_is_reported_by_context_manager(self):
        transcriber = Transcriber.from_adapter(ClosingAdapter("failed"))
        with self.assertRaises(STTError) as caught:
            with transcriber:
                pass
        self.assertEqual(caught.exception.code, "CLEANUP_FAILED")
        self.assertEqual(transcriber.close().status, "failed")

    def test_runtime_keeps_failure_while_other_adapter_is_pending(self):
        for prepared in (False, True):
            with self.subTest(prepared=prepared):
                pending = ClosingAdapter()
                runtime = STTRuntime.from_adapter(
                    ClosingAdapter("failed"),
                    realtime=RealtimeConfig.from_adapter(pending),
                )
                try:
                    if prepared:
                        runtime.prepare()
                    result = runtime.close(timeout=0.02)
                    self.assertEqual(result.status, "pending")
                    self.assertTrue(result.errors)
                    self.assertEqual(runtime.state, "closing")
                finally:
                    pending.released.set()
                    final = runtime.close(timeout=1)
                self.assertEqual(final.status, "failed")
                self.assertTrue(final.errors)

    def test_preparation_failure_keeps_pending_cleanup_and_original_error(self):
        for runtime_type in (Transcriber, STTRuntime):
            with self.subTest(runtime_type=runtime_type):
                adapter = ClosingAdapter(prepare_error=True)
                owner = runtime_type.from_adapter(adapter)
                with self.assertRaises(STTError) as caught:
                    owner.prepare()
                self.assertEqual(caught.exception.message, "original preparation failure")
                self.assertEqual(owner.state, "closing")
                adapter.released.set()
                self.assertEqual(owner.close().status, "ok")

    def test_body_error_is_not_replaced_by_cleanup_failure(self):
        with self.assertRaisesRegex(ValueError, "body failure"):
            with Transcriber.from_adapter(ClosingAdapter("failed")):
                raise ValueError("body failure")
