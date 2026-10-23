"""后台意外退出必须交付失败并释放资源；仅在目标故障处注入异常。"""

import unittest
from threading import Event
from unittest.mock import patch

from speechio import STTError, Transcript
from speechio.stt.runtime.asr import STTRuntime
from tests.runtime.test_engine_injection import RecordingEngine
from tests.test_api import SpeechDetector


class BrokenMetadata:
    """模拟提供方返回不能复制的元数据，触发推理后的整理异常。"""

    def __deepcopy__(self, memo):
        raise RuntimeError("metadata copy failed")


class BlockingEngine(RecordingEngine):
    """第二次调用可控暂停，让后续任务先排队；不依赖固定等待时间。"""

    def __init__(self, *, broken_metadata=False):
        super().__init__()
        self.broken_metadata = broken_metadata
        self.entered = Event()
        self.release = Event()

    def transcribe(self, *args, **kwargs):
        result = super().transcribe(*args, **kwargs)
        if len(self.inputs) == 2:
            self.entered.set()
            if not self.release.wait(5):
                raise TimeoutError("测试未释放模型")
            if self.broken_metadata:
                return Transcript(text="result", backend_metadata={"provider": BrokenMetadata()})
        return result


class WorkerFailureTests(unittest.TestCase):
    def prepare_queue(self, *, broken_metadata=False):
        engine = BlockingEngine(broken_metadata=broken_metadata)
        runtime = STTRuntime.from_adapter(adapter=engine)
        self.addCleanup(runtime.close)
        self.addCleanup(engine.release.set)
        runtime.prepare()
        session = runtime.create_session()
        self.submit(session)
        self.assertEqual(session.next_event(timeout=2).kind, "final")
        self.assertEqual(session.next_event(timeout=2).kind, "completed")
        second = self.submit(session)
        self.assertTrue(engine.entered.wait(2))
        third = self.submit(session)
        return engine, runtime, session, second, third

    @staticmethod
    def submit(session):
        turn = session.start()
        session.feed_audio(b"\x01\x00" * 1600)
        session.end_turn()
        return turn

    def assert_failed_shutdown(self, runtime, engine):
        result = runtime.close(timeout=2)
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.errors[0].code, "RUNTIME_WORKER_FAILED")
        self.assertEqual(runtime.state, "closed")
        self.assertEqual(runtime.capacity.tasks, 0)
        self.assertEqual(runtime.capacity.task_audio_samples, 0)
        self.assertEqual(engine.closes, 1)
        self.assertIs(runtime.close(), result)
        with self.assertRaises(STTError):
            runtime.create_session()

    def test_unexpected_execution_error_fails_running_and_queued_tasks(self):
        engine, runtime, session, second, third = self.prepare_queue(broken_metadata=True)
        engine.release.set()
        for turn in (second, third):
            event = session.next_event(timeout=2)
            self.assertIsNotNone(event)
            self.assertEqual((event.kind, event.turn), ("failed", turn))
            self.assertEqual(event.error.code, "RUNTIME_WORKER_FAILED")
        with self.assertRaises(STTError):
            session.start()
        self.assert_failed_shutdown(runtime, engine)

    def test_delivery_error_preserves_success_and_fails_remaining_queue(self):
        engine, runtime, session, second, third = self.prepare_queue()
        deliver = session.accept_completion
        fail_session = session.fail_runtime

        def fail_after_delivery(*args):
            deliver(*args)
            raise RuntimeError("delivery bookkeeping failed")

        def fail_after_notification(error):
            fail_session(error)
            raise RuntimeError("notification cleanup failed")

        with (
            patch.object(session, "accept_completion", side_effect=fail_after_delivery),
            patch.object(session, "fail_runtime", side_effect=fail_after_notification),
        ):
            engine.release.set()
            events = [session.next_event(timeout=2) for _ in range(3)]
            self.assertTrue(all(event is not None for event in events))
            self.assertEqual([event.kind for event in events], ["final", "completed", "failed"])
            self.assertEqual([event.turn for event in events], [second, second, third])
            self.assert_failed_shutdown(runtime, engine)
            self.assertTrue(
                any(error.operation == "fail_session" for error in runtime.close().errors)
            )

    def test_automatic_session_delivers_failure_and_stops_waiting(self):
        entered, release = Event(), Event()

        class Engine(RecordingEngine):
            def transcribe(self, *args, **kwargs):
                entered.set()
                if not release.wait(5):
                    raise TimeoutError("测试未释放模型")
                return Transcript(text="result", backend_metadata={"bad": BrokenMetadata()})

        engine = Engine()
        runtime = STTRuntime.from_adapter(adapter=engine)
        self.addCleanup(runtime.close)
        self.addCleanup(release.set)
        runtime.prepare()
        session = runtime.create_auto_session_from_detector(detector_factory=SpeechDetector)
        session.feed_audio(b"\xd0\x07" * 16000)
        session.end_turn()
        self.assertTrue(entered.wait(2))
        # 模型处理上一段时，自动采集仍在接收尚未封口的下一段。
        session.feed_audio(b"\xd0\x07" * 1600)
        release.set()
        event = session.next_event(timeout=2)
        self.assertIsNotNone(event)
        self.assertEqual(event.kind, "failed")
        self.assertEqual(event.error.code, "RUNTIME_WORKER_FAILED")
        self.assertEqual(session.input_state, "failed")
        while session.pending_segments:
            self.assertEqual(session.next_event(timeout=2).kind, "failed")
        with self.assertRaises(STTError) as caught:
            session.next_event(timeout=None)
        self.assertEqual(caught.exception.code, "RUNTIME_WORKER_FAILED")
        self.assert_failed_shutdown(runtime, engine)
