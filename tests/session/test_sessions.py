"""通过连续会话公开接口验证归属，只在外部模型边界控制时序。"""

import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import soxr

from realtime_asr import (
    ASRError,
    ASRRuntime,
    AudioFormat,
    RecognizerConfig,
    RuntimeLimits,
    SessionConfig,
)
from tests.runtime import test_recognizer


class SessionTests(unittest.TestCase):
    """复用真实模型清单与外部加载替身，不替换会话或调度逻辑。"""

    def setUp(self):
        test_recognizer.RecognizerTests.setUp(self)

    def test_repeated_finish_has_one_terminal_pair_even_for_empty_text(self):
        with patch(
            "faster_whisper.BatchedInferencePipeline.transcribe",
            return_value=(iter([]), SimpleNamespace(language=None, language_probability=None)),
        ):
            with ASRRuntime(RecognizerConfig(model=self.path)) as runtime:
                session = runtime.create_session()
                unit = session.start()
                session.feed_audio(b"\x00\x00")
                task = session.finish()
                self.assertIs(session.finish(), task)
                final = session.get_event(timeout=5)
                self.assertEqual((final.kind, final.unit, final.result.text), ("final", unit, ""))
                self.assertEqual(session.get_event(timeout=5).kind, "completed")
                self.assertIsNone(session.get_event())

    def test_next_unit_is_accepted_while_previous_model_call_is_blocked(self):
        entered, release = threading.Event(), threading.Event()
        received = []

        def native(audio, **kwargs):
            received.append(audio.copy())
            if len(received) == 1:
                entered.set()
                if not release.wait(5):
                    raise RuntimeError("测试未释放模型")
            text = str(len(received))
            return iter([SimpleNamespace(text=text, start=0, end=1)]), SimpleNamespace(
                language="en", language_probability=1.0
            )

        runtime = ASRRuntime(RecognizerConfig(model=self.path))
        self.addCleanup(runtime.close)
        with patch("faster_whisper.BatchedInferencePipeline.transcribe", side_effect=native):
            runtime.prepare()
            session = runtime.create_session()
            first = session.start()
            source = np.array([100, 200, 300], np.int16)
            session.feed_audio(source)
            task = session.finish()
            self.assertEqual(task.unit, first)
            self.assertTrue(entered.wait(5))
            try:
                source[:] = 0
                second = session.start()
                session.feed_audio(np.array([-100, -200], np.int16))
                session.finish()
            finally:
                release.set()
            events = [session.get_event(timeout=5) for _ in range(4)]
            self.assertEqual([event.kind for event in events], ["final", "completed"] * 2)
            self.assertEqual([event.unit for event in events], [first, first, second, second])
            self.assertEqual([event.sequence for event in events], [1, 2, 3, 4])
            np.testing.assert_array_equal(received[0], np.array([100, 200, 300]) / 32768)
            np.testing.assert_array_equal(received[1], np.array([-100, -200]) / 32768)
            self.assertIsNone(session.get_event())

    def test_invalid_finish_terminates_only_its_unit_and_accepts_next_unit(self):
        with ASRRuntime(RecognizerConfig(model=self.path)) as runtime:
            session = runtime.create_session()
            for payload, code in ((b"", "INVALID_AUDIO"), (b"\x01", "AUDIO_INPUT_TRUNCATED")):
                with self.subTest(code=code):
                    unit = session.start()
                    session.feed_audio(payload)
                    with self.assertRaises(ASRError) as caught:
                        session.finish()
                    self.assertEqual(caught.exception.code, code)
                    event = session.get_event(timeout=5)
                    self.assertEqual(
                        (event.kind, event.unit, event.error.code), ("failed", unit, code)
                    )
                    self.assertEqual(session.state, "idle")
            self.assertEqual(runtime.capacity.tasks, 0)

    def test_model_failure_does_not_prevent_next_unit_and_results_are_isolated(self):
        def native(audio, **kwargs):
            if audio[0] < 0:
                raise RuntimeError("外部推理失败")
            return iter([SimpleNamespace(text="ok", start=0, end=1)]), SimpleNamespace(
                language="en", language_probability=1
            )

        with patch("faster_whisper.BatchedInferencePipeline.transcribe", side_effect=native):
            with ASRRuntime(RecognizerConfig(model=self.path)) as runtime:
                session = runtime.create_session()
                failed = session.start()
                session.feed_audio(np.array([-1], np.int16))
                session.finish()
                for _ in range(2):
                    session.start()
                    session.feed_audio(np.array([1], np.int16))
                    session.finish()
                failure = session.get_event(timeout=5)
                self.assertEqual((failure.kind, failure.unit), ("failed", failed))
                self.assertEqual(failure.error.code, "INFERENCE_FAILED")
                finals = []
                for _ in range(2):
                    finals.append(session.get_event(timeout=5))
                    self.assertEqual(session.get_event(timeout=5).kind, "completed")
                finals[0].result.segments[0]["text"] = "changed"
                self.assertEqual(finals[1].result.segments[0]["text"], "ok")

    def test_cancel_scopes_keep_running_audio_until_physical_return(self):
        entered, release = threading.Event(), threading.Event()
        received = []

        def native(audio, **kwargs):
            received.append(float(audio[0]))
            if len(received) == 1:
                entered.set()
                if not release.wait(5):
                    raise RuntimeError("测试未释放模型")
            return iter([]), SimpleNamespace(language=None, language_probability=None)

        with patch("faster_whisper.BatchedInferencePipeline.transcribe", side_effect=native):
            with ASRRuntime(RecognizerConfig(model=self.path)) as runtime:
                session = runtime.create_session()
                running = session.start()
                session.feed_audio(np.array([1], np.int16))
                session.finish()
                self.assertTrue(entered.wait(5))
                try:
                    queued = session.start()
                    session.feed_audio(np.array([2], np.int16))
                    session.finish()
                    active = session.start()
                    session.feed_audio(np.array([3], np.int16))
                    self.assertTrue(session.cancel())
                    self.assertFalse(session.cancel())
                    self.assertEqual(runtime.capacity.tasks, 2)
                    self.assertTrue(session.cancel(queued))
                    self.assertEqual(runtime.capacity.tasks, 1)
                    self.assertTrue(session.cancel(running))
                    self.assertFalse(session.cancel(running))
                    self.assertEqual(runtime.capacity.tasks, 1)
                    cancelled = [session.get_event(timeout=5) for _ in range(3)]
                    self.assertEqual([e.unit for e in cancelled], [running, queued, active])
                    self.assertEqual([e.kind for e in cancelled], ["cancelled"] * 3)
                    next_unit = session.start()
                    session.feed_audio(np.array([4], np.int16))
                    session.finish()
                finally:
                    release.set()
                self.assertEqual(session.get_event(timeout=5).unit, next_unit)
                self.assertEqual(session.get_event(timeout=5).kind, "completed")
                self.assertIsNone(session.get_event())
                self.assertEqual(received, [1 / 32768, 4 / 32768])

    def test_reset_rejects_late_failure_and_preserves_other_session(self):
        entered, release = threading.Event(), threading.Event()

        def native(audio, **kwargs):
            if audio[0] < 0:
                entered.set()
                if not release.wait(5):
                    raise RuntimeError("测试未释放模型")
                raise RuntimeError("旧代次迟到失败")
            return iter([]), SimpleNamespace(language=None, language_probability=None)

        with patch("faster_whisper.BatchedInferencePipeline.transcribe", side_effect=native):
            with ASRRuntime(RecognizerConfig(model=self.path)) as runtime:
                session, other = runtime.create_session(), runtime.create_session()
                old = session.start()
                session.feed_audio(np.array([-1], np.int16))
                session.finish()
                self.assertTrue(entered.wait(5))
                try:
                    queued = session.start()
                    session.feed_audio(np.array([9], np.int16))
                    session.finish()
                    active = session.start()
                    session.feed_audio(b"\x01")
                    untouched = other.start()
                    other.feed_audio(np.array([2], np.int16))
                    other.finish()
                    change = session.reset()
                    self.assertEqual(change.generation, 1)
                    self.assertEqual([e.unit for e in change.cancelled], [old, queued, active])
                    self.assertEqual(change.pending_tasks, 1)
                    self.assertFalse(session.cancel(old))
                    with self.assertRaises(ASRError) as caught:
                        session.cancel(untouched)
                    self.assertEqual(caught.exception.code, "INVALID_TARGET")
                    fresh = session.start()
                    session.feed_audio(np.array([3], np.int16))
                    session.finish()
                finally:
                    release.set()
                self.assertEqual(session.get_event(timeout=5).unit, fresh)
                self.assertEqual(session.get_event(timeout=5).kind, "completed")
                self.assertIsNone(session.get_event())
                self.assertEqual(other.get_event(timeout=5).unit, untouched)
                self.assertEqual(other.get_event(timeout=5).kind, "completed")

    def test_session_close_discards_pending_events_without_closing_shared_model(self):
        def native(audio, **kwargs):
            return iter([]), SimpleNamespace(language=None, language_probability=None)

        with patch("faster_whisper.BatchedInferencePipeline.transcribe", side_effect=native):
            with ASRRuntime(RecognizerConfig(model=self.path)) as runtime:
                session = runtime.create_session()
                session.start()
                session.feed_audio(np.array([1], np.int16))
                session.finish()
                self.assertEqual(session.get_event(timeout=5).kind, "final")
                change = session.close()
                self.assertEqual(change.discarded_events, 1)
                self.assertIs(session.close(), change)
                self.assertEqual(session.state, "closed")
                self.assertIsNone(session.get_event(timeout=None))
                with self.assertRaises(ASRError) as caught:
                    session.start()
                self.assertEqual(caught.exception.code, "SESSION_CLOSED")
                self.assertEqual(runtime.capacity.open_sessions, 0)
                other = runtime.create_session()
                other.start()
                other.feed_audio(np.array([2], np.int16))
                other.finish()
                self.assertEqual(other.get_event(timeout=5).kind, "final")
                self.assertEqual(other.get_event(timeout=5).kind, "completed")

    def test_audio_and_result_limits_reject_without_losing_accepted_prefix(self):
        received = []

        def native(audio, **kwargs):
            received.append(audio.copy())
            return iter([]), SimpleNamespace(language=None, language_probability=None)

        with patch("faster_whisper.BatchedInferencePipeline.transcribe", side_effect=native):
            with ASRRuntime(RecognizerConfig(model=self.path)) as runtime:
                session = runtime.create_session(
                    SessionConfig(max_audio_samples=3, max_pending_units=1)
                )
                session.start()
                session.feed_audio(b"\x01")
                with self.assertRaises(ASRError) as caught:
                    session.feed_audio(bytes(6))
                self.assertEqual(caught.exception.code, "AUDIO_CAPACITY_EXCEEDED")
                session.feed_audio(b"\x00\x02\x00\x03\x00")
                self.assertEqual(session.finish().audio_samples, 3)
                with self.assertRaises(ASRError) as caught:
                    session.start()
                self.assertEqual(caught.exception.code, "RESULT_CAPACITY_EXCEEDED")
                session.get_event(timeout=5)
                with self.assertRaises(ASRError):
                    session.start()
                self.assertEqual(session.get_event(timeout=5).kind, "completed")
                session.start()
                session.cancel()
                np.testing.assert_array_equal(received[0], np.array([1, 2, 3]) / 32768)

    def test_task_limits_keep_sealed_audio_for_retry_and_count_running_cancel(self):
        entered, release = threading.Event(), threading.Event()
        received = []

        def native(audio, **kwargs):
            received.append(audio.copy())
            if len(received) == 1:
                entered.set()
                if not release.wait(5):
                    raise RuntimeError("测试未释放模型")
            return iter([]), SimpleNamespace(language=None, language_probability=None)

        with patch("faster_whisper.BatchedInferencePipeline.transcribe", side_effect=native):
            with ASRRuntime(
                RecognizerConfig(model=self.path),
                limits=RuntimeLimits(max_tasks=2, max_task_audio_samples=100),
            ) as runtime:
                first, second = runtime.create_session(), runtime.create_session()
                running = first.start()
                first.feed_audio(np.array([1, 2], np.int16))
                first.finish()
                self.assertTrue(entered.wait(5))
                try:
                    queued = first.start()
                    first.feed_audio(np.array([3], np.int16))
                    first.finish()
                    unit = second.start()
                    second.feed_audio(np.array([4], np.int16))
                    with self.assertRaises(ASRError) as caught:
                        second.finish()
                    self.assertEqual(caught.exception.code, "TASK_CAPACITY_EXCEEDED")
                    self.assertEqual(second.state, "sealed")
                    with self.assertRaises(ASRError) as caught:
                        second.feed_audio(np.array([5], np.int16))
                    self.assertEqual(caught.exception.code, "UNIT_SEALED")
                    first.cancel(running)
                    with self.assertRaises(ASRError):
                        second.finish()
                    first.cancel(queued)
                    task = second.finish()
                    self.assertEqual((task.unit, task.audio_samples), (unit, 1))
                    self.assertEqual(runtime.capacity.task_audio_samples, 3)
                finally:
                    release.set()
                second.get_event(timeout=5)
                second.get_event(timeout=5)
                self.assertEqual(runtime.capacity.tasks, 0)
                np.testing.assert_array_equal(received[1], np.array([4]) / 32768)

    def test_task_audio_limit_is_independent_of_task_count(self):
        entered, release = threading.Event(), threading.Event()

        def native(audio, **kwargs):
            entered.set()
            if not release.wait(5):
                raise RuntimeError("测试未释放模型")
            return iter([]), SimpleNamespace(language=None, language_probability=None)

        with patch("faster_whisper.BatchedInferencePipeline.transcribe", side_effect=native):
            with ASRRuntime(
                RecognizerConfig(model=self.path),
                limits=RuntimeLimits(max_tasks=4, max_task_audio_samples=2),
            ) as runtime:
                session = runtime.create_session()
                session.start()
                session.feed_audio(np.array([1, 2], np.int16))
                session.finish()
                self.assertTrue(entered.wait(5))
                try:
                    session.start()
                    session.feed_audio(np.array([3], np.int16))
                    with self.assertRaises(ASRError) as caught:
                        session.finish()
                    self.assertEqual(caught.exception.code, "TASK_CAPACITY_EXCEEDED")
                    self.assertEqual(runtime.capacity.tasks, 1)
                    session.cancel()
                finally:
                    release.set()

    def test_sessions_are_admitted_with_a_limit_and_scheduled_round_robin(self):
        entered, release = threading.Event(), threading.Event()
        order = []

        def native(audio, **kwargs):
            order.append(round(float(audio[0]) * 32768))
            if len(order) == 1:
                entered.set()
                if not release.wait(5):
                    raise RuntimeError("测试未释放模型")
            return iter([]), SimpleNamespace(language=None, language_probability=None)

        with patch("faster_whisper.BatchedInferencePipeline.transcribe", side_effect=native):
            with ASRRuntime(
                RecognizerConfig(model=self.path), limits=RuntimeLimits(max_sessions=2)
            ) as runtime:
                a, b = runtime.create_session(), runtime.create_session()
                a.start()
                a.feed_audio(np.array([1], np.int16))
                a.finish()
                self.assertTrue(entered.wait(5))
                try:
                    with self.assertRaises(ASRError) as caught:
                        runtime.create_session()
                    self.assertEqual(caught.exception.code, "SESSION_CAPACITY_EXCEEDED")
                    for session, sample in ((a, 2), (a, 3), (b, 4)):
                        session.start()
                        session.feed_audio(np.array([sample], np.int16))
                        session.finish()
                finally:
                    release.set()
                for session, count in ((a, 6), (b, 2)):
                    for _ in range(count):
                        self.assertIsNotNone(session.get_event(timeout=5))
                self.assertEqual(order, [1, 2, 4, 3])
                b.close()
                self.assertEqual(runtime.create_session().state, "idle")

    def test_runtime_close_timeout_preserves_running_resources_and_then_finishes(self):
        entered, release = threading.Event(), threading.Event()

        def native(audio, **kwargs):
            entered.set()
            if not release.wait(5):
                raise RuntimeError("测试未释放模型")
            return iter([]), SimpleNamespace(language=None, language_probability=None)

        runtime = ASRRuntime(RecognizerConfig(model=self.path))
        self.addCleanup(runtime.close)
        with patch("faster_whisper.BatchedInferencePipeline.transcribe", side_effect=native):
            runtime.prepare()
            session = runtime.create_session()
            session.start()
            session.feed_audio(np.array([1], np.int16))
            session.finish()
            self.assertTrue(entered.wait(5))
            try:
                session.start()
                session.feed_audio(np.array([2], np.int16))
                session.finish()
                outcome = runtime.close(timeout=0)
                self.assertEqual((outcome.status, outcome.pending_tasks), ("pending", 1))
                self.assertEqual(runtime.state, "closing")
                self.assertEqual(runtime.capacity.tasks, 1)
                self.assertEqual(session.state, "closed")
                with self.assertRaises(ASRError) as caught:
                    runtime.create_session()
                self.assertEqual(caught.exception.code, "RUNTIME_CLOSING")
            finally:
                release.set()
            final = runtime.close(timeout=5)
            self.assertEqual((final.status, final.pending_tasks), ("ok", 0))
            self.assertEqual(runtime.state, "closed")
            self.assertIs(runtime.close(), final)
            self.assertEqual(runtime.capacity.task_audio_samples, 0)

    def test_resampling_flush_and_source_limit_are_independent_of_byte_packets(self):
        received = []

        def native(audio, **kwargs):
            received.append(audio.copy())
            return iter([]), SimpleNamespace(language=None, language_probability=None)

        with patch("faster_whisper.BatchedInferencePipeline.transcribe", side_effect=native):
            with ASRRuntime(RecognizerConfig(model=self.path)) as runtime:
                session = runtime.create_session(
                    SessionConfig(format=AudioFormat(48000, 2), max_audio_samples=16)
                )
                for sign in (1, -1):
                    session.start()
                    source = np.tile(np.array([1000, 3000], np.int16) * sign, (48, 1))
                    raw = source.tobytes()
                    for offset in range(0, len(raw), 7):
                        session.feed_audio(raw[offset : offset + 7])
                    with self.assertRaises(ASRError) as caught:
                        session.feed_audio(b"\x00")
                    self.assertEqual(caught.exception.code, "AUDIO_CAPACITY_EXCEEDED")
                    self.assertEqual(session.finish().audio_samples, 16)
                for _ in range(4):
                    self.assertIsNotNone(session.get_event(timeout=5))
                expected = soxr.resample(np.full(48, 2000, np.float32), 48000, 16000)
                expected = expected.astype(np.int16).astype(np.float32) / 32768
                np.testing.assert_allclose(received[0], expected, atol=1 / 32768)
                np.testing.assert_allclose(received[1], -expected, atol=1 / 32768)

    def test_prepare_retry_and_context_body_exception_keep_runtime_contract(self):
        runtime = ASRRuntime(RecognizerConfig(model=self.path))
        self.addCleanup(runtime.close)
        with self.assertRaises(ASRError) as caught:
            runtime.create_session()
        self.assertEqual(caught.exception.code, "RUNTIME_NOT_READY")
        with patch("faster_whisper.WhisperModel", side_effect=OSError("外部加载失败")):
            with self.assertRaises(ASRError):
                runtime.prepare()
        self.assertEqual(runtime.state, "unprepared")
        original = ValueError("宿主正文错误")
        with self.assertRaises(ValueError) as caught:
            with runtime:
                raise original
        self.assertIs(caught.exception, original)
        self.assertEqual(runtime.close_result.status, "ok")
        with self.assertRaises(ASRError) as caught:
            runtime.prepare()
        self.assertEqual(caught.exception.code, "RUNTIME_CLOSED")

    def test_rejected_input_block_does_not_consume_audio_capacity(self):
        received = []

        def native(audio, **kwargs):
            received.append(audio.copy())
            return iter([]), SimpleNamespace(language=None, language_probability=None)

        with patch("faster_whisper.BatchedInferencePipeline.transcribe", side_effect=native):
            with ASRRuntime(RecognizerConfig(model=self.path)) as runtime:
                session = runtime.create_session(
                    SessionConfig(format=AudioFormat(sample_type="float32"), max_audio_samples=2)
                )
                session.start()
                session.feed_audio(np.array([0.25], np.float32))
                with self.assertRaises(ASRError) as caught:
                    session.feed_audio(np.array([np.nan], np.float32))
                self.assertEqual(caught.exception.code, "INVALID_AUDIO")
                self.assertEqual(session.state, "receiving")
                session.feed_audio(np.array([-0.25], np.float32))
                session.finish()
                self.assertEqual(session.get_event(timeout=5).kind, "final")
                session.get_event(timeout=5)
                np.testing.assert_array_equal(received[0], np.array([8191, -8191]) / 32768)
