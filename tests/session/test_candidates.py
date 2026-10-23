"""候选通过正式 SDK 执行，替身仅位于外部识别引擎边界。"""

import unittest
from threading import Event

import numpy as np

import speechio as sdk
from speechio.stt.runtime.asr import STTRuntime
from tests.runtime.test_engine_injection import RecordingEngine


class BlockingEngine(RecordingEngine):
    """外部模型的第一调用可控阻塞，确定容量和迟到结果的测试时序。"""

    def __init__(self):
        super().__init__()
        self.entered, self.release = Event(), Event()

    def transcribe(self, audio, language=None, *, role="final", word_timestamps=False):
        result = super().transcribe(audio, language, role=role, word_timestamps=word_timestamps)
        if len(self.inputs) == 1:
            self.entered.set()
            if not self.release.wait(5):
                raise TimeoutError("外部模型未释放")
        return result


class CandidateTests(unittest.TestCase):
    def test_public_audio_snapshot_includes_short_tail_and_is_independent(self):
        with STTRuntime.from_adapter(adapter=RecordingEngine()) as runtime:
            session = runtime.create_session()
            unit = session.start()
            source = np.arange(525, dtype=np.int16)
            session.feed_audio(source)
            snapshot = session.current_audio()
            self.assertEqual(
                (snapshot.turn, snapshot.start_sample, snapshot.end_sample), (unit, 0, 525)
            )
            self.assertEqual(snapshot.sample_rate, 16000)
            self.assertEqual(snapshot.pcm, source.astype("<i2").tobytes())
            source.fill(0)
            session.feed_audio(np.ones(13, dtype=np.int16))
            self.assertEqual(session.current_audio().end_sample, 538)
            self.assertEqual(np.frombuffer(snapshot.pcm, dtype="<i2")[-1], 524)

    def test_empty_correction_preserves_original_final(self):
        class EmptyCorrection(RecordingEngine):
            def transcribe(self, audio, language=None, *, role="final", word_timestamps=False):
                super().transcribe(audio, language, role=role, word_timestamps=word_timestamps)
                return sdk.Transcript(text="原正文" if role == "final" else "")

        engine = EmptyCorrection()
        with STTRuntime.from_adapter(adapter=engine, on_demand=sdk.OnDemandConfig()) as runtime:
            session = runtime.create_session(sdk.SessionConfig(retain_for_correction=True))
            unit = session.start()
            session.feed_audio(np.arange(525, dtype=np.int16))
            session.end_turn()
            final = session.next_event(timeout=5)
            self.assertEqual(final.result.text, "原正文")
            self.assertEqual(session.next_event(timeout=5).kind, "completed")
            session.request_correction(unit)
            event = session.next_event(timeout=5)
            self.assertEqual(event.kind, "correction_failed")
            self.assertEqual(event.error.code, "CANDIDATE_EMPTY")
            self.assertEqual(event.candidate.replaces_task_id, final.task_id)
            self.assertEqual(event.candidate.attempts, 2)
            self.assertIsNone(session.next_event())

    def test_rejected_new_request_does_not_invalidate_running_candidate(self):
        engine = BlockingEngine()
        with STTRuntime.from_adapter(
            adapter=engine, on_demand=sdk.OnDemandConfig(), limits=sdk.RuntimeLimits(max_tasks=1)
        ) as runtime:
            session = runtime.create_session()
            session.start()
            session.feed_audio(np.arange(525, dtype=np.int16))
            original = session.request_transcription()
            try:
                self.assertTrue(engine.entered.wait(5))
                session.feed_audio(np.arange(13, dtype=np.int16))
                with self.assertRaises(sdk.STTError) as caught:
                    session.request_transcription()
                self.assertEqual(caught.exception.code, "CANDIDATE_CAPACITY_EXCEEDED")
            finally:
                engine.release.set()
            event = session.next_event(timeout=5)
            self.assertEqual(event.candidate.request, original)
            self.assertFalse(event.candidate.current)

    def test_reset_and_close_keep_running_candidate_capacity_until_return(self):
        engine = BlockingEngine()
        runtime = STTRuntime.from_adapter(adapter=engine, on_demand=sdk.OnDemandConfig())
        self.addCleanup(runtime.close)
        runtime.prepare()
        session = runtime.create_session()
        session.start()
        session.feed_audio(np.arange(525, dtype=np.int16))
        session.request_transcription()
        try:
            self.assertTrue(engine.entered.wait(5))
            self.assertEqual(session.reset().pending_tasks, 1)
            self.assertEqual(runtime.capacity.tasks, 1)
            pending = runtime.close(timeout=0)
            self.assertEqual((pending.status, pending.pending_tasks), ("pending", 1))
        finally:
            engine.release.set()
        self.assertEqual(runtime.close(timeout=5).status, "ok")
        self.assertIsNone(session.next_event())
        self.assertEqual(engine.closes, 1)

    def test_resume_fences_snapshot_claimed_before_ack(self):
        class SnapshotEngine(BlockingEngine):
            def transcribe(self, audio, language=None, *, role="final", word_timestamps=False):
                super().transcribe(audio, language, role=role, word_timestamps=word_timestamps)
                return sdk.Transcript(text="旧观察" if len(self.inputs) == 1 else "新观察")

        engine = SnapshotEngine()
        with STTRuntime.from_adapter(
            adapter=engine,
            on_demand=sdk.OnDemandConfig(),
            realtime=sdk.RealtimeConfig(
                initial_delay_seconds=0, interval_seconds=0.01, min_snapshot_samples=512
            ),
        ) as runtime:
            session = runtime.create_session(sdk.SessionConfig(realtime=True))
            session.start()
            session.feed_audio(np.arange(525, dtype=np.int16))
            try:
                self.assertTrue(engine.entered.wait(5))
                ack = session.resume(expected_audio_samples=525)
                session.feed_audio(np.arange(512, dtype=np.int16))
            finally:
                engine.release.set()
            event = session.next_event(timeout=5)
            self.assertEqual((event.kind, event.result.text), ("realtime", "新观察"))
            self.assertEqual(
                (event.resume_epoch, event.resume_id), (ack.resume_epoch, ack.resume_id)
            )
            self.assertGreater(event.audio_samples, ack.candidate_start_sample)

    def test_late_correction_keeps_complete_audio_and_links_original_final(self):
        class CorrectingEngine(RecordingEngine):
            def transcribe(self, audio, language=None, *, role="final", word_timestamps=False):
                result = super().transcribe(
                    audio, language, role=role, word_timestamps=word_timestamps
                )
                return sdk.Transcript(text="修正完成") if role == "candidate" else result

        engine = CorrectingEngine()
        with STTRuntime.from_adapter(adapter=engine, on_demand=sdk.OnDemandConfig()) as runtime:
            session = runtime.create_session(sdk.SessionConfig(retain_for_correction=True))
            unit = session.start()
            source = np.arange(525, dtype=np.int16)
            session.feed_audio(source)
            original = session.end_turn()
            self.assertEqual(session.next_event(timeout=5).kind, "final")
            self.assertEqual(session.next_event(timeout=5).kind, "completed")
            correction = session.request_correction(unit)
            event = session.next_event(timeout=5)
            self.assertEqual((event.kind, event.candidate.text), ("correction", "修正完成"))
            self.assertEqual(event.candidate.request, correction)
            self.assertEqual(event.candidate.replaces_task_id, original.task_id)
            np.testing.assert_array_equal(engine.inputs[-1][0], source.astype(np.float32) / 32768)
            self.assertEqual(session.retained_audio_samples, 525)
            session.start()
            self.assertEqual(session.retained_audio_samples, 0)
            with self.assertRaises(sdk.STTError) as caught:
                session.request_correction(unit)
            self.assertEqual(caught.exception.code, "CORRECTION_UNAVAILABLE")

    def test_candidate_only_runtime_requires_no_final_model(self):
        engine = RecordingEngine()
        with STTRuntime.from_on_demand(sdk.OnDemandConfig.from_adapter(adapter=engine)) as runtime:
            self.assertFalse(runtime.final_enabled)
            session = runtime.create_session(sdk.SessionConfig(completion_mode="candidate"))
            session.start()
            session.feed_audio(np.arange(513, dtype=np.int16))
            session.end_turn()
            final, done = session.next_event(timeout=5), session.next_event(timeout=5)
            self.assertEqual(
                (final.kind, final.source, done.kind), ("final", "candidate-final", "completed")
            )
            self.assertEqual([entry[2] for entry in engine.inputs], ["candidate"])
        self.assertEqual((engine.preparations, engine.closes), (1, 1))

    def test_candidate_completion_reuses_only_unchanged_full_audio(self):
        for append in (False, True):
            with self.subTest(append=append):
                engine = RecordingEngine()
                with STTRuntime.from_adapter(
                    adapter=engine, on_demand=sdk.OnDemandConfig()
                ) as runtime:
                    session = runtime.create_session(sdk.SessionConfig(completion_mode="candidate"))
                    session.start()
                    session.feed_audio(np.arange(525, dtype=np.int16))
                    session.request_transcription()
                    self.assertEqual(session.next_event(timeout=5).kind, "candidate")
                    if append:
                        session.feed_audio(np.arange(13, dtype=np.int16))
                    session.end_turn()
                    final, completed = session.next_event(timeout=5), session.next_event(timeout=5)
                    self.assertEqual((final.kind, completed.kind), ("final", "completed"))
                    self.assertEqual(final.source, "candidate-final")
                    self.assertEqual(final.optimization["candidate_reused"], not append)
                    self.assertEqual(
                        [entry[2] for entry in engine.inputs], ["candidate"] * (2 if append else 1)
                    )
                    self.assertIsNone(session.next_event())

    def test_normal_empty_candidate_has_one_flush_retry_but_hint_has_none(self):
        class EmptyFirst(RecordingEngine):
            def transcribe(self, audio, language=None, *, role="final", word_timestamps=False):
                super().transcribe(audio, language, role=role, word_timestamps=word_timestamps)
                return sdk.Transcript(text="" if len(self.inputs) == 1 else "complete snapshot")

        source = np.arange(525, dtype=np.int16)
        for hint in (False, True):
            with self.subTest(hint=hint):
                engine = EmptyFirst()
                with STTRuntime.from_adapter(
                    adapter=engine, on_demand=sdk.OnDemandConfig()
                ) as runtime:
                    session = runtime.create_session()
                    session.start()
                    session.feed_audio(source)
                    session.request_transcription(hint=hint)
                    event = session.next_event(timeout=5)
                    self.assertEqual(event.kind, "candidate")
                    self.assertEqual(event.candidate.attempts, 1 if hint else 2)
                    self.assertEqual(event.candidate.status, "empty" if hint else "whole")
                    if not hint:
                        np.testing.assert_array_equal(
                            engine.inputs[1][0][:525], source.astype(np.float32) / 32768
                        )
                        self.assertEqual(len(engine.inputs[1][0]), 8525)
                        self.assertTrue(np.all(engine.inputs[1][0][525:] == 0))
                        self.assertEqual(event.candidate.added_silence_samples, 8000)
                    self.assertEqual(event.candidate.request.end_sample, 525)

    def test_resume_keeps_history_and_suppresses_old_candidate(self):
        class BlockingFirst(RecordingEngine):
            def __init__(self):
                super().__init__()
                self.entered, self.release = Event(), Event()

            def transcribe(self, audio, language=None, *, role="final", word_timestamps=False):
                result = super().transcribe(
                    audio, language, role=role, word_timestamps=word_timestamps
                )
                if len(self.inputs) == 1:
                    self.entered.set()
                    if not self.release.wait(5):
                        raise TimeoutError("旧候选未释放")
                return result

        engine = BlockingFirst()
        runtime = STTRuntime.from_adapter(adapter=engine, on_demand=sdk.OnDemandConfig())
        self.addCleanup(runtime.close)
        runtime.prepare()
        session = runtime.create_session()
        unit = session.start()
        first, second = np.arange(525, dtype=np.int16), -np.arange(513, dtype=np.int16)
        session.feed_audio(first)
        old = session.request_transcription()
        try:
            self.assertTrue(engine.entered.wait(5))
            with self.assertRaises(sdk.STTError) as caught:
                session.resume(expected_audio_samples=524)
            self.assertEqual(caught.exception.code, "RESUME_BOUNDARY_MISMATCH")
            ack = session.resume(expected_audio_samples=525)
            self.assertEqual(
                (ack.turn, ack.resume_epoch, ack.candidate_start_sample), (unit, 1, 525)
            )
            session.feed_audio(second)
            current = session.request_transcription()
            self.assertEqual((current.start_sample, current.end_sample), (0, 1038))
            self.assertEqual((current.resume_epoch, current.resume_id), (1, ack.resume_id))
            self.assertEqual(runtime.capacity.tasks, 2)
        finally:
            engine.release.set()
        event = session.next_event(timeout=5)
        self.assertEqual(event.candidate.request, current)
        self.assertNotEqual(event.task_id, old.request_id)
        np.testing.assert_array_equal(
            engine.inputs[1][0], np.concatenate((first, second)).astype(np.float32) / 32768
        )
        session.end_turn()
        while session.next_event(timeout=5).kind != "completed":
            pass

    def test_tail_snapshot_includes_short_samples_and_does_not_block_input(self):
        class BlockingCandidate(RecordingEngine):
            def __init__(self):
                super().__init__()
                self.entered, self.release = Event(), Event()

            def transcribe(self, audio, language=None, *, role="final", word_timestamps=False):
                result = super().transcribe(
                    audio, language, role=role, word_timestamps=word_timestamps
                )
                if role == "candidate":
                    self.entered.set()
                    if not self.release.wait(5):
                        raise TimeoutError("候选模型未释放")
                return result

        engine = BlockingCandidate()
        runtime = STTRuntime.from_adapter(
            adapter=engine, on_demand=sdk.OnDemandConfig(tail_seconds=0.02)
        )
        self.addCleanup(runtime.close)
        runtime.prepare()
        session = runtime.create_session()
        session.start()
        source = np.arange(525, dtype=np.int16)
        session.feed_audio(source)
        request = session.request_transcription(scope="tail")
        try:
            self.assertTrue(engine.entered.wait(5))
            self.assertEqual((request.start_sample, request.end_sample), (205, 525))
            session.feed_audio(-np.arange(513, dtype=np.int16))
            self.assertEqual(runtime.capacity.tasks, 1)
        finally:
            engine.release.set()
        candidate = session.next_event(timeout=5)
        self.assertEqual(candidate.kind, "candidate")
        self.assertEqual(candidate.candidate.request, request)
        self.assertEqual(
            (candidate.candidate.status, candidate.candidate.current), ("tail_only", False)
        )
        np.testing.assert_array_equal(engine.inputs[0][0], source[205:].astype(np.float32) / 32768)
        self.assertEqual(engine.inputs[0][2], "candidate")
        session.end_turn()
        while session.next_event(timeout=5).kind != "completed":
            pass
        self.assertEqual(runtime.close().status, "ok")
        self.assertEqual((engine.preparations, engine.closes), (1, 1))
