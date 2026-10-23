"""自动结束优化通过公开会话与实际门控验证，模型边界控制工作时序。"""

import unittest
from threading import Event
from time import monotonic
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from speechio import (
    LocalModelConfig,
    RealtimeConfig,
    SegmentationConfig,
    SessionConfig,
    SpeechEndConfig,
    StabilizationConfig,
)
from speechio.stt.runtime.asr import STTRuntime
from tests.session import test_automatic


class EndpointTests(unittest.TestCase):
    def setUp(self):
        test_automatic.AutomaticTests.setUp(self)

    def wait_completed(self, session):
        deadline = monotonic() + 5
        while session.optimization_status.get("early_completed", 0) == 0 and monotonic() < deadline:
            Event().wait(0.001)
        self.assertEqual(session.optimization_status.get("early_completed"), 1)

    def test_exact_early_result_is_reused_but_new_audio_is_not(self):
        for resume in (False, True):
            self.final_audio.clear()
            with STTRuntime(LocalModelConfig(adapter="whisper", model_path=self.path)) as runtime:
                session = runtime.create_auto_session(
                    vad=self.vad,
                    segmentation=SegmentationConfig(silence_confirmation=0),
                    endpoint=SpeechEndConfig(early_transcription_silence=0.1),
                )
                audio = np.concatenate((np.full(16384, 500, np.int16), np.zeros(2560, np.int16)))
                session.feed_audio(audio)
                self.wait_completed(session)
                if resume:
                    session.feed_audio(np.full(512, 700, np.int16))
                session.end_turn()
                final = session.next_event(timeout=5)
                self.assertEqual(final.kind, "final")
                self.assertEqual(final.source, "final" if resume else "early-final")
                self.assertEqual(final.audio_samples, len(audio) + (512 if resume else 0))
                self.assertEqual(len(self.final_audio), 2 if resume else 1)
                session.close()

    def test_dynamic_wait_uses_current_text_without_restarting_silence(self):
        endpoint = SpeechEndConfig(dynamic_silence=True, sentence_pause=0.2, unknown_pause=0.5)
        self.assertEqual(endpoint.pause_for("还要继续...", 0.6), 2.5)
        self.assertEqual(endpoint.pause_for("数值是3.", 0.6), 0.5)
        info = SimpleNamespace(language="zh", language_probability=1)
        with patch(
            "faster_whisper.BatchedInferencePipeline.transcribe",
            side_effect=lambda audio, **kw: (
                iter([SimpleNamespace(text="结束。", start=0, end=len(audio) / 16000)]),
                info,
            ),
        ):
            with STTRuntime(
                LocalModelConfig(adapter="whisper", model_path=self.path),
                realtime=RealtimeConfig(initial_delay_seconds=0),
            ) as runtime:
                session = runtime.create_auto_session(
                    vad=self.vad,
                    config=SessionConfig(realtime=True),
                    segmentation=SegmentationConfig(silence_confirmation=0),
                    endpoint=endpoint,
                )
                session.feed_audio(np.full(10240, 500, np.int16))
                self.assertEqual(session.next_event(timeout=5).result.text, "结束。")
                session.feed_audio(np.zeros(4096, np.int16))
                final = session.next_event(timeout=5)
                while final.kind == "realtime":
                    final = session.next_event(timeout=5)
                self.assertEqual((final.kind, final.boundary.reason), ("final", "silence"))
                self.assertEqual(session.input_state, "listening")

    def test_running_early_task_retains_capacity_until_return_and_cannot_replace_resumed_audio(
        self,
    ):
        from speechio import RuntimeLimits

        entered, release = Event(), Event()
        self.addCleanup(release.set)
        calls = []

        def model(audio, **kwargs):
            calls.append(audio.copy())
            if len(calls) == 1:
                entered.set()
                if not release.wait(5):
                    raise RuntimeError("未释放提前模型")
            return iter(
                [SimpleNamespace(text=str(len(audio)), start=0, end=len(audio) / 16000)]
            ), SimpleNamespace(language="en", language_probability=1)

        with patch("faster_whisper.BatchedInferencePipeline.transcribe", side_effect=model):
            with STTRuntime(
                LocalModelConfig(adapter="whisper", model_path=self.path),
                limits=RuntimeLimits(max_tasks=1),
            ) as runtime:
                session = runtime.create_auto_session(
                    vad=self.vad,
                    segmentation=SegmentationConfig(silence_confirmation=0),
                    endpoint=SpeechEndConfig(early_transcription_silence=0.1),
                )
                session.feed_audio(
                    np.concatenate((np.full(16384, 500, np.int16), np.zeros(2560, np.int16)))
                )
                self.assertTrue(entered.wait(5))
                session.feed_audio(np.full(515, 900, np.int16))
                session.end_input()
                self.assertEqual(runtime.capacity.tasks, 1)
                self.assertEqual(session.pending_segments, 1)
                release.set()
                final = session.next_event(timeout=5)
                self.assertEqual(
                    (final.kind, final.source, final.audio_samples), ("final", "final", 19459)
                )
                self.assertEqual(final.result.text, "19459")
                self.assertEqual(len(calls), 2)

    def test_early_failure_is_diagnosed_and_formal_final_continues(self):
        failed = Event()
        count = 0

        def model(audio, **kwargs):
            nonlocal count
            count += 1
            if count == 1:
                failed.set()
                raise RuntimeError("提前模型失败")
            return iter(
                [SimpleNamespace(text="完成", start=0, end=len(audio) / 16000)]
            ), SimpleNamespace(language="zh", language_probability=1)

        with patch("faster_whisper.BatchedInferencePipeline.transcribe", side_effect=model):
            with STTRuntime(LocalModelConfig(adapter="whisper", model_path=self.path)) as runtime:
                session = runtime.create_auto_session(
                    vad=self.vad,
                    segmentation=SegmentationConfig(silence_confirmation=0),
                    endpoint=SpeechEndConfig(early_transcription_silence=0.1),
                )
                session.feed_audio(
                    np.concatenate((np.full(16384, 500, np.int16), np.zeros(2560, np.int16)))
                )
                self.assertTrue(failed.wait(5))
                deadline = monotonic() + 5
                while (
                    not session.optimization_status.get("early_failed") and monotonic() < deadline
                ):
                    Event().wait(0.001)
                self.assertEqual(session.optimization_status.get("early_failed"), 1)
                session.end_input()
                final = session.next_event(timeout=5)
                self.assertEqual((final.kind, final.result.text), ("final", "完成"))
                self.assertEqual(final.optimization["last_error"], "INFERENCE_FAILED")

    def test_reset_discards_running_auxiliary_result_and_preserves_new_generation(self):
        entered, release = Event(), Event()
        self.addCleanup(release.set)
        count = 0

        def model(audio, **kwargs):
            nonlocal count
            count += 1
            if count == 1:
                entered.set()
                if not release.wait(5):
                    raise RuntimeError("未释放旧代次模型")
            return iter(
                [SimpleNamespace(text=str(len(audio)), start=0, end=len(audio) / 16000)]
            ), SimpleNamespace(language="en", language_probability=1)

        with patch("faster_whisper.BatchedInferencePipeline.transcribe", side_effect=model):
            with STTRuntime(LocalModelConfig(adapter="whisper", model_path=self.path)) as runtime:
                session = runtime.create_auto_session(
                    vad=self.vad,
                    segmentation=SegmentationConfig(silence_confirmation=0),
                    endpoint=SpeechEndConfig(early_transcription_silence=0.1),
                )
                session.feed_audio(
                    np.concatenate((np.full(16384, 500, np.int16), np.zeros(2560, np.int16)))
                )
                self.assertTrue(entered.wait(5))
                change = session.reset()
                self.assertEqual((change.generation, change.pending_tasks), (1, 1))
                self.assertEqual(session.optimization_status, {})
                session.feed_audio(np.full(515, 900, np.int16))
                session.end_input()
                release.set()
                final = session.next_event(timeout=5)
                self.assertEqual(
                    (final.kind, final.turn.generation, final.result.text), ("final", 1, "515")
                )
                self.assertIsNone(final.optimization)

    def test_sentence_timestamp_split_preserves_audio_arriving_during_model_call(self):
        self.assert_sentence_cut(chinese=False)

    def test_chinese_cut_uses_confirmed_raw_mark_when_stable_display_omitted_it(self):
        self.assert_sentence_cut(chinese=True)

    def assert_sentence_cut(self, *, chinese):
        """复用真实会话流程，分别验证空格分词与丢失标点的中文字符分支。"""
        entered, release = Event(), Event()
        self.addCleanup(release.set)
        text = "今天天气好。我们继续说话" if chinese else "one two three. four five six"
        tokens = (
            ("今天", "天气", "好。", "我们", "继续", "说话")
            if chinese
            else (
                "one",
                " two",
                " three.",
                " four",
                " five",
                " six",
            )
        )
        words = [
            SimpleNamespace(word=token, start=start, end=end, probability=1)
            for token, (start, end) in zip(
                tokens,
                ((0, 0.2), (0.2, 0.4), (0.4, 0.777), (0.8, 1), (1, 1.2), (1.2, 1.5)),
                strict=True,
            )
        ]
        finals = []

        def model(audio, **kwargs):
            if kwargs["word_timestamps"]:
                entered.set()
                if not release.wait(5):
                    raise RuntimeError("没有释放词时间模型")
            elif kwargs["beam_size"] == 5:
                finals.append(audio.copy())
            return iter(
                [
                    SimpleNamespace(
                        text=text,
                        start=0,
                        end=len(audio) / 16000,
                        words=words,
                    )
                ]
            ), SimpleNamespace(language="zh" if chinese else "en", language_probability=1)

        stable = StabilizationConfig(
            initial_prefix_span=0,
            min_char_evidence_span=0,
            space_span=0,
            space_confirmations=2,
            punctuation_confirmations=4 if chinese else 2,
        )
        with patch("faster_whisper.BatchedInferencePipeline.transcribe", side_effect=model):
            with STTRuntime(
                LocalModelConfig(adapter="whisper", model_path=self.path),
                realtime=RealtimeConfig(initial_delay_seconds=0, interval_seconds=0.001),
            ) as runtime:
                session = runtime.create_auto_session(
                    vad=self.vad,
                    config=SessionConfig(realtime=True, stabilization=stable),
                    endpoint=SpeechEndConfig(punctuation_split=True),
                )
                session.feed_audio(np.full(1024, 500, np.int16))
                session.next_event(timeout=5)
                session.feed_audio(np.full(32768, 500, np.int16))
                second = session.next_event(timeout=5)
                if chinese:
                    self.assertNotIn("。", second.stabilization.stable_text)
                session.feed_audio(np.full(512, 500, np.int16))
                self.assertTrue(entered.wait(5), "稳定句末未请求真实词时间")
                session.feed_audio(np.full(515, 900, np.int16))
                release.set()
                first = session.next_event(timeout=5)
                while first.kind != "final":
                    self.assertIsNotNone(first)
                    first = session.next_event(timeout=5)
                self.assertEqual(
                    (first.boundary.reason, first.boundary.end_sample), ("punctuation", 12432)
                )
                session.end_input()
                events = [first]
                while session.pending_segments:
                    events.append(session.next_event(timeout=5))
                final_events = [event for event in events if event.kind == "final"]
                self.assertEqual(len(final_events), 2)
                self.assertEqual(final_events[1].boundary.start_sample, 12432)
                expected = np.concatenate((np.full(34304, 500), np.full(515, 900))) / 32768
                np.testing.assert_array_equal(np.concatenate(finals), expected)
