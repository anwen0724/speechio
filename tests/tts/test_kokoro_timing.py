"""通过公开合成接口验证 Kokoro 时间信息；只替换外部模型与文本处理库。"""

import asyncio
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from speechio import (
    AudioChunk,
    LocalModelConfig,
    SynthesisOptions,
    Synthesizer,
    TextTiming,
    TTSError,
    WordTiming,
)

from .test_local import ArrayTensor


@contextmanager
def kokoro_model(*, context_length=512, phonemizer=None, prediction=None):
    """每个音素固定生成 600 个采样，让段时间有独立的可计算预期。"""

    class Model:
        def __init__(self, **kwargs):
            self.context_length = context_length

        def to(self, device):
            return self

        def eval(self):
            return self

        def __call__(self, phonemes, voice, speed, return_output=True):
            if prediction is not None:
                return prediction(phonemes, speed)
            return SimpleNamespace(audio=ArrayTensor(np.ones(len(phonemes) * 600) * 0.25))

    class G2P:
        def __init__(self, **kwargs):
            pass

        def __call__(self, text):
            if phonemizer is not None:
                return phonemizer(text)
            return "x" * len(text), None

    modules = {
        "torch": SimpleNamespace(load=lambda *a, **kw: [0] * 510),
        "kokoro": SimpleNamespace(KModel=Model),
        "spacy": SimpleNamespace(load=lambda *a, **kw: object()),
        "misaki": SimpleNamespace(
            en=SimpleNamespace(G2P=G2P, Lexicon=lambda british: None),
            espeak=SimpleNamespace(EspeakFallback=lambda **kw: None),
        ),
        "misaki.zh": SimpleNamespace(ZHG2P=G2P),
    }
    with tempfile.TemporaryDirectory() as temporary, patch.dict("sys.modules", modules):
        folder = Path(temporary)
        (folder / "voices").mkdir()
        (folder / "english").mkdir()
        for name in (
            "config.json",
            "kokoro-v1_1-zh.pth",
            "voices/zf_test.pt",
            "voices/af_test.pt",
            "english/config.cfg",
        ):
            (folder / name).write_text("{}", encoding="utf-8")
        yield LocalModelConfig(adapter="kokoro", model_path=folder)


def english_tokens(text):
    """保留原文标点与数字，发音结果故意与原文字数不同。"""
    rows = {
        "Dr. Smith paid $3.14!": [
            ("Dr.", " ", "dr"),
            ("Smith", " ", "sm"),
            ("paid", " ", "pd"),
            ("$3.14", "", "nn"),
            ("!", "", "!"),
        ],
        "Hi!": [("Hi", "", "hi"), ("!", "", "!")],
        "Dr. Smith ": [("Dr.", " ", "dr"), ("Smith", " ", "sm")],
        "paid $3.14!": [("paid", " ", "pd"), ("$3.14", "", "nn"), ("!", "", "!")],
    }[text]
    tokens = [SimpleNamespace(text=word, whitespace=space, phonemes=ps) for word, space, ps in rows]
    return "".join(ps + space for word, space, ps in rows), tokens


def english_prediction(phonemes, speed):
    durations = np.array([3, *([int(4 / speed)] * len(phonemes)), 2])
    return SimpleNamespace(
        audio=ArrayTensor(np.ones(int(durations.sum()) * 600) * 0.25),
        pred_dur=ArrayTensor(durations),
    )


class KokoroTimingTests(unittest.TestCase):
    def test_complete_result_maps_generated_segments_to_sample_positions(self):
        with kokoro_model(context_length=6) as config, Synthesizer(config) as tts:
            result = tts.synthesize(
                "你好。世界你好啊！",
                options=SynthesisOptions(voice="zf_test", text_timestamps="sentence"),
            )
        self.assertEqual(result.data, b"\x00\x20" * 5400)
        self.assertEqual(
            result.text_timings,
            (
                TextTiming(0, "你好。", 0.0, 0.075),
                TextTiming(1, "世界你", 0.075, 0.15),
                TextTiming(2, "好啊！", 0.15, 0.225),
            ),
        )

    def test_stream_delivers_timing_before_its_audio_through_async_event_reads(self):
        async def consume(config):
            events = []
            async with Synthesizer(config, max_buffer_seconds=0.02) as tts:
                async with await tts.stream_async(
                    "你好！再见！",
                    options=SynthesisOptions(voice="zf_test", text_timestamps="sentence"),
                ) as stream:
                    while (event := await stream.read_event_async()) is not None:
                        events.append(event)
            return events

        with kokoro_model() as config:
            events = asyncio.run(consume(config))
        samples = 0
        segments = []
        for event in events:
            if isinstance(event, TextTiming):
                self.assertEqual(event.start_seconds, samples / 24000)
                segments.append(event)
            else:
                self.assertIsInstance(event, AudioChunk)
                self.assertTrue(segments, "音频前必须已交付对应段时间")
                samples += len(event.data) // 2
                self.assertLessEqual(samples / 24000, segments[-1].end_seconds)
        self.assertEqual(
            segments,
            [
                TextTiming(0, "你好！", 0.0, 0.075),
                TextTiming(1, "再见！", 0.075, 0.15),
            ],
        )
        self.assertEqual(samples, 3600)

    def test_english_words_keep_source_text_and_follow_model_durations_at_requested_speed(self):
        with (
            kokoro_model(phonemizer=english_tokens, prediction=english_prediction) as config,
            Synthesizer(config) as tts,
        ):
            result = tts.synthesize(
                "Dr. Smith paid $3.14!Hi!",
                options=SynthesisOptions(
                    voice="af_test",
                    language="en",
                    speed=2,
                    text_timestamps="word",
                ),
            )
        self.assertEqual(result.duration_seconds, 1.0)
        self.assertEqual(
            result.text_timings,
            (
                TextTiming(
                    0,
                    "Dr. Smith paid $3.14!",
                    0.0,
                    0.725,
                    (
                        WordTiming("Dr.", 0.0, 0.125),
                        WordTiming("Smith", 0.125, 0.275),
                        WordTiming("paid", 0.275, 0.425),
                        WordTiming("$3.14", 0.425, 0.55),
                        WordTiming("!", 0.55, 0.6),
                    ),
                ),
                TextTiming(
                    1,
                    "Hi!",
                    0.725,
                    1.0,
                    (
                        WordTiming("Hi", 0.725, 0.825),
                        WordTiming("!", 0.825, 0.875),
                    ),
                ),
            ),
        )

    def test_english_long_input_keeps_word_mapping_across_capacity_splits(self):
        text = "Dr. Smith paid $3.14!"
        with (
            kokoro_model(
                context_length=10, phonemizer=english_tokens, prediction=english_prediction
            ) as config,
            Synthesizer(config) as tts,
        ):
            with tts.stream(
                text,
                options=SynthesisOptions(
                    voice="af_test",
                    language="en-gb",
                    text_timestamps="word",
                ),
            ) as stream:
                events = []
                while (event := stream.read_event()) is not None:
                    events.append(event)
        segments = [event for event in events if isinstance(event, TextTiming)]
        self.assertGreater(len(segments), 1)
        self.assertEqual("".join(segment.text for segment in segments), text)
        self.assertEqual(
            [word.text for segment in segments for word in segment.words],
            ["Dr.", "Smith", "paid", "$3.14", "!"],
        )
        samples = 0
        for event in events:
            if isinstance(event, TextTiming):
                self.assertEqual(event.start_seconds, samples / 24000)
                for word in event.words:
                    self.assertGreaterEqual(word.start_seconds, event.start_seconds)
                    self.assertLessEqual(word.end_seconds, event.end_seconds)
            else:
                samples += len(event.data) // 2
        self.assertEqual(segments[-1].end_seconds, samples / 24000)

    def test_chinese_word_request_is_rejected_without_silent_sentence_fallback(self):
        with kokoro_model() as config, Synthesizer(config) as tts:
            for method in (tts.synthesize, tts.stream):
                with self.subTest(method=method.__name__), self.assertRaises(TTSError) as caught:
                    method(
                        "你好",
                        options=SynthesisOptions(
                            voice="af_test",
                            language="zh",
                            text_timestamps="word",
                        ),
                    )
                self.assertEqual(caught.exception.code, "CAPABILITY_UNSUPPORTED")

    def test_missing_phoneme_duration_is_an_error_instead_of_fabricated_word_times(self):
        def missing_duration(phonemes, speed):
            output = english_prediction(phonemes, speed)
            output.pred_dur = ArrayTensor([3, 2, 2])
            return output

        with (
            kokoro_model(phonemizer=english_tokens, prediction=missing_duration) as config,
            Synthesizer(config) as tts,
        ):
            with self.assertRaises(TTSError) as caught:
                tts.synthesize(
                    "Hi!",
                    options=SynthesisOptions(
                        voice="af_test",
                        text_timestamps="word",
                    ),
                )
            self.assertEqual(caught.exception.code, "PROTOCOL_ERROR")

    def test_words_that_cannot_be_mapped_to_source_text_are_not_silently_returned(self):
        def changed_text(text):
            phonemes, tokens = english_tokens(text)
            tokens[0].text = "Other"
            return phonemes, tokens

        with (
            kokoro_model(phonemizer=changed_text, prediction=english_prediction) as config,
            Synthesizer(config) as tts,
        ):
            with self.assertRaises(TTSError) as caught:
                tts.synthesize(
                    "Hi!", options=SynthesisOptions(voice="af_test", text_timestamps="word")
                )
            self.assertEqual(caught.exception.code, "PROTOCOL_ERROR")
