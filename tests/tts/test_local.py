"""从公开 SDK 入口验证本地适配器，替身仅替换第三方模型推理库。"""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from speechio import AudioFormat, AudioResult, LocalModelConfig, SynthesisOptions, Synthesizer


class ArrayTensor:
    def __init__(self, value):
        self.value = np.asarray(value)

    def detach(self):
        return self

    def cpu(self):
        return self

    def float(self):
        return self

    def numpy(self):
        return self.value


def qwen_folder(folder):
    folder = Path(folder)
    (folder / "speech_tokenizer").mkdir()
    config = {
        "tts_model_type": "base",
        "tts_model_size": "0b6",
        "tokenizer_type": "qwen3_tts_tokenizer_12hz",
    }
    (folder / "config.json").write_text(json.dumps(config), encoding="utf-8")
    for name in (
        "model.safetensors",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "preprocessor_config.json",
        "generation_config.json",
        "speech_tokenizer/config.json",
        "speech_tokenizer/model.safetensors",
        "speech_tokenizer/preprocessor_config.json",
    ):
        (folder / name).write_text("{}", encoding="utf-8")
    return folder


class LocalTTSTests(unittest.TestCase):
    def test_qwen_clone_is_reused_without_reference_or_reextracting_features(self):
        class Backend:
            def create_voice_clone_prompt(self, **kwargs):
                self.assert_reference = kwargs["ref_audio"][1]
                return [
                    SimpleNamespace(
                        ref_code=ArrayTensor([[1, 2]]),
                        ref_spk_embedding=ArrayTensor([0.25, 0.5]),
                        x_vector_only_mode=False,
                        icl_mode=True,
                        ref_text=kwargs["ref_text"],
                    )
                ]

            def generate_voice_clone(self, *, text, language, voice_clone_prompt):
                assert voice_clone_prompt[0].ref_text == "你好"
                return [np.ones(240, dtype=np.float32) * 0.25], 24000

        backend = Backend()
        qwen = SimpleNamespace(
            Qwen3TTSModel=SimpleNamespace(from_pretrained=lambda *a, **kw: backend)
        )
        torch = SimpleNamespace(
            float32="float32",
            float16="float16",
            bfloat16="bfloat16",
            long="int64",
            tensor=lambda a, **kw: a,
            cuda=SimpleNamespace(is_available=lambda: False),
        )

        def prompt_type(**kw):
            return SimpleNamespace(**kw)

        with (
            tempfile.TemporaryDirectory() as folder,
            patch.dict(
                "sys.modules",
                {
                    "torch": torch,
                    "qwen_tts": qwen,
                    "qwen_tts.inference.qwen3_tts_model": SimpleNamespace(
                        VoiceClonePromptItem=prompt_type
                    ),
                },
            ),
        ):
            config = LocalModelConfig(adapter="qwen_tts", model_path=qwen_folder(folder))
            with Synthesizer(config) as tts:
                voice = tts.clone_voice(
                    AudioResult(b"\x01\x00" * 240, AudioFormat(sample_rate=24000)),
                    reference_text="你好",
                )
            # 另一实例只接收克隆结果；不再执行提取参考音频的操作。
            backend.create_voice_clone_prompt = lambda **kw: self.fail("合成不应重新提取声音")
            with Synthesizer(config) as tts:
                audio = tts.synthesize("再见", options=SynthesisOptions(voice=voice, language="zh"))
            self.assertEqual(audio.data, b"\x00\x20" * 240)
            self.assertEqual(backend.assert_reference, 24000)

    def test_kokoro_stream_preserves_long_text_with_small_audio_buffer(self):
        class Model:
            context_length = 512

            def __init__(self, **kwargs):
                pass

            def to(self, device):
                return self

            def eval(self):
                return self

            def __call__(self, phonemes, voice, speed, return_output=True):
                return SimpleNamespace(audio=ArrayTensor(np.ones(len(phonemes), np.float32) * 0.25))

        class G2P:
            def __init__(self, **kwargs):
                pass

            def __call__(self, text):
                return "x" * (3 * len(text)), []

        torch = SimpleNamespace(load=lambda *a, **kw: [0] * 510)
        modules = {
            "torch": torch,
            "kokoro": SimpleNamespace(KModel=Model),
            "spacy": SimpleNamespace(load=lambda *a, **kw: object()),
            "misaki": SimpleNamespace(
                en=SimpleNamespace(G2P=G2P, Lexicon=lambda british: None),
                espeak=SimpleNamespace(EspeakFallback=lambda **kw: None),
            ),
            "misaki.zh": SimpleNamespace(ZHG2P=G2P),
        }
        with tempfile.TemporaryDirectory() as folder, patch.dict("sys.modules", modules):
            path = Path(folder)
            (path / "voices").mkdir()
            (path / "english").mkdir()
            for file in (
                "config.json",
                "kokoro-v1_1-zh.pth",
                "voices/zf_test.pt",
                "english/config.cfg",
            ):
                (path / file).write_text("{}", encoding="utf-8")
            config = LocalModelConfig(adapter="kokoro", model_path=path)
            with Synthesizer(config, max_buffer_seconds=0.001) as tts:
                with tts.stream(
                    "中" * 401, options=SynthesisOptions(voice="zf_test", language="zh")
                ) as stream:
                    audio = b"".join(chunk.data for chunk in stream)
            self.assertEqual(audio, b"\x00\x20" * (401 * 3))
