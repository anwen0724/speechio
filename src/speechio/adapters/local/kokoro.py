"""Kokoro 中文版：本地预设音色、中英文转换和逐片段合成。"""

import re
from threading import Event

from speechio.adapters.contracts import TTSAdapter, TTSCapabilities
from speechio.adapters.local.tts import torch_device, validate_local_config, waveform_result
from speechio.audio.formats import AudioFormat
from speechio.audio.results import AudioChunk, AudioResult
from speechio.errors import TTSError
from speechio.models.files import check_tts_model_files


def _english_g2p(folder, british=False):
    """使用随模型取得的语言资源，避免 Misaki 在构造时下载 spaCy 模型。"""
    import spacy
    from misaki import en, espeak

    # Misaki 0.9.x 的初始化把模型下载写在构造中。这里仅替换资源初始化，
    # 词典、分词与发音规则仍由同一个上游 G2P 实现处理，不修改进程全局函数。
    class LocalG2P(en.G2P):
        def __init__(self):
            self.version = None
            self.british = british
            self.nlp = spacy.load(str(folder), enable=["tok2vec", "tagger"])
            self.lexicon = en.Lexicon(british)
            self.fallback = espeak.EspeakFallback(british=british)
            self.unk = "❓"

    return LocalG2P()


class KokoroAdapter(TTSAdapter):
    """逐段生成音频；不接收持续文字，不支持声音克隆。"""

    capabilities = TTSCapabilities(
        default_sample_rate=24000,
        synthesis=True,
        audio_streaming=True,
        language_selection=True,
        speed_control=True,
        sample_rates=(24000,),
    )

    def __init__(self, config):
        validate_local_config(config)
        if config.compute_type not in (None, "float32"):
            raise TTSError("CAPABILITY_UNSUPPORTED", "configure", "Kokoro 首版使用 float32")
        self.config = config
        self._model = None
        self._voices = {}
        self._g2p = {}
        self._folder = None

    def prepare(self):
        if self._model is not None:
            return
        folder = check_tts_model_files("kokoro", self.config.model_path)
        try:
            import torch
            from kokoro import KModel
        except ImportError as exc:
            raise TTSError("DEPENDENCY_MISSING", "prepare", "请安装 speechio[tts-kokoro]") from exc
        self._model = (
            KModel(
                repo_id="hexgrad/Kokoro-82M-v1.1-zh",
                config=str(folder / "config.json"),
                model=str(folder / "kokoro-v1_1-zh.pth"),
            )
            .to(torch_device(self.config, torch))
            .eval()
        )
        self._folder = folder

    def _language(self, options):
        language = options.language
        if language is None:
            language = (
                "zh"
                if options.voice.startswith("z")
                else "en-gb"
                if options.voice.startswith("b")
                else "en"
            )
        language = {"zh-cn": "zh", "en-us": "en"}.get(language, language)
        if language not in ("zh", "en", "en-gb"):
            raise TTSError(
                "CAPABILITY_UNSUPPORTED", "synthesize", "Kokoro 首版支持 zh / en / en-gb"
            )
        return language

    def _phonemizer(self, language):
        if language not in self._g2p:
            if language == "zh":
                from misaki.zh import ZHG2P

                english = self._phonemizer("en")
                self._g2p[language] = ZHG2P(
                    version="1.1", en_callable=lambda text: english(text)[0]
                )
            else:
                self._g2p[language] = _english_g2p(self._folder / "english", language == "en-gb")
        return self._g2p[language]

    def _voice(self, name):
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", name):
            raise TTSError("INVALID_INPUT", "synthesize", "Kokoro 需要本地预设音色名称")
        self.prepare()
        if name not in self._voices:
            path = self._folder / "voices" / f"{name}.pt"
            if not path.is_file():
                raise TTSError("VOICE_NOT_FOUND", "synthesize", f"本地模型没有音色：{name}")
            import torch

            self._voices[name] = torch.load(str(path), map_location="cpu", weights_only=True)
        return self._voices[name]

    def _phonemes(self, text, g2p, limit):
        """按实际音素长度再分段，避免上游对长句静默截断。"""
        phonemes = g2p(text)[0]
        if not phonemes:
            if any(char.isalnum() for char in text):
                raise TTSError("INVALID_INPUT", "synthesize", "这段文字未能转换为发音")
            return
        if "❓" in phonemes:
            raise TTSError("INVALID_INPUT", "synthesize", "文字中有无法转换的发音")
        if len(phonemes) <= limit:
            yield phonemes
            return
        if len(text) < 2:
            raise TTSError("INVALID_INPUT", "synthesize", "单个字符的发音超过模型容量")
        midpoint = len(text) // 2
        boundaries = [
            m.end() for m in re.finditer(r"[\s，。！？,;.!?]", text) if 0 < m.end() < len(text)
        ]
        split = min(boundaries, key=lambda i: abs(i - midpoint)) if boundaries else midpoint
        yield from self._phonemes(text[:split], g2p, limit)
        yield from self._phonemes(text[split:], g2p, limit)

    def _generate(self, text, options):
        pack = self._voice(options.voice)
        g2p = self._phonemizer(self._language(options))
        limit = min(self._model.context_length - 2, len(pack))
        for sentence in re.findall(r"[^。！？.!?\n]+[。！？.!?]*|[。！？.!?]+", text):
            for phonemes in self._phonemes(sentence, g2p, limit):
                output = self._model(
                    phonemes, pack[len(phonemes) - 1], options.speed, return_output=True
                )
                yield waveform_result(output.audio.detach().cpu().numpy(), 24000)

    def synthesize(self, text, *, options):
        blocks = [audio.data for audio in self._generate(text, options)]
        if not blocks:
            raise TTSError("INVALID_INPUT", "synthesize", "文字没有可合成内容")
        return AudioResult(b"".join(blocks), AudioFormat(sample_rate=24000))

    def stream(self, text, *, options):
        self._voice(options.voice)
        self._language(options)
        return KokoroStream(self._generate(text, options))

    def close(self):
        self._voices.clear()
        self._g2p.clear()
        self._model = None


class KokoroStream:
    """按读取需求推进生成，取消后不再推理下一段。"""

    pull_based = True
    format = AudioFormat(sample_rate=24000)

    def __init__(self, audio):
        self._audio = iter(audio)
        self._cancelled = Event()
        self._pending = b""

    def read(self, *, timeout=None):
        if self._cancelled.is_set():
            return None
        while not self._pending:
            result = next(self._audio, None)
            if result is None or self._cancelled.is_set():
                return None
            self._pending = result.data
        # 短块只用于交付已生成的一段音频，不把整篇预生成冒充实时。
        chunk, self._pending = self._pending[:4800], self._pending[4800:]
        return AudioChunk(chunk, self.format)

    def cancel(self):
        self._cancelled.set()

    def close(self, *, timeout=5.0):
        self._cancelled.set()
        self._audio.close()
        self._pending = b""
