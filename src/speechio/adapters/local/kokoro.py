"""Kokoro 中文版：本地预设音色、中英文转换和逐片段合成。"""

import re
from pathlib import Path
from threading import Event

import numpy as np

from speechio.adapters.contracts import TTSAdapter, TTSCapabilities
from speechio.adapters.local.tts import torch_device, validate_local_config, waveform_result
from speechio.audio.formats import AudioFormat
from speechio.audio.results import AudioChunk, AudioResult
from speechio.errors import TTSError
from speechio.models.files import check_tts_model_files
from speechio.tts.timing import TextTiming, WordTiming
from speechio.tts.voices import VoiceInfo


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


def _word_timings(text, phonemes, tokens, pred_dur, sample_offset, sample_end):
    """把同次推理的音素时长对应到英文原词，不按字数均分音频。"""
    if (
        not tokens
        or pred_dur is None
        or "".join(token.text + token.whitespace for token in tokens).strip() != text.strip()
        or "".join((token.phonemes or "") + token.whitespace for token in tokens) != phonemes
        or any(not token.phonemes and any(c.isalnum() for c in token.text) for token in tokens)
    ):
        raise TTSError("PROTOCOL_ERROR", "synthesize", "Kokoro 未返回可对应原文的英文词时间数据")
    durations = pred_dur.detach().cpu().numpy()
    # 模型可能过滤不在词表中的音素；这种情况下不能沿用原音素下标。
    if (
        durations.ndim != 1
        or len(durations) != len(phonemes) + 2
        or not np.isfinite(durations).all()
        or (durations <= 0).any()
    ):
        raise TTSError("PROTOCOL_ERROR", "synthesize", "Kokoro 发音时长与输入音素不对应")

    # Kokoro 一个预测时长单位对应 600 个采样（24 kHz 下为 25 ms）。
    # 沿用上游时间定位规则：起始标记扣除 3 个单位，词间空白由相邻词各占一半。
    # 用采样位置累计，避免多个片段的浮点加法产生时间漂移。
    positions = np.concatenate(([0], np.cumsum(durations[1:-1]))) * 600
    positions += sample_offset + max(0, durations[0] - 3) * 600
    words = []
    index = 0
    previous_space = 0
    for token in tokens:
        end = index + len(token.phonemes or "")
        space_end = end + len(token.whitespace)
        space_samples = positions[space_end] - positions[end]
        if token.phonemes:
            start_sample = positions[index] - previous_space / 2
            end_sample = positions[end] + space_samples / 2
            if not sample_offset <= start_sample <= end_sample <= sample_end:
                raise TTSError("PROTOCOL_ERROR", "synthesize", "Kokoro 词时间超出生成音频范围")
            words.append(
                WordTiming(token.text, float(start_sample / 24000), float(end_sample / 24000))
            )
        index = space_end
        previous_space = space_samples
    return tuple(words)


class KokoroAdapter(TTSAdapter):
    """逐段生成音频；不接收持续文字，不支持声音克隆。"""

    capabilities = TTSCapabilities(
        default_sample_rate=24000,
        synthesis=True,
        audio_streaming=True,
        language_selection=True,
        speed_control=True,
        sample_rates=(24000,),
        languages=("zh", "en", "en-gb"),
        devices=("cpu", "gpu"),
        voice_required=True,
        text_timestamps=("sentence", "word"),
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

    def list_voices(self, *, include_account=False):
        """只读取音色文件名；音色权重在实际合成时才加载。"""
        folder = Path(self.config.model_path).expanduser() / "voices"
        try:
            paths = sorted(folder.iterdir())
        except OSError as exc:
            raise TTSError(
                "MODEL_FILES_MISSING", "list_voices", "无法读取模型的 voices 目录"
            ) from exc
        return tuple(
            VoiceInfo(
                id=path.stem,
                name=path.stem,
                languages={"a": ("en",), "b": ("en-gb",), "z": ("zh",)}.get(path.stem[0]),
                instruction_control=False,
                text_timestamps=("sentence",)
                if path.stem.startswith("z")
                else ("sentence", "word"),
            )
            for path in paths
            if path.is_file()
            and path.suffix == ".pt"
            and re.fullmatch(r"[A-Za-z0-9_-]+", path.stem)
        )

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
        if language not in self.capabilities.languages:
            raise TTSError(
                "CAPABILITY_UNSUPPORTED", "synthesize", "Kokoro 首版支持 zh / en / en-gb"
            )
        if options.text_timestamps == "word" and language == "zh":
            raise TTSError(
                "CAPABILITY_UNSUPPORTED", "synthesize", "Kokoro 中文只支持 sentence 时间信息"
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

    def _segments(self, text, g2p, limit):
        """按音素容量分段，保留每段原文和分词，避免上游静默截断。"""
        phonemes, tokens = g2p(text)
        if not phonemes:
            if any(char.isalnum() for char in text):
                raise TTSError("INVALID_INPUT", "synthesize", "这段文字未能转换为发音")
            return
        if "❓" in phonemes:
            raise TTSError("INVALID_INPUT", "synthesize", "文字中有无法转换的发音")
        if len(phonemes) <= limit:
            yield text, phonemes, tokens
            return
        if len(text) < 2:
            raise TTSError("INVALID_INPUT", "synthesize", "单个字符的发音超过模型容量")
        midpoint = len(text) // 2
        boundaries = [
            m.end() for m in re.finditer(r"[\s，。！？,;!?]", text) if 0 < m.end() < len(text)
        ]
        split = min(boundaries, key=lambda i: abs(i - midpoint)) if boundaries else midpoint
        yield from self._segments(text[:split], g2p, limit)
        yield from self._segments(text[split:], g2p, limit)

    def _generate(self, text, options):
        from speechio._async import check_cancelled

        pack = self._voice(options.voice)
        language = self._language(options)
        g2p = self._phonemizer(language)
        limit = min(self._model.context_length - 2, len(pack))
        sample_offset = 0
        segment_index = 0
        # 英文句点保留给 G2P 解释小数、缩写；超长输入再按词边界拆分。
        for sentence in re.findall(r"[^。！？!?\n]+[。！？!?]*|[。！？!?]+", text):
            for segment_text, phonemes, tokens in self._segments(sentence, g2p, limit):
                check_cancelled()
                output = self._model(
                    phonemes, pack[len(phonemes) - 1], options.speed, return_output=True
                )
                check_cancelled()
                audio = waveform_result(output.audio.detach().cpu().numpy(), 24000)
                sample_end = sample_offset + len(audio.data) // 2
                if options.text_timestamps is not None:
                    words = None
                    if options.text_timestamps == "word":
                        words = _word_timings(
                            segment_text,
                            phonemes,
                            tokens,
                            getattr(output, "pred_dur", None),
                            sample_offset,
                            sample_end,
                        )
                    timing = TextTiming(
                        segment_index=segment_index,
                        text=segment_text,
                        start_seconds=sample_offset / 24000,
                        end_seconds=sample_end / 24000,
                        words=words,
                    )
                    audio = AudioResult(audio.data, audio.format, (timing,))
                yield audio
                sample_offset = sample_end
                segment_index += 1

    def synthesize(self, text, *, options):
        blocks = []
        timings = []
        for audio in self._generate(text, options):
            blocks.append(audio.data)
            timings.extend(audio.text_timings)
        if not blocks:
            raise TTSError("INVALID_INPUT", "synthesize", "文字没有可合成内容")
        return AudioResult(b"".join(blocks), AudioFormat(sample_rate=24000), tuple(timings))

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
        """原有音频读取跳过时间信息，保持音频块顺序不变。"""
        while isinstance(event := self.read_event(timeout=timeout), TextTiming):
            pass
        return event

    def read_event(self, *, timeout=None):
        """每段推理完成后先交付时间信息，再按需交付该段音频。"""
        if self._cancelled.is_set():
            return None
        while not self._pending:
            result = next(self._audio, None)
            if result is None or self._cancelled.is_set():
                return None
            self._pending = result.data
            if result.text_timings:
                return result.text_timings[0]
        # 短块只用于交付已生成的一段音频，不把整篇预生成冒充实时。
        chunk, self._pending = self._pending[:4800], self._pending[4800:]
        return AudioChunk(chunk, self.format)

    def cancel(self):
        self._cancelled.set()

    def close(self, *, timeout=5.0):
        self._cancelled.set()
        self._audio.close()
        self._pending = b""
