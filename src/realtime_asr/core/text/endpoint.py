"""文本只提出等待与句末提示；时间证据和音频切分由会话验证。"""

import re
from dataclasses import dataclass
from math import isfinite

from realtime_asr.core.errors import ASRError
from realtime_asr.core.text.stabilizer import project_text

SENTENCE_MARKS = ".?!。？！"
ABBREVIATIONS = {"mr", "dr", "mrs", "ms", "prof", "sr", "jr", "st", "vs", "etc", "e.g", "i.e"}


@dataclass(frozen=True, kw_only=True)
class EndpointConfig:
    """自动会话的可选结束策略；默认保留固定 VAD 边界。"""

    early_transcription_silence: float = 0
    dynamic_silence: bool = False
    sentence_pause: float = 0.9
    unknown_pause: float = 1.2
    continuation_pause: float = 2.5
    punctuation_split: bool = False

    def __post_init__(self):
        seconds = (
            self.early_transcription_silence,
            self.sentence_pause,
            self.unknown_pause,
            self.continuation_pause,
        )
        if (
            any(
                not isinstance(value, (float, int))
                or isinstance(value, bool)
                or not isfinite(value)
                or value < 0
                for value in seconds
            )
            or min(self.sentence_pause, self.unknown_pause, self.continuation_pause) == 0
            or type(self.dynamic_silence) is not bool
            or type(self.punctuation_split) is not bool
        ):
            raise ASRError("INVALID_CONFIGURATION", "configure", "自动结束优化配置无效")

    def pause_for(self, text, baseline):
        """文本选择静音等待，不更新本次静音的起点。"""
        if not self.dynamic_silence:
            return baseline
        text = text.rstrip()
        if text.endswith(("...", "…")):
            return self.continuation_pause
        if (
            text
            and text[-1] in SENTENCE_MARKS
            and not (text[-1] == "." and len(text) > 1 and text[-2].isdigit())
        ):
            return self.sentence_pause
        return self.unknown_pause


@dataclass(frozen=True)
class SentenceHint:
    """句末标点与左侧锚点，字符模式保留中文语义。"""

    mark: str
    anchor: tuple[str, ...]
    character_mode: bool


def _units(text, character_mode):
    normalized = project_text(text).text
    if character_mode:
        return tuple(char for char in normalized if not char.isspace())
    return tuple(normalized.split())


def _hints(text):
    """只接受句末类别，跳过数字、缩写、省略号与过短上下文。"""
    character_mode = any("\u3400" <= char <= "\u9fff" for char in text)
    for index, mark in enumerate(text):
        if mark not in SENTENCE_MARKS or not text[index + 1 :].strip():
            continue
        if mark == ".":
            if (index > 0 and text[index - 1] == ".") or text[index + 1] == ".":
                continue
            if index > 0 and text[index - 1].isdigit() and text[index + 1].isdigit():
                continue
            previous = re.search(r"([\w.]+)$", text[:index])
            if previous and (
                previous.group(1).casefold() in ABBREVIATIONS or len(previous.group(1)) == 1
            ):
                continue
        left, right = (
            _units(text[:index], character_mode),
            _units(text[index + 1 :], character_mode),
        )
        # 中文按汉字上下文确认；英文按词确认，不能机械套用空格分词。
        segment = re.split(r"[.!?。！？]", text[:index])[-1]
        left_count = (
            sum("\u3400" <= char <= "\u9fff" for char in segment)
            if character_mode
            else len(_units(segment, False))
        )
        right_count = (
            sum("\u3400" <= char <= "\u9fff" for char in text[index + 1 :])
            if character_mode
            else len(right)
        )
        if left_count < 3 or right_count < 2:
            continue
        yield index, SentenceHint(mark, left[-6:] if character_mode else left[-3:], character_mode)


class SentenceBoundaryPolicy:
    """同一锚点需要两次音频推进，每个输入秒最多尝试一次。"""

    def __init__(self):
        self._hint = None
        self._confirmations = 0
        self._last_audio = 0
        self._last_attempt = None

    @property
    def current_hint(self):
        return self._hint

    def observe(self, stable_text, audio_samples, *, observation_text=None):
        hints = list(_hints(stable_text))
        if not hints and observation_text:
            # 调用方只传接受观察；备用原文仍需两次音频推进和独立词时间确认。
            hints = list(_hints(observation_text))
        hint = hints[-1][1] if hints else None
        if hint is None:
            self._hint = None
            self._confirmations = 0
            return None
        if audio_samples <= self._last_audio:
            return None
        self._last_audio = audio_samples
        self._confirmations = self._confirmations + 1 if hint == self._hint else 1
        self._hint = hint
        attempt = (hint, audio_samples // 16000)
        if audio_samples < 32000 or self._confirmations < 2 or attempt == self._last_attempt:
            return None
        self._last_attempt = attempt
        return hint

    @staticmethod
    def locate(hint, result):
        """从真实词文本匹配锚点，再返回该词的结束位置。"""
        if result.timestamp_status != "reported" or not result.words:
            return None
        prefix = ""
        for word in result.words:
            prefix += word.text
            units = _units(prefix, hint.character_mode)
            if units[-len(hint.anchor) :] == hint.anchor:
                return word.end_sample
        return None
