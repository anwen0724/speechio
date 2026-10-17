"""以有界观察证据提交显示前缀，最终正文只用于一致性诊断。"""

import unicodedata
from collections import deque
from dataclasses import dataclass
from difflib import SequenceMatcher
from math import isfinite

from realtime_asr.core.errors import ASRError


@dataclass(frozen=True, kw_only=True)
class StabilizationConfig:
    """字符、空格与标点分别确认；音频不推进时不增加确认。"""

    min_char_confirmations: int = 2
    min_char_evidence_span: float = 0.60
    initial_prefix_confirmations: int = 2
    initial_prefix_span: float = 0.80
    space_confirmations: int = 4
    space_span: float = 0.75
    punctuation_confirmations: int = 4
    right_context_chars: int = 2
    evidence_window: float = 8
    max_observations: int = 128
    outlier_similarity: float = 0.35
    language_mode: str = "auto"

    def __post_init__(self):
        counts = (
            self.min_char_confirmations,
            self.initial_prefix_confirmations,
            self.space_confirmations,
            self.punctuation_confirmations,
            self.right_context_chars,
            self.max_observations,
        )
        seconds = (
            self.min_char_evidence_span,
            self.initial_prefix_span,
            self.space_span,
            self.evidence_window,
        )
        if (
            any(type(value) is not int or value <= 0 for value in counts)
            or any(
                not isinstance(value, (float, int))
                or isinstance(value, bool)
                or not isfinite(value)
                or value < 0
                for value in seconds
            )
            or not isinstance(self.outlier_similarity, (float, int))
            or not 0 <= self.outlier_similarity <= 1
            or self.language_mode not in ("auto", "character", "space-tokenized")
        ):
            raise ASRError("INVALID_CONFIGURATION", "configure", "稳定文本配置无效")


@dataclass(frozen=True)
class TextObservation:
    """一个模型观察；序号独立于消费顺序，终点为单元内标准样本。"""

    unit: object
    sequence: int
    text: str
    audio_end_sample: int
    completed_at_monotonic: float


@dataclass(frozen=True)
class StabilizationResult:
    """完整显示状态可恢复，增量只说明本次新增部分。"""

    accepted: bool
    ignored_reason: str | None
    raw_text: str
    stable_text: str
    stable_delta: str
    unstable_text: str
    display_text: str
    stable_prefix_conflict: bool = False
    confirmation_count: int = 0
    evidence_span: float = 0
    contributing_sequences: tuple[int, ...] = ()
    retained_observations: int = 0


@dataclass(frozen=True)
class FinalizationResult:
    """报告最终差异，不修订稳定历史，也不改写最终正文。"""

    stable_text: str
    final_text: str
    final_suffix_after_stable: str
    agrees_with_stable_prefix: bool
    mismatch_reason: str | None


@dataclass(frozen=True)
class TextProjection:
    """将比较位置映射回原文；规范化不会代替显示正文。"""

    raw: str
    text: str
    starts: tuple[int, ...]
    ends: tuple[int, ...]

    def punctuation_before(self, offset):
        if offset <= 0 or offset >= len(self.starts):
            return ""
        raw = self.raw[self.ends[offset - 1] : self.starts[offset]]
        return "".join(char for char in raw if unicodedata.category(char).startswith("P"))


def project_text(raw):
    """统一 NFKC、大小写和空白，同时保留展开字符的原文位置。"""
    chars, starts, ends = [], [], []
    for index, original in enumerate(raw):
        for char in unicodedata.normalize("NFKC", original).casefold():
            if unicodedata.category(char).startswith("P"):
                continue
            if char.isspace():
                if not chars or chars[-1] == " ":
                    continue
                char = " "
            chars.append(char)
            starts.append(index)
            ends.append(index + 1)
    if chars and chars[-1] == " ":
        chars.pop()
        starts.pop()
        ends.pop()
    return TextProjection(raw, "".join(chars), tuple(starts), tuple(ends))


class TextStabilizer:
    """隔离一个单元的证据，隐藏规范化、突变与原文前缀提交。"""

    def __init__(self, unit, config=None):
        self.unit = unit
        self.config = config if config is not None else StabilizationConfig()
        if not isinstance(self.config, StabilizationConfig):
            raise ASRError("INVALID_CONFIGURATION", "configure", "需要 StabilizationConfig")
        self._records = deque(maxlen=self.config.max_observations)
        self._outlier = None
        self._sequence = 0
        self._stable = ""
        self._comparison = ""
        self._unstable = ""
        self._display = ""
        self._finalized = False
        self._snapshot = self._result(False, "no-observation", "")

    def snapshot(self) -> StabilizationResult:
        """返回不可变显示结果，不暴露证据容器。"""
        return self._snapshot

    def observe(self, observation: TextObservation) -> StabilizationResult:
        """只有当前归属和新序号能进入证据；重复音频仍可更新尾部。"""
        if observation.unit != self.unit:
            return self._result(False, "wrong-unit", observation.text)
        if self._finalized:
            return self._result(False, "finalized", observation.text)
        if observation.sequence <= self._sequence:
            return self._result(False, "stale-sequence", observation.text)
        self._sequence = observation.sequence
        projection = project_text(observation.text)
        if not projection.text:
            return self._result(False, "empty-text", observation.text)
        recent = list(self._records)[-5:]
        if recent and not any(self._similar(projection.text, item[1].text) for item in recent):
            if (
                not self._comparison
                and self._outlier
                and self._similar(projection.text, self._outlier[1].text)
            ):
                self._records.clear()
                self._records.append(self._outlier)
            else:
                self._outlier = (observation, projection)
                return self._result(False, "outlier", observation.text)
        self._outlier = None
        self._records.append((observation, projection))
        oldest = observation.completed_at_monotonic - self.config.evidence_window
        while self._records and self._records[0][0].completed_at_monotonic < oldest:
            self._records.popleft()
        conflict = bool(self._comparison and not projection.text.startswith(self._comparison))
        delta, points = "", []
        if not conflict:
            frontier = len(self._comparison)
            character_mode = self.config.language_mode == "character" or (
                self.config.language_mode == "auto"
                and any("\u3400" <= char <= "\u9fff" for char in projection.text)
            )
            while frontier < len(projection.text):
                char = projection.text[frontier]
                is_space = char == " " and not character_mode
                count, span = self._threshold(is_space)
                candidate_points = self._evidence(frontier, char)
                if not self._passes(candidate_points, count, span):
                    break
                if is_space and not self._right_confirmed(frontier + 1, projection.text):
                    break
                points = candidate_points
                frontier += 1
            for index in range(len(self._comparison), frontier):
                punctuation = projection.punctuation_before(index)
                if (
                    punctuation
                    and self._passes(
                        self._evidence(index, punctuation, punctuation=True),
                        self.config.punctuation_confirmations,
                        self.config.min_char_evidence_span,
                    )
                    and self._right_confirmed(index, projection.text)
                ):
                    delta += punctuation
                # 一个原始字符可展开多个比较字符，但只能显示一次。
                if index == 0 or projection.ends[index] != projection.ends[index - 1]:
                    delta += projection.raw[projection.starts[index] : projection.ends[index]]
            self._stable += delta
            self._comparison = projection.text[:frontier]
            raw_end = projection.ends[frontier - 1] if frontier else 0
            self._unstable = projection.raw[raw_end:]
        else:
            self._unstable = " ... " + projection.raw.strip()
        self._display = self._stable + self._unstable
        self._snapshot = self._result(True, None, observation.text, delta, conflict, points)
        return self._snapshot

    def finalize(self, final_text: str) -> FinalizationResult:
        """最终原文保持不变；比较只确定前缀诊断与原文剩余部分。"""
        projection = project_text(final_text)
        agrees = projection.text.startswith(self._comparison)
        end = projection.ends[len(self._comparison) - 1] if self._comparison and agrees else 0
        self._finalized = True
        return FinalizationResult(
            self._stable,
            final_text,
            final_text[end:] if agrees else "",
            agrees,
            None if agrees else "stable-prefix-mismatch",
        )

    def _similar(self, left, right):
        return (
            left.startswith(right)
            or right.startswith(left)
            or SequenceMatcher(None, left, right, autojunk=False).ratio()
            >= self.config.outlier_similarity
        )

    def _threshold(self, space=False):
        if space:
            return self.config.space_confirmations, self.config.space_span
        if not self._comparison:
            return self.config.initial_prefix_confirmations, self.config.initial_prefix_span
        return self.config.min_char_confirmations, self.config.min_char_evidence_span

    def _evidence(self, offset, char, *, punctuation=False):
        """逐项保留向前音频证据，固定音频补识别不能重复确认。"""
        points, last_end = [], -1
        for observation, projection in self._records:
            match = (
                projection.punctuation_before(offset)
                if punctuation
                else (projection.text[offset] if offset < len(projection.text) else None)
            )
            if match == char and observation.audio_end_sample > last_end:
                points.append(observation)
                last_end = observation.audio_end_sample
        return points

    @staticmethod
    def _passes(points, count, span):
        return len(points) >= count and (
            points[-1].completed_at_monotonic - points[0].completed_at_monotonic >= span
        )

    def _right_confirmed(self, offset, text):
        count, span = self._threshold()
        confirmed = 0
        for index in range(offset, len(text)):
            if text[index] != " " and self._passes(self._evidence(index, text[index]), count, span):
                confirmed += 1
                if confirmed >= self.config.right_context_chars:
                    return True
        return False

    def _result(self, accepted, reason, raw, delta="", conflict=False, points=()):
        return StabilizationResult(
            accepted,
            reason,
            raw,
            self._stable,
            delta,
            self._unstable,
            self._display,
            conflict,
            len(points),
            points[-1].completed_at_monotonic - points[0].completed_at_monotonic if points else 0,
            tuple(point.sequence for point in points),
            len(self._records),
        )
