"""双实时文本合并：准确正文拥有基准，极速流只补充有锚点的有限尾部。"""

import re
from dataclasses import dataclass, replace

from speechio.errors import ASRError
from speechio.stt.text.stabilizer import TextObservation, project_text


@dataclass(frozen=True, kw_only=True)
class RealtimeMergeConfig:
    """尾部容量与模糊确认；模型选择和执行周期由运行时角色配置负责。"""

    max_tail_units: int = 5
    fuzzy_confirmations: int = 1

    def __post_init__(self):
        if any(
            type(value) is not int or value <= 0
            for value in (self.max_tail_units, self.fuzzy_confirmations)
        ):
            raise ASRError(
                "INVALID_CONFIGURATION", "configure_merge", "合并容量与确认次数应为正整数"
            )


@dataclass(frozen=True)
class RealtimeMergeResult:
    """可独立恢复的合并快照；两路序号和音频覆盖不互相替代。"""

    turn: object
    text: str = ""
    accurate_text: str = ""
    ultrafast_text: str = ""
    ultrafast_suffix: str = ""
    status: str = "waiting_for_accurate"
    revision: int = 0
    accurate_generation: int = 0
    accurate_sequence: int = 0
    ultrafast_sequence: int = 0
    accurate_audio_end_sample: int | None = None
    ultrafast_audio_end_sample: int | None = None
    suffix_sequence: int | None = None
    suffix_audio_end_sample: int | None = None
    matched: bool = False
    used_fuzzy_match: bool = False
    anchor_length: int = 0
    distance: int = 0
    held: bool = False
    accepted: bool = True
    ignored_reason: str | None = None


@dataclass(frozen=True)
class _Token:
    value: str
    raw: str
    character: bool


@dataclass(frozen=True)
class _Anchor:
    end: int
    length: int
    status: str = "exact"
    distance: int = 0


@dataclass(frozen=True)
class _Frontier:
    """单调路径的证据；间隔与词关系分开累计，禁止任意替换。"""

    score: int = 0
    supports: int = 0
    strong: int = 0
    distance: int = 0
    gaps: int = 0
    end: int = 0

    @property
    def rank(self):
        return self.score, self.supports, -self.distance, -self.gaps, self.end


def _tokens(text):
    """比较位置映射回原文；标点只影响显示，不成为锚点证据。"""
    projection = project_text(text)
    matches = list(re.finditer(r"[\u3400-\u9fff]|[^\W_\u3400-\u9fff]+", projection.text))
    result = []
    for index, match in enumerate(matches):
        start = projection.starts[match.start()]
        end = (
            projection.starts[matches[index + 1].start()] if index + 1 < len(matches) else len(text)
        )
        value = match.group()
        result.append(
            _Token(
                value, text[start:end].strip(), any("\u3400" <= char <= "\u9fff" for char in value)
            )
        )
    return result


def _append_text(left, right):
    if not left or not right:
        return left + right
    characters = "\u3400" <= left[-1] <= "\u9fff" or "\u3400" <= right[0] <= "\u9fff"
    return left + ("" if characters else " ") + right


def _render(tokens):
    text = ""
    for token in tokens:
        text = _append_text(text, token.raw)
    return text


def _one_edit(left, right):
    """只判断一次字符编辑，不为窄匹配建立通用编辑距离框架。"""
    if abs(len(left) - len(right)) > 1:
        return False
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            if len(left) == len(right):
                return left[index + 1 :] == right[index + 1 :]
            if len(left) > len(right):
                return left[index + 1 :] == right[index:]
            return left[index:] == right[index + 1 :]
    return True


def _relation(left, right):
    """返回匹配分值、编辑代价及支持强度；无关系不进入路径。"""
    size = min(len(left), len(right))
    if left == right:
        return 4, 0, int(size >= 2), int(size >= 3)
    if size >= 2 and (left.startswith(right) or right.startswith(left)):
        return 3, 1, 1, int(size >= 3)
    if size >= 3 and _one_edit(left, right):
        return 2, 1, 1, 1
    return None


def _soft_frontier(slow, fast):
    """在八词准确尾部寻找至多四个间隔的单调路径，匹配到末词。"""
    slow = slow[-8:]
    if len(slow) < 2 or len(fast) < 2:
        return None
    paths = [[[None] * 5 for _ in range(len(fast) + 1)] for _ in range(len(slow) + 1)]
    # 极速的开头在重叠区之外；只有证据内部的插入和遗漏消耗间隔。
    for prefix in range(len(fast) + 1):
        paths[0][prefix][0] = _Frontier()
    best = None

    def retain(i, j, gaps, candidate):
        current = paths[i][j][gaps]
        if current is None or candidate.rank > current.rank:
            paths[i][j][gaps] = candidate

    for i in range(len(slow) + 1):
        for j in range(len(fast) + 1):
            for gaps in range(5):
                state = paths[i][j][gaps]
                if state is None:
                    continue
                if gaps < 4:
                    skipped = replace(state, gaps=gaps + 1, distance=state.distance + 1)
                    if i < len(slow):
                        retain(i + 1, j, gaps + 1, skipped)
                    if j < len(fast):
                        retain(i, j + 1, gaps + 1, skipped)
                if i >= len(slow) or j >= len(fast):
                    continue
                relation = _relation(slow[i], fast[j])
                if relation is None:
                    continue
                score, distance, support, strong = relation
                matched = replace(
                    state,
                    score=state.score + score,
                    supports=state.supports + support,
                    strong=state.strong + strong,
                    distance=state.distance + distance,
                    end=j + 1,
                )
                retain(i + 1, j + 1, gaps, matched)
                if (
                    i == len(slow) - 1
                    and matched.supports >= 2
                    and matched.strong >= 2
                    and matched.score >= 6
                ):
                    if best is None or matched.rank > best.rank:
                        best = matched
    return _Anchor(best.end, best.supports, "soft_frontier", best.distance) if best else None


class RealtimeMerger:
    """一个单元的两路状态；隐藏对齐、保留及准确内容代次。"""

    def __init__(self, unit, config=None):
        self.turn = unit
        self.config = config if config is not None else RealtimeMergeConfig()
        if not isinstance(self.config, RealtimeMergeConfig):
            raise ASRError("INVALID_CONFIGURATION", "configure_merge", "需要 RealtimeMergeConfig")
        self._observations = {"accurate": None, "ultrafast": None}
        self._accurate = ""
        self._body_observation = None
        self._suffix_observation = None
        self._accurate_tokens = []
        self._fast_tokens = []
        self._tail = ()
        self._generation = 0
        self._revision = 0
        self._status = "waiting_for_accurate"
        self._anchor = None
        self._pending_key = ()
        self._pending_count = 0

    def observe(self, role, observation: TextObservation):
        """先核对来源与覆盖；过期观察不能改变尾部或确认状态。"""
        if role not in self._observations:
            raise ASRError("INVALID_CONFIGURATION", "merge", "实时角色应为 accurate 或 ultrafast")
        previous = self._observations[role]
        reason = None
        if observation.turn != self.turn:
            reason = "unit_mismatch"
        elif observation.sequence <= (previous.sequence if previous else 0):
            reason = "sequence_stale"
        elif (
            previous
            and previous.audio_end_sample is not None
            and observation.audio_end_sample is not None
            and observation.audio_end_sample < previous.audio_end_sample
        ):
            reason = "audio_regressed"
        if reason:
            return replace(
                self.snapshot(), accepted=False, ignored_reason=reason, status="stale_ignored"
            )
        self._observations[role] = observation
        self._revision += 1
        advanced = False
        if role == "accurate":
            tokens = _tokens(observation.text)
            if tokens:
                self._body_observation = observation
                changed = tuple(token.value for token in tokens) != tuple(
                    token.value for token in self._accurate_tokens
                )
                self._accurate = observation.text.strip()
                if not changed:
                    return self.snapshot()
                advanced = self._generation > 0
                self._generation += 1
                self._accurate_tokens = tokens
                self._tail = ()
                self._suffix_observation = None
                self._clear_confirmation()
            elif self._accurate:
                return self.snapshot()
        else:
            self._fast_tokens = _tokens(observation.text)
        self._merge(advanced=advanced)
        return self.snapshot()

    def _find_anchor(self):
        """优先完整末尾的长精确锚点；重复出现时取极速文本的最后一次。"""
        slow = tuple(token.value for token in self._accurate_tokens)
        fast = tuple(token.value for token in self._fast_tokens)
        characters = tuple(token.character for token in self._accurate_tokens)
        if self._generation == 1 and len(slow) == 1 and not characters[0] and fast[:1] == slow:
            return _Anchor(1, 1, "bootstrap_exact")
        for length in range(min(8, len(slow)), 1, -1):
            if length < 4 and any(characters[-length:]):
                continue
            for end in range(len(fast), length - 1, -1):
                if slow[-length:] == fast[end - length : end]:
                    return _Anchor(end, length)
        # 混合正文只用末尾连续英文词的模糊证据，不越过汉字边界。
        last_character = next(
            (index for index in range(len(characters) - 1, -1, -1) if characters[index]), -1
        )
        english_tail = slow[last_character + 1 :]
        if len(english_tail) >= 2:
            for length in range(min(8, len(english_tail)), 2, -1):
                for end in range(len(fast), length - 1, -1):
                    pairs = list(zip(slow[-length:], fast[end - length : end], strict=True))
                    differences = [
                        (index, left, right)
                        for index, (left, right) in enumerate(pairs)
                        if left != right
                    ]
                    if len(differences) != 1:
                        continue
                    index, left, right = differences[0]
                    completion = index == length - 1 and right.startswith(left)
                    if min(len(left), len(right)) >= 3 and (completion or _one_edit(left, right)):
                        return _Anchor(end, length, "fuzzy", 1)
            return _soft_frontier(english_tail, fast)
        return None

    def _clear_confirmation(self):
        self._pending_key = ()
        self._pending_count = 0

    def _merge(self, *, advanced):
        if not self._accurate_tokens:
            self._status = "waiting_for_accurate"
            return
        self._anchor = self._find_anchor()
        if self._anchor is None:
            self._clear_confirmation()
            self._status = (
                "accurate_advanced_no_anchor"
                if advanced
                else ("held_no_anchor" if self._observations["ultrafast"] else "accurate_only")
            )
            return
        candidate = tuple(
            self._fast_tokens[self._anchor.end : self._anchor.end + self.config.max_tail_units]
        )
        old = tuple(token.value for token in self._tail)
        new = tuple(token.value for token in candidate)
        if old and new[: len(old)] != old:
            self._clear_confirmation()
            self._status = "held_fast_conflict"
            return
        if self._anchor.distance > 0 and candidate:
            if self._pending_key and new[: len(self._pending_key)] == self._pending_key:
                self._pending_count += 1
            else:
                self._pending_count = 1
            self._pending_key = new
            if self._pending_count < self.config.fuzzy_confirmations:
                self._status = "held_fuzzy_pending"
                return
        self._clear_confirmation()
        if len(candidate) > len(self._tail):
            self._suffix_observation = self._observations["ultrafast"]
        self._tail += candidate[len(self._tail) :]
        self._status = self._anchor.status if candidate else "accurate_only"

    def snapshot(self):
        """返回完整状态；调用方无需重放增量即可恢复当前显示。"""
        accurate = self._body_observation
        fast = self._observations["ultrafast"]
        origin = self._suffix_observation
        suffix = _render(self._tail)
        return RealtimeMergeResult(
            turn=self.turn,
            text=_append_text(self._accurate, suffix),
            accurate_text=self._accurate,
            ultrafast_text=fast.text if fast else "",
            ultrafast_suffix=suffix,
            status=self._status,
            revision=self._revision,
            accurate_generation=self._generation,
            accurate_sequence=accurate.sequence if accurate else 0,
            ultrafast_sequence=fast.sequence if fast else 0,
            accurate_audio_end_sample=accurate.audio_end_sample if accurate else None,
            ultrafast_audio_end_sample=fast.audio_end_sample if fast else None,
            suffix_sequence=origin.sequence if origin else None,
            suffix_audio_end_sample=origin.audio_end_sample if origin else None,
            matched=self._anchor is not None and not self._status.startswith("held_"),
            used_fuzzy_match=self._anchor is not None and self._anchor.distance > 0,
            anchor_length=self._anchor.length if self._anchor else 0,
            distance=self._anchor.distance if self._anchor else 0,
            held=self._status.startswith("held_"),
        )
