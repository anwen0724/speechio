"""有限尾部的文本修复；模型输入、请求归属与最终完成由会话负责。"""

import re
from dataclasses import dataclass

from realtime_asr.core.errors import ASRError
from realtime_asr.core.text.stabilizer import project_text


@dataclass(frozen=True)
class TailRepairResult:
    """保留可用正文和对齐证据，失败不触发整轮推理补救。"""

    text: str
    status: str
    matched: bool = False
    used_fuzzy_match: bool = False
    anchor_length: int = 0
    distance: int = 0


@dataclass(frozen=True)
class _Unit:
    value: str
    start: int
    end: int
    han: bool


def _units(text):
    """比较使用规范化单位，替换位置仍使用原文下标。"""
    projection = project_text(text)
    return [
        _Unit(
            match.group(),
            projection.starts[match.start()],
            projection.ends[match.end() - 1],
            any("\u3400" <= char <= "\u9fff" for char in match.group()),
        )
        for match in re.finditer(r"[\u3400-\u9fff]|[^\W_\u3400-\u9fff]+", projection.text)
    ]


def _join(prefix, suffix):
    prefix, suffix = prefix.rstrip(), suffix.lstrip()
    if not prefix or not suffix:
        return prefix + suffix
    adjacent_han = "\u3400" <= prefix[-1] <= "\u9fff" or "\u3400" <= suffix[0] <= "\u9fff"
    return prefix + ("" if adjacent_han else " ") + suffix


def _exact_anchor(live, tail):
    """优先最长且最靠右的锚点；短同词不能证明两个音频片段重叠。"""
    for length in range(min(8, len(live)), 2, -1):
        anchor = live[-length:]
        han = any(unit.han for unit in anchor)
        if length < (4 if han else 3) or length > (8 if han else 4):
            continue
        values = [unit.value for unit in anchor]
        for end in range(len(tail), length - 1, -1):
            if values == [unit.value for unit in tail[end - length : end]]:
                return end - length, end, length
    return None


def _partial_before_exact_suffix(live_text, tail_text, live, tail):
    """两个以上精确后续词证明一个截断词的补全，不允许任意换词。"""
    for length in range(min(3, len(live) - 1), 1, -1):
        partial = live[-length - 1]
        anchor = live[-length:]
        if partial.han or any(unit.han for unit in anchor) or len(partial.value) < 2:
            continue
        for end in range(len(tail), length, -1):
            start = end - length
            completed = tail[start - 1]
            if (
                not completed.han
                and completed.value != partial.value
                and completed.value.startswith(partial.value)
                and [unit.value for unit in anchor] == [unit.value for unit in tail[start:end]]
            ):
                text = _join(live_text[: partial.start], tail_text[completed.start :])
                return TailRepairResult(text, "fuzzy", True, True, length + 1, 1)
    return None


def _one_edit(left, right):
    """一次字符编辑的窄判定，不构建通用字符串距离服务。"""
    if abs(len(left) - len(right)) > 1:
        return False
    for index, (first, second) in enumerate(zip(left, right)):
        if first == second:
            continue
        if len(left) == len(right):
            return left[index + 1 :] == right[index + 1 :]
        if len(left) > len(right):
            return left[index + 1 :] == right[index:]
        return left[index:] == right[index + 1 :]
    return True


def _fuzzy_anchor(live, tail):
    for length in range(min(4, len(live)), 2, -1):
        anchor = live[-length:]
        if any(unit.han for unit in anchor):
            continue
        for end in range(len(tail), length - 1, -1):
            window = tail[end - length : end]
            differences = [
                (index, left, right)
                for index, (left, right) in enumerate(zip(anchor, window, strict=True))
                if left.value != right.value
            ]
            if len(differences) != 1:
                continue
            index, left, right = differences[0]
            size = min(len(left.value), len(right.value))
            completion = index == length - 1 and size >= 2 and right.value.startswith(left.value)
            close = size >= 4 and _one_edit(left.value, right.value)
            if not right.han and (completion or close):
                return end - length, length
    return None


def repair_tail(live_text: str, tail_text: str) -> TailRepairResult:
    """准确前缀与独立尾部对齐，返回修复结果而不修改任何会话状态。"""
    if not isinstance(live_text, str) or not isinstance(tail_text, str):
        raise ASRError("INVALID_CONFIGURATION", "repair_tail", "实时与尾部正文应为字符串")
    live_text, tail_text = live_text.strip(), tail_text.strip()
    if not tail_text:
        return TailRepairResult(live_text, "empty")
    if not live_text:
        return TailRepairResult(tail_text, "tail_only")
    live, tail = _units(live_text), _units(tail_text)
    partial = _partial_before_exact_suffix(live_text, tail_text, live, tail)
    if partial is not None:
        return partial
    anchor = _exact_anchor(live, tail)
    if anchor is not None:
        _, end, length = anchor
        suffix = tail_text[tail[end].start :] if end < len(tail) else ""
        return TailRepairResult(_join(live_text, suffix), "exact", True, anchor_length=length)
    if live and not live[-1].han:
        stable_anchor = _exact_anchor(live[:-1], tail)
        if stable_anchor is not None:
            _, end, length = stable_anchor
            if end < len(tail):
                text = _join(live_text[: live[-1].start], tail_text[tail[end].start :])
                return TailRepairResult(text, "fuzzy", True, True, length, 1)
    fuzzy = _fuzzy_anchor(live, tail)
    if fuzzy is not None:
        start, length = fuzzy
        text = _join(live_text[: live[-length].start], tail_text[tail[start].start :])
        return TailRepairResult(text, "fuzzy", True, True, length, 1)
    return TailRepairResult(live_text, "alignment_failed")
