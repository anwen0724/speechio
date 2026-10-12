"""一个音频单元的候选账本，隐藏快照版本、最新请求和发布有效性。"""

from dataclasses import dataclass
from hashlib import sha256
from typing import TYPE_CHECKING
from uuid import uuid4

from realtime_asr.text.tail_repair import repair_tail

if TYPE_CHECKING:
    from realtime_asr.session.session import SessionEvent


@dataclass(frozen=True)
class CandidateRequest:
    """公开接纳凭据，标准输入范围与继续边界分别记录。"""

    unit: object
    request_id: str
    kind: str
    hint: bool
    start_sample: int
    end_sample: int
    audio_version: int
    resume_epoch: int = 0
    resume_id: str | None = None
    candidate_start_sample: int = 0


@dataclass(frozen=True)
class CandidateResult:
    """候选正文与原始模型结果并存，不伪装成覆盖新音频的最终结果。"""

    request: CandidateRequest
    text: str
    live_text: str
    model_text: str
    status: str
    current: bool
    matched: bool = False
    used_fuzzy_match: bool = False
    anchor_length: int = 0
    distance: int = 0
    attempts: int = 1
    added_silence_samples: int = 0
    replaces_task_id: str | None = None


@dataclass(frozen=True)
class ResumeAck:
    """同轮继续的标准采样边界；字节位置由 PCM16 乘二取得。"""

    unit: object
    resume_epoch: int
    resume_id: str
    candidate_start_sample: int


@dataclass(frozen=True)
class _Pending:
    request: CandidateRequest
    live_text: str
    pcm_digest: bytes


@dataclass
class RetainedCorrection:
    """最近完成单元的受限修正上下文，音频与原任务关系不可混用。"""

    unit: object
    task_id: str
    pcm: bytes
    state: "CandidateState"
    event: "SessionEvent | None" = None


class CandidateState:
    """只保存每种候选的最新归属，音频副本由任务或封口上下文拥有。"""

    def __init__(self, unit, *, epoch=0, resume_id=None, start_sample=0):
        self.unit = unit
        self.epoch = epoch
        self.resume_id = resume_id
        self.start_sample = start_sample
        self._latest = {}
        self._results = {}

    def describe(self, kind, hint, start, end):
        return CandidateRequest(
            self.unit,
            uuid4().hex,
            kind,
            hint,
            start,
            end,
            end,
            self.epoch,
            self.resume_id,
            self.start_sample,
        )

    def replacing(self, kind):
        previous = self._latest.get(kind)
        return previous.request.request_id if previous is not None else None

    @property
    def pending_count(self):
        return sum(kind not in self._results for kind in self._latest)

    def accept(self, request, live_text, pcm):
        self._latest[request.kind] = _Pending(request, live_text, sha256(pcm).digest())
        self._results.pop(request.kind, None)

    def resolve(self, request_id, result, current_samples):
        """旧请求无法修改最新缓存；真实输入范围不由返回正文估计。"""
        pending = next(
            (item for item in self._latest.values() if item.request.request_id == request_id), None
        )
        if pending is None or pending.request.resume_epoch != self.epoch:
            return None
        request = pending.request
        text = result.text.strip()
        execution = result.backend_metadata.get("candidate_execution", {})
        repaired = repair_tail(pending.live_text, text) if request.kind == "tail" else None
        value = CandidateResult(
            request=request,
            text=repaired.text if repaired else text,
            live_text=pending.live_text,
            model_text=text,
            status=repaired.status if repaired else (request.kind if text else "empty"),
            current=request.end_sample == current_samples,
            matched=repaired.matched if repaired else False,
            used_fuzzy_match=repaired.used_fuzzy_match if repaired else False,
            anchor_length=repaired.anchor_length if repaired else 0,
            distance=repaired.distance if repaired else 0,
            attempts=execution.get("attempts", 1),
            added_silence_samples=execution.get("added_silence_samples", 0),
        )
        self._results[request.kind] = value, result, pending.pcm_digest
        return value

    def reusable_whole(self, pcm):
        """版本与实际字节均一致才复用，提示或尾部结果不能终结整轮。"""
        cached = self._results.get("whole")
        if cached is None:
            return None
        value, result, digest = cached
        request = value.request
        if (
            request.resume_epoch == self.epoch
            and not request.hint
            and request.end_sample == len(pcm) // 2
            and value.status == "whole"
            and result.text.strip()
            and digest == sha256(pcm).digest()
        ):
            return result
        return None

    def clear(self):
        ids = tuple(item.request.request_id for item in self._latest.values())
        self._latest.clear()
        self._results.clear()
        return ids

    def resume(self, samples):
        """继续只推进归属屏障，不拥有或清空历史音频。"""
        invalidated = self.clear()
        self.epoch += 1
        self.resume_id = uuid4().hex
        self.start_sample = samples
        return ResumeAck(self.unit, self.epoch, self.resume_id, samples), invalidated
