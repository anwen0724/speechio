"""CLI 的候选请求时钟，只按已投递源音频计时，不承担版本或模型规则。"""

import numpy as np

from speechio.errors import ASRError


class CandidateTicker:
    """本地与远程共用输入时钟，模型计算仍由正式会话异步执行。"""

    def __init__(self, args, format, request):
        self.enabled = args.candidates
        self.kind, self.hint = args.candidate_kind, args.candidate_hint
        self.period_frames = max(1, int(args.candidate_every * format.sample_rate))
        self.frame_bytes = format.channels * (2 if format.sample_type == "int16" else 4)
        self.accepted_frames = 0
        self.next_request = self.period_frames
        self.request = request

    def accept(self, block):
        """一次大块最多提出一次请求，容量不足和自动静音不终止输入。"""
        if not self.enabled:
            return None
        frames = block.shape[0] if isinstance(block, np.ndarray) else len(block) // self.frame_bytes
        self.accepted_frames += frames
        if self.accepted_frames < self.next_request:
            return None
        self.next_request = self.accepted_frames + self.period_frames
        try:
            self.request(kind=self.kind, hint=self.hint)
        except ASRError as error:
            if error.code in ("CANDIDATE_CAPACITY_EXCEEDED", "NO_ACTIVE_UNIT"):
                return error.code
            raise
        return None
