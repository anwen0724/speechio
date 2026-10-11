"""按绝对音频时钟组织快照请求，最多保留一个待执行机会。"""

from uuid import uuid4


class SnapshotSchedule:
    """机会时间与模型完成时间分开，不因推理积压移动回放原点。"""

    def __init__(self, sample_count):
        self.sample_count = sample_count
        self.end_due_ns = round(sample_count * 1_000_000_000 / 16000)
        self.next_end_sample = 3200
        self.pending = None
        self.requests = []
        self.ended = False

    def advance(self, elapsed_ns):
        """记录已发生的机会；文件结束优先于同一时刻的新机会。"""
        available = min(self.sample_count, elapsed_ns * 16000 // 1_000_000_000)
        while self.next_end_sample <= available and self.next_end_sample < self.sample_count:
            if self.pending is not None:
                self.pending.update(status="skipped", reason="superseded")
            request = {
                "snapshot_id": uuid4().hex,
                "requested_end_sample": self.next_end_sample,
                "due_ns": self.next_end_sample * 1_000_000_000 // 16000,
                "status": "pending",
                "reason": None,
            }
            self.requests.append(request)
            self.pending = request
            self.next_end_sample += 3200
        if elapsed_ns >= self.end_due_ns:
            self.ended = True
            self.discard("discarded_at_end")

    def start(self, elapsed_ns):
        """在实际启动时固定前缀终点；之后不随输入到达改变。"""
        if self.ended or self.pending is None:
            return None
        request, self.pending = self.pending, None
        endpoint = min(self.sample_count, elapsed_ns * 16000 // 1_000_000_000)
        request.update(status="started", accepted_range=[0, endpoint])
        return request

    def discard(self, reason):
        """结束或取消未启动机会，不生成假文本。"""
        if self.pending is not None:
            self.pending.update(status="skipped", reason=reason)
            self.pending = None
