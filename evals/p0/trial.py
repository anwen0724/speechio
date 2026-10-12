"""一个试次拥有一个单调时钟、操作记录和原始观察。"""

from dataclasses import asdict
from time import perf_counter_ns
from uuid import uuid4

from realtime_asr.errors import error_record


class Trial:
    """计时覆盖真实操作与结果转换，记录写盘发生在测量之后。"""

    def __init__(self, sample_count, *, clock=perf_counter_ns):
        self.trial_id = uuid4().hex
        self.clock = clock
        self.origin_ns = clock()
        self.sample_count = sample_count
        self.calls = []
        self.observations = []
        self.timestamps = {"input_end_due_ns": round(sample_count * 1_000_000_000 / 16000)}
        self.status = "ok"
        self.confirmed_end_sample = 0
        self.error = None
        self.cleanup = None

    def now(self):
        """返回本试次的整数纳秒，不混用准备阶段时钟。"""
        return self.clock() - self.origin_ns

    def call(self, operation, action, *, parameter_role=None):
        """异常时仍保存实际调用区间；原异常继续交给试次入口。"""
        record = {"operation": operation, "started_ns": self.now(), "status": "ok", "error": None}
        if parameter_role is not None:
            record["parameter_role"] = parameter_role
        try:
            return action()
        except KeyboardInterrupt as exc:
            record["status"] = "cancelled"
            record["error"] = error_record(exc, operation)
            record["error"]["error_code"] = "CANCELLED"
            raise
        except Exception as exc:
            record["status"] = "failed"
            record["error"] = error_record(exc, operation)
            raise
        finally:
            record["completed_ns"] = self.now()
            self.calls.append(record)

    def observe(self, result, output_kind, endpoint, *, stream_id=None, snapshot_id=None, **fields):
        """接受终点只表示输入归属，不推测已解码音频的终点。"""
        call = self.calls[-1]
        observed = self.now()
        record = {
            "record_type": "observation",
            "observation_index": len(self.observations),
            "output_kind": output_kind,
            "accepted_range": [0, endpoint],
            "decoded_end_sample": None,
            "block_range": None,
            "due_ns": None,
            "accepted_start_ns": None,
            "accepted_end_ns": None,
            "call_started_ns": call["started_ns"],
            "call_completed_ns": call["completed_ns"],
            "observed_ns": observed,
            "result": asdict(result),
            "status": "ok",
            "stream_id": stream_id,
            "snapshot_id": snapshot_id,
            **fields,
        }
        self.observations.append(record)
        if "partial" in output_kind and result.text.strip():
            self.timestamps.setdefault("first_nonempty_partial_ns", observed)
        if output_kind.endswith("final"):
            self.timestamps["final_received_ns"] = observed
        return record

    def fail(self, exc, operation, confirmed=0, unconfirmed=None):
        """保留已经形成的观察，只终结当前试次。"""
        self.status = "failed"
        self.error = error_record(exc, operation, [0, confirmed], unconfirmed)

    def cancel(self, source=None):
        """停止新工作，不将请求时刻当作资源关闭时刻。"""
        self.status = "cancelled"
        requested = getattr(source, "requested_at_ns", None)
        elapsed = max(0, requested - self.origin_ns) if requested is not None else self.now()
        self.timestamps.setdefault("cancel_requested_ns", elapsed)
        for call in self.calls:
            if call["status"] == "ok" and call["completed_ns"] >= elapsed:
                call["status"] = "cancelled"

    def close_stream(self, stream):
        """清理错误单独保存，不覆盖原推理错误。"""
        if self.cleanup is not None:
            return
        try:
            stream.close()
        except Exception as exc:
            if self.error is None:
                self.fail(exc, "close", stream.accepted_samples)
        self.timestamps["close_completed_ns"] = self.now()
        self.cleanup = {**stream.cleanup, "completed_ns": self.timestamps["close_completed_ns"]}
