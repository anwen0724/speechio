"""准备操作使用各自的时钟，下载、校验和加载不合并计时。"""

from contextlib import contextmanager
from time import perf_counter_ns
from uuid import uuid4

from speechio.errors import error_record


@contextmanager
def preparation_step(steps, phase):
    """返回可独立定位的阶段记录；错误仍向调用方传播。"""
    origin = perf_counter_ns()
    record = {
        "phase": phase,
        "clock_origin_id": uuid4().hex,
        "started_ns": 0,
        "completed_ns": None,
        "status": "ok",
        "error": None,
    }
    try:
        yield record
    except Exception as exc:
        record.update(status="failed", error=error_record(exc, phase))
        exc.preparation_steps = steps
        raise
    finally:
        record["completed_ns"] = perf_counter_ns() - origin
        steps.append(record)
