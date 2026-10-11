"""100 ms 采样观察峰值，不承诺捕获真实瞬时最大需求。"""

import csv
import os
import subprocess
from threading import Event, Thread
from time import perf_counter_ns


class ResourceSampler:
    """自有进程资源采样；短阶段也保留首尾观测。"""

    def __init__(self, *, gpu=False, origin_ns=None):
        import psutil

        self.process = psutil.Process(os.getpid())
        self.gpu = gpu
        self.origin_ns = origin_ns if origin_ns is not None else perf_counter_ns()
        self.samples = []
        self._stop = Event()
        self._thread = None

    def _sample(self):
        rss, gpu_bytes, scope, reason = None, None, None, None
        try:
            rss = self.process.memory_info().rss
        except Exception as exc:
            reason = str(exc)
        if self.gpu:
            try:
                output = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-compute-apps=pid,used_gpu_memory",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=1,
                    check=True,
                )
                for pid, memory in csv.reader(output.stdout.splitlines()):
                    if int(pid) == self.process.pid:
                        gpu_bytes, scope = int(memory.strip()) * 1024 * 1024, "process"
                if gpu_bytes is None:
                    reason = "驱动未提供当前进程显存数值"
            except (OSError, ValueError, subprocess.SubprocessError) as exc:
                reason = str(exc)
        self.samples.append(
            {
                "process_id": self.process.pid,
                "observed_ns": perf_counter_ns() - self.origin_ns,
                "rss_bytes": rss,
                "gpu_memory_bytes": gpu_bytes,
                "gpu_memory_scope": scope,
                "interval_ns": 100_000_000,
                "status": "ok",
                "unavailable_reason": reason,
            }
        )

    def _run(self):
        next_due = self.origin_ns + 100_000_000
        while not self._stop.wait(max(0, (next_due - perf_counter_ns()) / 1_000_000_000)):
            self._sample()
            next_due += 100_000_000
            # 采样本身慢于周期时跳过过期机会，不补造过去的数据。
            if next_due < perf_counter_ns():
                next_due = perf_counter_ns() + 100_000_000

    def __enter__(self):
        self._sample()
        self._thread = Thread(target=self._run, name="asr-resource-sampler", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._stop.set()
        self._thread.join()
        self._sample()
