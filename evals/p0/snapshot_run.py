"""Whisper 前缀快照回放；单工作线程串行访问同一模型。"""

from concurrent.futures import ThreadPoolExecutor
from threading import Event

from evals.p0.snapshots import SnapshotSchedule
from evals.p0.trial import Trial
from realtime_asr.core.audio import as_audio
from realtime_asr.core.errors import error_record


def run_snapshots(engine, audio, language, *, cancel=None, trial=None):
    """输入结束独立于推理完成，最后重新识别完整音频。"""
    array = as_audio(audio)
    cancel = cancel if cancel is not None else Event()
    trial = trial if trial is not None else Trial(array.size)
    schedule = SnapshotSchedule(array.size)
    future, active = None, None

    def transcribe_prefix(request):
        # 在线程实际开始时确定范围，副本不会随后续回放变化。
        elapsed = trial.now()
        if elapsed >= schedule.end_due_ns or cancel.is_set():
            request.update(
                status="skipped", reason="discarded_at_end" if not cancel.is_set() else "cancelled"
            )
            return None
        end = min(array.size, elapsed * 16000 // 1_000_000_000)
        request["accepted_range"] = [0, end]
        prefix = array[:end].copy()
        return trial.call(
            "transcribe",
            lambda: engine.transcribe(prefix, language=language, role="snapshot"),
            parameter_role="snapshot",
        )

    def consume_active():
        result = future.result()
        if result is not None:
            kind = "late_result" if cancel.is_set() else "snapshot_partial"
            trial.observe(
                result,
                kind,
                active["accepted_range"][1],
                snapshot_id=active["snapshot_id"],
                due_ns=active["due_ns"],
            )
            active["status"] = "ok" if not cancel.is_set() else "cancelled"

    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="asr-snapshot") as worker:
        try:
            while not schedule.ended:
                elapsed = trial.now()
                schedule.advance(elapsed)
                if cancel.is_set():
                    trial.cancel(cancel)
                    schedule.discard("cancelled")
                    break
                if future is not None and future.done():
                    consume_active()
                    future, active = None, None
                if schedule.ended:
                    break
                if future is None:
                    active = schedule.start(trial.now())
                    if active is not None:
                        future = worker.submit(transcribe_prefix, active)
                # 只等待下个机会或短轮询，不等待推理才推进音频时钟。
                next_due = min(
                    schedule.next_end_sample * 1_000_000_000 // 16000, schedule.end_due_ns
                )
                cancel.wait(min(0.01, max(0, (next_due - trial.now()) / 1_000_000_000)))
            if trial.status == "ok":
                trial.timestamps["finish_called_ns"] = schedule.end_due_ns
                trial.timestamps["stop_observed_ns"] = trial.now()
            if future is not None:
                consume_active()
                future = None
            if cancel.is_set():
                trial.cancel(cancel)
            if trial.status == "ok":
                result = trial.call(
                    "transcribe",
                    lambda: engine.transcribe(array, language=language, role="final"),
                    parameter_role="final",
                )
                if cancel.is_set():
                    trial.cancel(cancel)
                trial.observe(
                    result, "late_result" if cancel.is_set() else "snapshot_final", array.size
                )
        except KeyboardInterrupt:
            cancel.set()
            trial.cancel(cancel)
            schedule.discard("cancelled")
            if future is not None:
                try:
                    consume_active()
                except Exception as exc:
                    trial.error = error_record(exc, "transcribe")
        except Exception as exc:
            schedule.discard("failed")
            trial.fail(exc, "transcribe")
            if cancel.is_set():
                trial.cancel(cancel)

    for request in schedule.requests:
        if request["status"] == "skipped":
            trial.observations.append(
                {
                    "record_type": "observation",
                    "observation_index": len(trial.observations),
                    "snapshot_id": request["snapshot_id"],
                    "stream_id": None,
                    "output_kind": None,
                    "result": None,
                    "status": "skipped",
                    "accepted_range": [0, request["requested_end_sample"]],
                    "decoded_end_sample": None,
                    "requested_end_sample": request["requested_end_sample"],
                    "due_ns": request["due_ns"],
                    "reason": request["reason"],
                    "call_started_ns": None,
                    "call_completed_ns": None,
                    "observed_ns": None,
                    "block_range": None,
                    "accepted_start_ns": None,
                    "accepted_end_ns": None,
                }
            )
    trial.timestamps["close_completed_ns"] = trial.now()
    trial.cleanup = {
        "method": "reference_drop",
        "status": "ok",
        "completed_ns": trial.now(),
        "errors": [],
    }
    return trial
