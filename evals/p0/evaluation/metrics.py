"""按 P0 时间点计算指标，不将回放等待计为推理耗时。"""

TIME_FIELDS = (
    "speech_start_ns",
    "speech_end_ns",
    "input_end_due_ns",
    "first_nonempty_partial_ns",
    "finish_called_ns",
    "final_received_ns",
    "call_started_ns",
    "call_completed_ns",
    "close_completed_ns",
)
METRIC_FIELDS = (
    "first_partial_wait_ms",
    "speech_to_final_wait_ms",
    "drain_ms",
    "snapshot_end_to_final_ms",
    "inference_ms",
    "accept_lag_ms",
    "call_ms",
    "rtf",
    "snapshot_rtf",
)


def compute_metrics(trial, sample, *, mode, path):
    """保留不适用字段和原因；语音前误输出保留负值而不截断。"""
    times = {field: trial.timestamps.get(field) for field in TIME_FIELDS}
    intervals = sample["speech_intervals"]
    if intervals:
        times["speech_start_ns"] = round(intervals[0][0] * 1_000_000_000 / 16000)
        times["speech_end_ns"] = round(intervals[-1][1] * 1_000_000_000 / 16000)
    metrics = {field: None for field in METRIC_FIELDS}
    reasons, flags = {}, []
    for name in ("first_partial_wait_ms", "speech_to_final_wait_ms"):
        reasons[name] = (
            "not_paced"
            if mode != "paced"
            else ("no_speech_annotation" if not intervals else "no_successful_result")
        )
    final = times["final_received_ns"]
    first = times["first_nonempty_partial_ns"]
    if mode == "paced" and intervals:
        if first is not None:
            metrics["first_partial_wait_ms"] = (first - times["speech_start_ns"]) / 1_000_000
            if metrics["first_partial_wait_ms"] < 0:
                flags.append("pre_speech_output")
        else:
            reasons["first_partial_wait_ms"] = "no_nonempty_partial"
        if final is not None:
            metrics["speech_to_final_wait_ms"] = (final - times["speech_end_ns"]) / 1_000_000
    if final is not None and times["finish_called_ns"] is not None:
        key = "drain_ms" if path == "stream" else "snapshot_end_to_final_ms"
        metrics[key] = (final - times["finish_called_ns"]) / 1_000_000
    successful = [call for call in trial.calls if call["status"] == "ok"]
    metrics["call_ms"] = [
        {
            "operation": call["operation"],
            "value": (call["completed_ns"] - call["started_ns"]) / 1_000_000,
        }
        for call in successful
    ]
    complete_calls = [
        call
        for call in successful
        if call["operation"] == "transcribe" and call.get("parameter_role") != "snapshot"
    ]
    if complete_calls and trial.status == "ok":
        call = complete_calls[-1]
        times["call_started_ns"], times["call_completed_ns"] = (
            call["started_ns"],
            call["completed_ns"],
        )
        metrics["inference_ms"] = (call["completed_ns"] - call["started_ns"]) / 1_000_000
    if trial.status == "ok" and sample["sample_count"] > 0:
        duration = sample["sample_count"] / 16000
        if path == "whole":
            metrics["rtf"] = metrics["inference_ms"] / 1000 / duration
        elif path == "stream" and mode == "fast":
            metrics["rtf"] = (
                sum(
                    call["completed_ns"] - call["started_ns"]
                    for call in successful
                    if call["operation"] in ("accept_audio", "decode", "get_result", "finish")
                )
                / 1_000_000_000
                / duration
            )
    snapshot_costs = []
    for observation in trial.observations:
        if observation.get("output_kind") == "snapshot_partial":
            duration = observation["accepted_range"][1] / 16000
            snapshot_costs.append(
                {
                    "snapshot_id": observation["snapshot_id"],
                    "value": (observation["call_completed_ns"] - observation["call_started_ns"])
                    / 1_000_000_000
                    / duration,
                }
            )
        if mode == "paced" and observation.get("accepted_start_ns") is not None:
            observation["accept_lag_ms"] = (
                observation["accepted_start_ns"] - observation["due_ns"]
            ) / 1_000_000
    metrics["snapshot_rtf"] = snapshot_costs if path == "snapshot" else None
    for name, value in metrics.items():
        if value is None:
            reasons.setdefault(
                name, "not_applicable" if trial.status == "ok" else "trial_not_successful"
            )
    reasons = {name: reason for name, reason in reasons.items() if metrics[name] is None}
    for name, value in times.items():
        if value is None:
            reasons[name] = (
                "not_available"
                if name not in ("speech_start_ns", "speech_end_ns")
                else "no_speech_annotation"
            )
    return times, metrics, reasons, flags
