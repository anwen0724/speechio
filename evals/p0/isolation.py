"""M06 顺序重置与两流交错；访问同一识别器始终串行。"""

from evals.p0.stream_run import run_stream
from evals.p0.trial import Trial


def run_reset_pair(engine, first_audio, second_audio, *, cancel=None):
    """使用同一个自动语言流重置，保留原语言，独立对照采用 M07 自动模式。"""
    stream = engine.create_streaming_session(None)
    first = run_stream(engine, first_audio, None, cancel=cancel, session=stream, close=False)
    if first.status != "ok":
        return [first]
    first_id = stream.stream_id
    try:
        stream.reset()
    except Exception as exc:
        first.fail(exc, "reset", stream.accepted_samples)
        first.close_stream(stream)
        return [first]
    first.cleanup = {**stream.cleanup, "completed_ns": first.now()}
    first.timestamps["close_completed_ns"] = first.cleanup["completed_ns"]
    first.reset_check = {
        "old_stream_id": first_id,
        "new_stream_id": stream.stream_id,
        "accepted_samples": stream.accepted_samples,
        "cache_text": stream.get_result().text,
    }
    second = run_stream(engine, second_audio, None, cancel=cancel, session=stream)
    return [first, second]


def run_interleaved(engine, audios, languages, *, chunk_samples=512, cancel=None):
    """交错新增块并分别结束，归属不依赖完成先后。"""
    arrays = audios
    streams = [engine.create_streaming_session(language) for language in languages]
    trials = [Trial(array.size) for array in arrays]
    for start in range(0, max(array.size for array in arrays), chunk_samples):
        for array, stream, trial in zip(arrays, streams, trials, strict=True):
            if cancel is not None and cancel.is_set():
                trial.cancel(cancel)
                continue
            if start >= array.size or trial.status != "ok":
                continue
            end = min(start + chunk_samples, array.size)
            operation = "accept_audio"
            try:
                trial.call(operation, lambda: stream.accept_audio(array[start:end]))
                trial.confirmed_end_sample = stream.accepted_samples
                if cancel is not None and cancel.is_set():
                    trial.cancel(cancel)
                    continue
                accepted = trial.calls[-1]
                operation = "decode"
                trial.call(operation, stream.decode)
                if cancel is not None and cancel.is_set():
                    trial.cancel(cancel)
                    trial.close_stream(stream)
                    trial.observe(
                        stream.get_result(),
                        "late_result",
                        stream.accepted_samples,
                        stream_id=stream.stream_id,
                    )
                    continue
                operation = "get_result"
                result = trial.call(operation, stream.get_result)
                if cancel is not None and cancel.is_set():
                    trial.cancel(cancel)
                    trial.observe(
                        result, "late_result", stream.accepted_samples, stream_id=stream.stream_id
                    )
                    continue
                trial.observe(
                    result,
                    "stream_partial",
                    stream.accepted_samples,
                    stream_id=stream.stream_id,
                    block_range=[start, end],
                    due_ns=round(end * 1_000_000_000 / 16000),
                    accepted_start_ns=accepted["started_ns"],
                    accepted_end_ns=accepted["completed_ns"],
                )
                if end == array.size:
                    operation = "finish"
                    trial.timestamps["finish_called_ns"] = trial.now()
                    result = trial.call(operation, stream.finish)
                    if cancel is not None and cancel.is_set():
                        trial.cancel(cancel)
                    trial.observe(
                        result,
                        "late_result" if trial.status == "cancelled" else "stream_final",
                        stream.accepted_samples,
                        stream_id=stream.stream_id,
                    )
            except Exception as exc:
                trial.fail(
                    exc,
                    operation,
                    stream.accepted_samples,
                    [start, end] if operation == "accept_audio" else None,
                )
    for stream, trial in zip(streams, trials, strict=True):
        trial.close_stream(stream)
    return trials
