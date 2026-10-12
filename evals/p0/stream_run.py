"""原生流式试次；接受、解码、读取和结束分别计时。"""

from threading import Event

from evals.p0.trial import Trial


def run_stream(
    engine,
    audio,
    language,
    *,
    mode="fast",
    chunk_samples=512,
    cancel=None,
    session=None,
    close=True,
    trial=None,
):
    """按矩阵声明回放已校验的语料数组，保留不足一块的尾部。"""
    array = audio
    cancel = cancel if cancel is not None else Event()
    stream = session if session is not None else engine.create_streaming_session(language)
    trial = trial if trial is not None else Trial(array.size)
    operation, block_range = "accept_audio", None
    try:
        for start in range(0, array.size, chunk_samples):
            end = min(start + chunk_samples, array.size)
            due = round(end * 1_000_000_000 / 16000)
            if mode == "paced":
                cancel.wait(max(0, (due - trial.now()) / 1_000_000_000))
            if cancel.is_set():
                trial.cancel(cancel)
                break
            block_range = [start, end]
            operation = "accept_audio"
            trial.call(operation, lambda: stream.accept_audio(array[start:end]))
            trial.confirmed_end_sample = stream.accepted_samples
            if cancel.is_set():
                trial.cancel(cancel)
                break
            accepted_call = trial.calls[-1]
            operation = "decode"
            trial.call(operation, stream.decode)
            if cancel.is_set():
                trial.cancel(cancel)
                trial.close_stream(stream)
                trial.observe(
                    stream.get_result(),
                    "late_result",
                    stream.accepted_samples,
                    stream_id=stream.stream_id,
                )
                break
            operation = "get_result"
            result = trial.call(operation, stream.get_result)
            kind = "late_result" if cancel.is_set() else "stream_partial"
            trial.observe(
                result,
                kind,
                stream.accepted_samples,
                stream_id=stream.stream_id,
                block_range=block_range,
                due_ns=due,
                accepted_start_ns=accepted_call["started_ns"],
                accepted_end_ns=accepted_call["completed_ns"],
            )
            if cancel.is_set():
                trial.cancel(cancel)
                break
        if trial.status == "ok":
            trial.timestamps["finish_called_ns"] = trial.now()
            operation = "finish"
            result = trial.call(operation, stream.finish)
            if cancel.is_set():
                trial.cancel(cancel)
                trial.observe(
                    result, "late_result", stream.accepted_samples, stream_id=stream.stream_id
                )
            else:
                trial.observe(
                    result, "stream_final", stream.accepted_samples, stream_id=stream.stream_id
                )
    except KeyboardInterrupt:
        cancel.set()
        trial.cancel(cancel)
    except Exception as exc:
        unconfirmed = block_range if operation == "accept_audio" else None
        trial.fail(exc, operation, stream.accepted_samples, unconfirmed)
        if cancel.is_set():
            trial.cancel(cancel)
    finally:
        if close or trial.status != "ok":
            trial.close_stream(stream)
    return trial
