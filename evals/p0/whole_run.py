"""完整音频试次；结果消费与转换属于推理计时区间。"""

from threading import Event

from evals.p0.trial import Trial
from realtime_asr.core.audio import as_audio


def run_whole(engine, audio, language, *, cancel=None, trial=None):
    """只生成一个完整结果；取消后的返回仅保存为迟到诊断。"""
    array = as_audio(audio)
    trial = trial if trial is not None else Trial(array.size)
    cancel = cancel if cancel is not None else Event()
    if cancel.is_set():
        trial.cancel(cancel)
        return trial
    stream = None
    try:
        if hasattr(engine, "create_streaming_session"):
            stream = engine.create_streaming_session(language)

            def complete():
                # 整段包装只提交一次正文，关闭放在推理计时区间之外。
                stream.accept_audio(array)
                trial.confirmed_end_sample = stream.accepted_samples
                return stream.finish()

            result = trial.call("transcribe", complete)
        else:
            result = trial.call("transcribe", lambda: engine.transcribe(array, language=language))
        if cancel.is_set():
            trial.cancel(cancel)
            trial.observe(result, "late_result", array.size)
        else:
            trial.observe(result, "whole_final", array.size)
    except KeyboardInterrupt:
        cancel.set()
        trial.cancel(cancel)
    except Exception as exc:
        confirmed = stream.accepted_samples if stream is not None else 0
        unconfirmed = [0, array.size] if stream is not None and confirmed == 0 else None
        trial.fail(exc, "transcribe", confirmed, unconfirmed)
        if cancel.is_set():
            trial.cancel(cancel)
    if stream is not None:
        trial.close_stream(stream)
        for observation in trial.observations:
            observation["stream_id"] = stream.stream_id
        return trial
    trial.cleanup = {
        "method": "reference_drop",
        "status": "ok",
        "completed_ns": trial.now(),
        "errors": [],
    }
    trial.timestamps["close_completed_ns"] = trial.cleanup["completed_ns"]
    return trial
