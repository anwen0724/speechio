"""按声明执行单个真实试次，统一接入资源观测。"""

from evals.p0.diagnostics.resources import ResourceSampler
from evals.p0.snapshot_run import run_snapshots
from evals.p0.stream_run import run_stream
from evals.p0.trial import Trial
from evals.p0.whole_run import run_whole


def execute_trial(task, engine, audio, language, cancel=None):
    """标准音频与模型已就绪后建立试次原点，不计文件读取和预热。"""
    trial = Trial(audio.size)
    with ResourceSampler(gpu=task.backend == "whisper-gpu", origin_ns=trial.origin_ns) as resources:
        if task.input_path == "whole":
            trial = run_whole(engine, audio, language, cancel=cancel, trial=trial)
        elif task.input_path == "snapshot":
            trial = run_snapshots(engine, audio, language, cancel=cancel, trial=trial)
        else:
            trial = run_stream(
                engine,
                audio,
                language,
                mode=task.input_mode,
                chunk_samples=task.chunk_samples,
                cancel=cancel,
                trial=trial,
            )
    trial.resource_samples = resources.samples
    return trial
