"""按矩阵顺序执行验证；单次失败保留记录并继续其他独立项。"""

import signal
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from evals.p0.cancellation import CancellationSource
from evals.p0.corpus.manifest import load_corpus
from evals.p0.diagnostics.environment import inspect_environment
from evals.p0.diagnostics.records import RecordWriter
from evals.p0.diagnostics.resources import ResourceSampler
from evals.p0.engines import EnginePool
from evals.p0.execution import execute_trial
from evals.p0.isolation import run_interleaved, run_reset_pair
from evals.p0.matrix import declare_matrix
from evals.p0.storage import BaselineStorage


def run_matrix(manifest, runtime_root, output_root, *, exploratory=False, selected=None):
    """默认要求正式标注；显式探索允许草稿，仍完整列出未执行项。"""
    corpus, audio = load_corpus(manifest, allow_unreviewed=exploratory)
    environment = inspect_environment()
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ-") + uuid4().hex[:8]
    writer = RecordWriter(Path(output_root) / run_id, run_id)
    writer.write("environment", **environment)
    writer.write("corpus", **corpus)
    for sample in corpus["samples"]:
        writer.write("sample", corpus_version=corpus["corpus_version"], **sample)
    storage = BaselineStorage(writer, environment["environment_id"], corpus)
    pool = EnginePool(storage, audio, runtime_root, environment)
    cancel = CancellationSource()
    previous_handler = signal.signal(signal.SIGINT, lambda *_: cancel.set())
    tasks = declare_matrix(corpus)
    from dataclasses import asdict

    writer.write("matrix_declaration", tasks=[asdict(task) for task in tasks])
    try:
        position = 0
        while position < len(tasks):
            task = tasks[position]
            group = [task]
            if task.matrix_item == "M06":
                group = tasks[position : position + 2]
            position += len(group)
            if selected is not None and task.matrix_item not in selected:
                for item in group:
                    storage.save_missing(item, "skipped", "本次命令未选择此矩阵")
                continue
            if cancel.is_set():
                for item in group:
                    storage.save_missing(item, "cancelled", "取消后不开始新工作")
                continue
            print(
                f"{task.matrix_item} {task.backend} {task.sample_id} "
                f"{task.input_path} #{task.trial_index}",
                flush=True,
            )
            engine = pool.get(task)
            if engine is None:
                status, reason = pool.failures[task.backend]
                for item in group:
                    storage.save_missing(item, status, reason)
                continue
            try:
                if task.matrix_item == "M06":
                    with ResourceSampler() as resources:
                        arrays = [audio[item.sample_id] for item in group]
                        if task.scenario == "reset":
                            trials = run_reset_pair(engine, *arrays, cancel=cancel)
                        else:
                            trials = run_interleaved(
                                engine,
                                arrays,
                                [storage.language(item) for item in group],
                                cancel=cancel,
                            )
                    # 两流共享模型，资源使用组原点，不冒充单流独占值。
                    writer.write(
                        "resource_group",
                        matrix_item="M06",
                        scenario=task.scenario,
                        clock_origin_id=uuid4().hex,
                        samples=resources.samples,
                        trial_ids=[item.trial_id for item in trials],
                    )
                    for trial in trials:
                        # 转换到各自试次原点，组内 RSS 仍表示整个进程占用。
                        trial.resource_samples = [
                            {
                                **item,
                                "observed_ns": resources.origin_ns
                                + item["observed_ns"]
                                - trial.origin_ns,
                                "shared_model_group": task.scenario,
                            }
                            for item in resources.samples
                            if 0
                            <= resources.origin_ns + item["observed_ns"] - trial.origin_ns
                            <= trial.timestamps.get("close_completed_ns", trial.now())
                        ]
                else:
                    trials = [
                        execute_trial(
                            task, engine, audio[task.sample_id], storage.language(task), cancel
                        )
                    ]
                for item, trial in zip(group, trials, strict=False):
                    storage.save_trial(item, trial, engine, exploratory=exploratory)
                    print(f"  {trial.status}", flush=True)
                for item in group[len(trials) :]:
                    storage.save_missing(item, "blocked", "前置重置或流创建失败")
            except Exception as exc:
                from realtime_asr.core.errors import error_record

                for item in group:
                    storage.save_missing(item, "failed", error_record(exc, "execute_trial"))
        from evals.p0.boundaries import run_boundaries

        run_boundaries(storage, pool, audio, cancel, selected)
    finally:
        pool.close()
        signal.signal(signal.SIGINT, previous_handler)
    from evals.p0.saved_report import rebuild_report

    report = rebuild_report(writer.folder)
    print(f"阶段结论：{report['status']}；记录：{writer.folder}", flush=True)
    return report
