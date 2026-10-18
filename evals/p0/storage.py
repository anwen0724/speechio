"""将引擎结果补全为 P0 记录，配置身份由实际参数确定。"""

import hashlib
import importlib.metadata
import json
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

from evals.p0.evaluation.metrics import compute_metrics


class BaselineStorage:
    """配置复用按内容判断，试次与观察分别保存，保留取消及失败。"""

    def __init__(self, writer, environment_id, corpus):
        self.writer = writer
        self.environment_id = environment_id
        self.corpus = corpus
        self.samples = {sample["sample_id"]: sample for sample in corpus["samples"]}
        self.configurations = {}
        self.trials = []

    def language(self, task):
        """非语音沿用中文提示，自动配置独立记录。"""
        if task.auto_language:
            return None if task.backend.startswith("whisper") else "auto"
        language = self.samples[task.sample_id]["language"]
        if task.backend.startswith("whisper"):
            return "en" if language == "en" else "zh"
        return "en-US" if language == "en" else "zh-CN"

    def configuration(self, task, engine):
        """快照与最终参数组同时进入配置，不让不同 beam size 共用摘要。"""
        # 评测自行登记本次模型身份，不要求加载目录提供说明文件或哈希。
        engine.manifest.setdefault("model_record_id", uuid4().hex)
        engine.manifest.setdefault("model_id", Path(engine.manifest["model_path"]).name)
        if task.input_path == "snapshot":
            parameters = {
                "snapshot_parameters": engine.resolved_parameters("snapshot"),
                "final_parameters": engine.resolved_parameters("final"),
            }
        else:
            parameters = engine.resolved_parameters()
        if task.backend.startswith("whisper"):
            groups = parameters.values() if task.input_path == "snapshot" else [parameters]
            for group in groups:
                group["language"] = self.language(task)
        else:
            parameters["stream_options"] = {"language": self.language(task)}
        version = importlib.metadata.version(
            "faster-whisper" if task.backend.startswith("whisper") else "sherpa-onnx"
        )
        configuration = {
            "model_record_id": engine.manifest["model_record_id"],
            "model_id": engine.manifest["model_id"],
            "backend": task.backend,
            "backend_version": version,
            "device": "gpu" if task.backend == "whisper-gpu" else "cpu",
            "compute_type": "float16" if task.backend == "whisper-gpu" else "int8",
            "device_index": 0 if task.backend == "whisper-gpu" else None,
            "language_mode": "auto" if task.auto_language else "fixed",
            "requested_language": self.language(task),
            "input_path": task.input_path,
            "input_mode": task.input_mode,
            "chunk_samples": task.chunk_samples,
            "snapshot_interval_samples": 3200 if task.input_path == "snapshot" else None,
            "resolved_parameters": parameters,
            "environment_id": self.environment_id,
            "corpus_version": self.corpus["corpus_version"],
            "normalization_id": "p0-text-v1",
        }
        digest = hashlib.sha256(
            json.dumps(configuration, sort_keys=True, allow_nan=False).encode()
        ).hexdigest()[:20]
        config_id = f"{task.backend}-{digest}"
        if config_id not in self.configurations:
            self.configurations[config_id] = configuration
            self.writer.write("configuration", config_id=config_id, **configuration)
        return config_id

    def save_trial(self, task, trial, engine, *, exploratory=False, role=None):
        """草稿探索不进入正式主评分；原定角色独立保留供核对。"""
        config_id = self.configuration(task, engine)
        role = role or ("diagnostic" if exploratory else task.evaluation_role)
        context = {
            "config_id": config_id,
            "sample_id": task.sample_id,
            "trial_id": trial.trial_id,
            "matrix_item": task.matrix_item,
            "trial_index": None if role == "warmup" else task.trial_index,
            "evaluation_role": role,
            "planned_evaluation_role": task.evaluation_role,
            "input_path": task.input_path,
            "input_mode": task.input_mode,
            "scenario": task.scenario,
            "language_mode": "auto" if task.auto_language else "fixed",
            "requested_language": self.language(task),
            "corpus_version": self.corpus["corpus_version"],
        }
        times, metrics, reasons, flags = compute_metrics(
            trial,
            self.samples[task.sample_id],
            mode=task.input_mode,
            path=task.input_path,
        )
        for observation in trial.observations:
            self.writer.write(
                "observation",
                **context,
                **{key: value for key, value in observation.items() if key != "record_type"},
            )
        finals = [
            item
            for item in trial.observations
            if item["output_kind"] in ("whole_final", "snapshot_final", "stream_final")
        ]
        result = finals[0]["result"] if len(finals) == 1 and trial.status == "ok" else None
        record = self.writer.write(
            "trial_result",
            **context,
            status=trial.status,
            output_kind=finals[0]["output_kind"] if result is not None else None,
            result=result,
            accepted_range=[0, trial.sample_count]
            if result is not None
            else (trial.error or {}).get("confirmed_range", [0, trial.confirmed_end_sample]),
            decoded_end_sample=None,
            stream_id=finals[0].get("stream_id") if finals else None,
            snapshot_id=finals[0].get("snapshot_id") if finals else None,
            error=trial.error,
            timestamps={**trial.timestamps, **times},
            metrics=metrics,
            null_reasons=reasons,
            flags=flags,
            calls=trial.calls,
            cleanup=trial.cleanup,
            reset_check=getattr(trial, "reset_check", None),
            exploratory=exploratory,
            annotation_based_metrics_provisional=not self.samples[task.sample_id][
                "annotation_checked"
            ],
        )
        self.trials.append(record)
        for resource in getattr(trial, "resource_samples", []):
            self.writer.write(
                "resource_sample",
                **context,
                phase="warmup" if role == "warmup" else "inference",
                clock_origin_id=trial.trial_id,
                **resource,
            )
        return record

    def save_missing(self, task, status, reason):
        """未执行项保留声明身份与原因，不生成模型返回或假耗时。"""
        declared = {
            **asdict(task),
            "requested_language": self.language(task),
            "environment_id": self.environment_id,
            "state": "unprepared",
            "corpus_version": self.corpus["corpus_version"],
        }
        digest = hashlib.sha256(json.dumps(declared, sort_keys=True).encode()).hexdigest()[:20]
        config_id = f"unprepared-{task.backend}-{digest}"
        if config_id not in self.configurations:
            self.configurations[config_id] = declared
            self.writer.write("configuration", config_id=config_id, **declared)
        record = self.writer.write(
            "trial_result",
            **asdict(task),
            config_id=config_id,
            trial_id=uuid4().hex,
            status=status,
            reason=reason,
            result=None,
            output_kind=None,
            language_mode="auto" if task.auto_language else "fixed",
            corpus_version=self.corpus["corpus_version"],
        )
        self.trials.append(record)
        return record
