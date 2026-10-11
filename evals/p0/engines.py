"""验证进程的模型拥有者；预热与重新加载均使用真实后端。"""

import importlib.metadata
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

from evals.p0.diagnostics.native_libraries import loaded_libraries
from evals.p0.diagnostics.resources import ResourceSampler
from evals.p0.execution import execute_trial
from realtime_asr.core.errors import ASRError, error_record
from realtime_asr.runtime.gpu import CUDNN_VERSION, GPUContext


class EnginePool:
    """每种配置复用一份模型；失败不静默换设备或后端。"""

    def __init__(self, storage, audio, runtime_root, environment):
        self.storage = storage
        self.audio = audio
        self.root = Path(runtime_root)
        self.environment = environment
        self.engines = {}
        self.failures = {}
        self.gpu_context = None

    def _create(self, backend):
        """只在显式加载阶段导入后端；设备由声明决定。"""
        if backend == "nemotron":
            from realtime_asr.core.engines.nemotron import NemotronEngine

            return NemotronEngine(self.root / "models/nemotron")
        from realtime_asr.core.engines.whisper import WhisperEngine

        if backend == "whisper-gpu":
            self.gpu_context = GPUContext(self.root / "cudnn" / CUDNN_VERSION)
            self.gpu_context.__enter__()
        return WhisperEngine(
            self.root / "models/whisper-tiny",
            device="cuda" if backend == "whisper-gpu" else "cpu",
            compute_type="float16" if backend == "whisper-gpu" else "int8",
        )

    def _save_resources(self, sampler, backend, phase, config_id=None):
        """模型阶段没有具体样本，使用独立原点及明确空归属。"""
        origin_id = uuid4().hex
        for item in sampler.samples:
            self.storage.writer.write(
                "resource_sample",
                backend=backend,
                phase=phase,
                config_id=config_id,
                sample_id=None,
                trial_id=None,
                clock_origin_id=origin_id,
                **item,
            )

    def get(self, task):
        """首次加载前校验，加载后独立预热，后续正式试次复用该模型。"""
        backend = task.backend
        if backend in self.engines:
            return self.engines[backend]
        if backend in self.failures:
            return None
        if backend == "whisper-gpu" and not self.environment["gpus"]:
            unknown = any(item["name"] == "gpus" for item in self.environment["unknown_fields"])
            self.failures[backend] = (
                "blocked" if unknown else "not_applicable",
                "无法读取 GPU 条件" if unknown else "没有 NVIDIA GPU",
            )
            return None
        try:
            with ResourceSampler(gpu=backend == "whisper-gpu") as resources:
                engine = self._create(backend)
        except Exception as exc:
            status = (
                "blocked"
                if isinstance(exc, ASRError)
                and exc.code in ("DEPENDENCY_MISSING", "MODEL_INCOMPLETE")
                else "failed"
            )
            self.failures[backend] = (status, error_record(exc, "load_model"))
            for step in getattr(exc, "preparation_steps", []):
                self.storage.writer.write("preparation_step", backend=backend, **step)
            if backend == "whisper-gpu" and self.gpu_context is not None:
                self.gpu_context.__exit__(None, None, None)
                self.gpu_context = None
            return None
        self.engines[backend] = engine
        self._save_resources(resources, backend, "load", self.storage.configuration(task, engine))
        version = importlib.metadata.version(
            "sherpa-onnx" if backend == "nemotron" else "faster-whisper"
        )
        self.storage.writer.write("model", **{**engine.manifest, "backend_version": version})
        for step in engine.preparation_steps:
            self.storage.writer.write("preparation_step", backend=backend, **step)
        warm_task = replace(
            task,
            sample_id=self.storage.corpus["representative_sample_id"]["C01"],
            input_path="whole" if backend.startswith("whisper") else "stream",
            input_mode="offline" if backend.startswith("whisper") else "fast",
            chunk_samples=None if backend.startswith("whisper") else 512,
            evaluation_role="warmup",
            auto_language=False,
            scenario=None,
        )
        warm = execute_trial(
            warm_task, engine, self.audio[warm_task.sample_id], self.storage.language(warm_task)
        )
        self.storage.save_trial(warm_task, warm, engine, role="warmup")
        if warm.status != "ok":
            self.failures[backend] = ("failed", "真实模型预热失败")
            engine.close()
            self.engines.pop(backend)
            return None
        self.storage.writer.write(
            "native_runtime",
            backend=backend,
            loaded_library_paths=loaded_libraries(),
            note="实际已加载路径；同名库可能有多份，不从目录存在推导实际绑定版本。",
        )
        return engine

    def reload(self, task, *, exploratory, cancel):
        """关闭自有引用后重新加载，同一代表样本取得独立结果。"""
        engine = self.engines.pop(task.backend, None)
        if engine is None or cancel.is_set():
            return None
        with ResourceSampler() as resources:
            engine.close()
        self._save_resources(resources, task.backend, "close")
        engine = self.get(task)
        if engine is None:
            return None
        reload_task = replace(task, evaluation_role="diagnostic", scenario="reload", trial_index=3)
        trial = execute_trial(
            reload_task, engine, self.audio[task.sample_id], self.storage.language(task), cancel
        )
        return self.storage.save_trial(reload_task, trial, engine, exploratory=exploratory)

    def close(self):
        """移除所有自有模型引用，GPU 目录句柄最后关闭。"""
        for backend, engine in self.engines.items():
            with ResourceSampler(gpu=backend == "whisper-gpu") as resources:
                engine.close()
            self._save_resources(resources, backend, "close")
        self.engines.clear()
        if self.gpu_context is not None:
            self.gpu_context.__exit__(None, None, None)
            self.gpu_context = None
