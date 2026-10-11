"""Whisper 批处理适配器；完整消费分段后才返回识别结果。"""

import inspect
from dataclasses import asdict

from realtime_asr.core.audio import as_audio
from realtime_asr.core.errors import ASRError
from realtime_asr.core.results import RecognitionResult
from realtime_asr.runtime.models.manifest import verify_manifest
from realtime_asr.runtime.stages import preparation_step


def consume_output(segments, info, requested_language):
    """保留全部分段及后端语言信息。"""
    try:
        complete = list(segments)
    except Exception as exc:
        raise ASRError("INFERENCE_FAILED", "transcribe", str(exc)) from exc
    language = getattr(info, "language", None)
    return RecognitionResult(
        text=" ".join(item.text for item in complete).strip(),
        requested_language=requested_language,
        reported_language=language,
        language_source="detected" if language else "unavailable",
        language_probability=getattr(info, "language_probability", None),
        segments=[
            {"text": item.text, "start_s": item.start, "end_s": item.end} for item in complete
        ],
    )


class WhisperEngine:
    """复用本地 tiny 模型，快照与最终调用使用独立参数组。"""

    def __init__(self, model_path, *, device="cpu", compute_type="int8", device_index=0):
        self.preparation_steps = []
        with preparation_step(self.preparation_steps, "verification"):
            self.manifest = verify_manifest(model_path)
        try:
            from faster_whisper import BatchedInferencePipeline, WhisperModel
            from faster_whisper.vad import VadOptions
        except ImportError as exc:
            raise ASRError("DEPENDENCY_MISSING", "load_model", str(exc)) from exc
        self.device = device
        self.compute_type = compute_type
        self.device_index = device_index
        try:
            with preparation_step(self.preparation_steps, "load"):
                self._model = WhisperModel(
                    self.manifest["model_path"],
                    device=device,
                    compute_type=compute_type,
                    device_index=device_index,
                    local_files_only=True,
                )
                self._pipeline = BatchedInferencePipeline(self._model)
        except Exception as exc:
            error = ASRError("MODEL_LOAD", "load_model", str(exc))
            error.preparation_steps = self.preparation_steps
            raise error from exc
        signature = inspect.signature(BatchedInferencePipeline.transcribe)
        self._defaults = {
            key: parameter.default
            for key, parameter in signature.parameters.items()
            if key not in ("self", "audio") and parameter.default is not inspect.Parameter.empty
        }
        # 批处理入口的默认 VAD 使用模型块长和 160 ms 静音，不是普通入口的默认配置。
        self._vad = asdict(
            VadOptions(
                max_speech_duration_s=self._model.feature_extractor.chunk_length,
                min_silence_duration_ms=160,
            )
        )

    def resolved_parameters(self, role="final"):
        """展开本次调用参数，不能把默认值只记录成一个字符串。"""
        parameters = dict(self._defaults)
        parameters.update(
            {
                "beam_size": 3 if role == "snapshot" else 5,
                "batch_size": 16,
                "vad_filter": True,
                "vad_parameters": self._vad,
                "initial_prompt": None,
                "suppress_tokens": [-1],
                "word_timestamps": False,
            }
        )
        return {
            **parameters,
            "normalize_audio": False,
            "use_prompt": True,
            "feature_parameters": {
                key: value
                for key, value in vars(self._model.feature_extractor).items()
                if isinstance(value, (int, float, bool, str))
            },
        }

    def transcribe(
        self, audio, language=None, use_prompt=True, word_timestamps=False, *, role="final"
    ):
        """完整消费批处理结果；数组、语言和模型异常不转换为空白成功。"""
        array = as_audio(audio)
        if role not in ("snapshot", "final"):
            raise ValueError("未知的参数角色")
        try:
            segments, info = self._pipeline.transcribe(
                array,
                language=language,
                beam_size=3 if role == "snapshot" else 5,
                batch_size=16,
                vad_filter=True,
                initial_prompt=None,
                suppress_tokens=[-1],
                word_timestamps=word_timestamps,
            )
            return consume_output(segments, info, language)
        except ASRError:
            raise
        except Exception as exc:
            raise ASRError("INFERENCE_FAILED", "transcribe", str(exc)) from exc

    def close(self):
        """移除自有模型引用，不将其声明为显存回收成功。"""
        self._pipeline = None
        self._model = None
