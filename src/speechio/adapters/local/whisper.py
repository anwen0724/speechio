"""Whisper 批处理适配器；完整消费分段后才返回识别结果。"""

import inspect
from dataclasses import asdict
from math import isfinite
from pathlib import Path

from speechio.errors import STTError
from speechio.models.files import check_model_files
from speechio.models.stages import preparation_step
from speechio.stt.input import as_audio
from speechio.stt.results import Transcript, WordTimestamp


class WhisperEngine:
    """只加载本地目录，快照与最终调用使用独立参数组。"""

    def __init__(self, config):
        self.preparation_steps = []
        model_path = Path(config.model_path)
        with preparation_step(self.preparation_steps, "verification"):
            source = str(check_model_files("whisper", model_path))
            self.manifest = {"backend": "whisper", "model_path": source}
        try:
            from faster_whisper import BatchedInferencePipeline, WhisperModel
            from faster_whisper.vad import VadOptions
        except ImportError as exc:
            raise STTError(
                "DEPENDENCY_MISSING",
                "load_model",
                '缺少 faster-whisper；运行 pip install "speechio[stt-whisper]" 后重试',
            ) from exc
        self.device = config.device
        self.compute_type = config.compute_type or "default"
        self.device_index = config.device_index
        options = config.adapter_options or {}
        beam_size = options.get("beam_size")
        if beam_size is not None and (type(beam_size) is not int or beam_size <= 0):
            raise STTError(
                "INVALID_CONFIGURATION", "configure_engine", "engine_options.beam_size 应为正整数"
            )
        self.beam_size = beam_size
        try:
            with preparation_step(self.preparation_steps, "load"):
                self._model = WhisperModel(
                    source,
                    device="cuda" if config.device == "gpu" else "cpu",
                    compute_type=self.compute_type,
                    device_index=config.device_index,
                    local_files_only=True,
                )
                self._pipeline = BatchedInferencePipeline(self._model)
        except Exception as exc:
            error = STTError("MODEL_LOAD", "load_model", str(exc))
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
                beam_size=self.beam_size or (3 if role == "snapshot" else 5),
                batch_size=16,
                vad_filter=True,
                initial_prompt=None,
                suppress_tokens=[-1],
                word_timestamps=word_timestamps,
            )
            # 分段是惰性迭代器，必须在本次调用内全部消费并转换。
            complete = list(segments)
            words = None
            timestamp_status = "not_requested"
            if word_timestamps:
                reported = [word for segment in complete for word in (segment.words or [])]
                valid = reported and all(
                    isfinite(word.start)
                    and isfinite(word.end)
                    and 0 <= word.start <= word.end
                    and round(word.end * 16000) <= len(array)
                    for word in reported
                )
                timestamp_status = "reported" if valid else "unavailable"
                if valid:
                    words = tuple(
                        WordTimestamp(
                            word.word,
                            round(word.start * 16000),
                            round(word.end * 16000),
                            word.probability,
                        )
                        for word in reported
                    )
            return Transcript(
                text=" ".join(item.text for item in complete).strip(),
                requested_language=language,
                reported_language=info.language,
                language_source="detected" if info.language else "unavailable",
                language_probability=info.language_probability,
                segments=[
                    {"text": item.text, "start_s": item.start, "end_s": item.end}
                    for item in complete
                ],
                words=words,
                timestamp_status=timestamp_status,
            )
        except Exception as exc:
            raise STTError("INFERENCE_FAILED", "transcribe", str(exc)) from exc

    def close(self):
        """移除自有模型引用，不将其声明为显存回收成功。"""
        self._pipeline = None
        self._model = None
