"""Qwen3-TTS Base 本地适配器；两个参数规格共用克隆与合成契约。"""

import json

import numpy as np

from speechio.adapters.contracts import TTSAdapter, TTSCapabilities
from speechio.adapters.local.tts import torch_device, validate_local_config, waveform_result
from speechio.errors import TTSError
from speechio.models.files import check_tts_model_files
from speechio.tts.voices import Voice


class QwenTTSAdapter(TTSAdapter):
    """完整合成与独立声音克隆；不宣称上游未交付的流式能力。"""

    capabilities = TTSCapabilities(
        default_sample_rate=24000,
        synthesis=True,
        language_selection=True,
        voice_cloning=True,
        sample_rates=(24000,),
    )

    def __init__(self, config):
        validate_local_config(config)
        self.config = config
        self._model = None
        self._identity = None

    def prepare(self):
        if self._model is not None:
            return
        folder = check_tts_model_files("qwen_tts", self.config.model_path)
        try:
            metadata = json.loads((folder / "config.json").read_text(encoding="utf-8"))
            sizes = {"0b6": "0.6b", "1b7": "1.7b"}
            size = sizes[metadata["tts_model_size"]]
            if (
                metadata["tts_model_type"] != "base"
                or metadata["tokenizer_type"] != "qwen3_tts_tokenizer_12hz"
            ):
                raise ValueError("仅支持 12Hz Base")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise TTSError(
                "MODEL_INCOMPATIBLE", "prepare", "需要 Qwen3-TTS 12Hz Base 0.6B 或 1.7B 模型"
            ) from exc
        try:
            import torch
            from qwen_tts import Qwen3TTSModel
        except ImportError as exc:
            raise TTSError("DEPENDENCY_MISSING", "prepare", "请安装 speechio[tts-qwen]") from exc
        device = torch_device(self.config, torch)
        dtype = self.config.compute_type or (
            "bfloat16" if self.config.device == "gpu" else "float32"
        )
        self._model = Qwen3TTSModel.from_pretrained(
            str(folder),
            device_map=device,
            dtype=getattr(torch, dtype),
            attn_implementation="sdpa",
            local_files_only=True,
        )
        self._identity = f"qwen3-tts-12hz-{size}-base"

    def clone_voice(self, reference_audio, *, reference_text=None):
        """提取声音特征并编码为纯数据；后续合成无需原录音。"""
        self.prepare()
        samples = np.frombuffer(reference_audio.data, dtype="<i2").astype(np.float32) / 32768
        item = self._model.create_voice_clone_prompt(
            ref_audio=(samples, reference_audio.format.sample_rate),
            ref_text=reference_text,
            x_vector_only_mode=reference_text is None,
        )[0]
        value = {
            "embedding": item.ref_spk_embedding.detach().cpu().float().numpy().tolist(),
            "codes": None
            if item.ref_code is None
            else item.ref_code.detach().cpu().numpy().tolist(),
            "reference_text": item.ref_text,
            "embedding_only": bool(item.x_vector_only_mode),
        }
        return Voice(
            adapter="qwen_tts",
            model=self._identity,
            data=json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8"),
        )

    def synthesize(self, text, *, options):
        """只使用保存的声音特征生成目标文字，不重复提取参考声音。"""
        if not isinstance(options.voice, Voice):
            raise TTSError("INVALID_INPUT", "synthesize", "Qwen Base 需要 clone_voice 返回的 Voice")
        self.prepare()
        voice = options.voice
        if voice.adapter != "qwen_tts" or voice.model != self._identity or voice.data_version != 1:
            raise TTSError("VOICE_INCOMPATIBLE", "synthesize", "声音数据与当前 Qwen 模型不兼容")
        import torch
        from qwen_tts.inference.qwen3_tts_model import VoiceClonePromptItem

        try:
            value = json.loads(voice.data)
            prompt = VoiceClonePromptItem(
                ref_code=None
                if value["codes"] is None
                else torch.tensor(value["codes"], dtype=torch.long),
                ref_spk_embedding=torch.tensor(value["embedding"], dtype=torch.float32),
                x_vector_only_mode=value["embedding_only"],
                icl_mode=not value["embedding_only"],
                ref_text=value["reference_text"],
            )
        except (ValueError, KeyError, TypeError, RuntimeError) as exc:
            raise TTSError("VOICE_INCOMPATIBLE", "synthesize", "声音数据内容无效") from exc
        languages = {"zh": "Chinese", "zh-cn": "Chinese", "en": "English", None: "Auto"}
        language = languages.get(options.language, options.language)
        waveforms, rate = self._model.generate_voice_clone(
            text=text, language=language, voice_clone_prompt=[prompt]
        )
        return waveform_result(waveforms[0], rate)

    def close(self):
        """释放本实例持有的模型；Voice 不引用这些资源。"""
        self._model = None
