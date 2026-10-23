"""按引擎名组装适配器；名字到类的映射集中在这里，调用点不做分支。"""

from importlib import import_module

from speechio.errors import STTError

# 名字只用于选择实现；所有适配器只加载已有本地目录。
ENGINE_CLASSES = {
    "whisper": ("speechio.adapters.local.whisper", "WhisperEngine"),
    "faster_whisper": ("speechio.adapters.local.whisper", "WhisperEngine"),
    "nemotron": ("speechio.adapters.local.nemotron", "NemotronEngine"),
    "sherpa_nemotron": ("speechio.adapters.local.nemotron", "NemotronEngine"),
}

ALIASES = {
    "faster_whisper": "whisper",
    "sherpa_nemotron": "nemotron",
}


def normalize_engine(name):
    """统一大小写和连字符写法，返回公开引擎名。"""
    value = str(name or "whisper").strip().lower().replace("-", "_")
    return ALIASES.get(value, value)


def available_engines():
    """返回去重后的公开引擎名，供错误信息使用。"""
    return sorted(set(ENGINE_CLASSES) - set(ALIASES))


def create_engine(config):
    """按配置创建本地适配器；未知名字在组装边界立即失败。"""
    name = str(config.adapter or "whisper").strip().lower().replace("-", "_")
    if name not in ENGINE_CLASSES:
        raise STTError(
            "BACKEND_INCOMPATIBLE",
            "configure",
            f"不支持的引擎 {name}；可用：{', '.join(available_engines())}",
        )
    module_name, class_name = ENGINE_CLASSES[name]
    return getattr(import_module(module_name), class_name)(config)


# 每个已支持模型明确声明能力；不按名称后缀猜测未知模型的协议。
LOCAL_CAPABILITIES = {
    "whisper": {"snapshot": True, "word_timestamps": True},
    "nemotron": {"streaming": True},
}
REMOTE_MODELS = {
    ("bailian", "qwen-audio-3.1-asr-flash-filetrans"): {
        "transcription": False,
        "file_transcription": True,
        "url_transcription": True,
        "word_timestamps": True,
    },
    ("bailian", "qwen-audio-3.1-asr-flash"): {"word_timestamps": True},
    ("bailian", "qwen-audio-3.1-asr-flash-streaming"): {
        "streaming": True,
        "word_timestamps": True,
    },
}


def engine_capabilities(config):
    """集中声明已适配模型的能力，不加载运行库或联网探测。"""
    from speechio.adapters.contracts import STTCapabilities
    from speechio.models.config import LocalModelConfig, RemoteModelConfig

    if isinstance(config, LocalModelConfig):
        name = normalize_engine(config.adapter)
        if name not in LOCAL_CAPABILITIES:
            raise STTError("BACKEND_INCOMPATIBLE", "configure", "未知本地引擎")
        return STTCapabilities(
            engine=name,
            model=str(config.model_path),
            device=config.device,
            **LOCAL_CAPABILITIES[name],
        )
    if isinstance(config, RemoteModelConfig):
        key = (config.provider, config.model)
        if key not in REMOTE_MODELS:
            raise STTError("BACKEND_INCOMPATIBLE", "configure", "尚未适配该远程服务或模型")
        return STTCapabilities(engine=config.provider, model=config.model, **REMOTE_MODELS[key])
    raise STTError("INVALID_CONFIGURATION", "configure", "需要本地或远程模型配置")


def build_engine(config):
    """根据来源创建统一引擎；专有运行库在 prepare 时才导入。"""
    from speechio.models.config import LocalModelConfig, RemoteModelConfig

    if isinstance(config, LocalModelConfig):
        from speechio.adapters.local.engine import LocalEngine

        return LocalEngine(config)
    engine_capabilities(config)
    if isinstance(config, RemoteModelConfig):
        if (config.provider, config.model) == ("bailian", "qwen-audio-3.1-asr-flash-filetrans"):
            from speechio.adapters.remote.bailian_file import BailianFileEngine

            return BailianFileEngine(config)
        from speechio.adapters.remote.bailian_stt import BailianEngine

        return BailianEngine(config)


def build_synthesis_engine(config):
    """按合成操作选择适配器，不使用原识别能力查询猜测功能。"""
    from speechio.errors import TTSError
    from speechio.models.config import LocalModelConfig, RemoteModelConfig

    if isinstance(config, LocalModelConfig):
        adapters = {
            "qwen_tts": ("speechio.adapters.local.qwen_tts", "QwenTTSAdapter"),
            "kokoro": ("speechio.adapters.local.kokoro", "KokoroAdapter"),
        }
        selected = adapters.get(config.adapter)
        if selected is None:
            raise TTSError("CAPABILITY_UNSUPPORTED", "configure", "未知本地合成适配器")
        module, name = selected
        return getattr(import_module(module), name)(config)
    if not isinstance(config, RemoteModelConfig) or (config.provider, config.model) != (
        "bailian",
        "qwen-audio-3.1-tts-flash",
    ):
        raise TTSError("CAPABILITY_UNSUPPORTED", "configure", "尚未适配该来源的合成模型")
    from speechio.adapters.remote.bailian_tts import BailianSynthesisEngine

    return BailianSynthesisEngine(config)
