"""按引擎名组装适配器；名字到类的映射集中在这里，调用点不做分支。"""

from importlib import import_module

from realtime_asr.core.errors import ASRError

# 名字只用于选择实现；模型如何获取由各适配器决定。
ENGINE_CLASSES = {
    "whisper": ("realtime_asr.core.engines.whisper", "WhisperEngine"),
    "faster_whisper": ("realtime_asr.core.engines.whisper", "WhisperEngine"),
    "nemotron": ("realtime_asr.core.engines.nemotron", "NemotronEngine"),
    "sherpa_nemotron": ("realtime_asr.core.engines.nemotron", "NemotronEngine"),
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
    name = str(config.engine or "whisper").strip().lower().replace("-", "_")
    if name not in ENGINE_CLASSES:
        raise ASRError(
            "BACKEND_INCOMPATIBLE",
            "configure",
            f"不支持的引擎 {name}；可用：{', '.join(available_engines())}",
        )
    module_name, class_name = ENGINE_CLASSES[name]
    return getattr(import_module(module_name), class_name)(config)
