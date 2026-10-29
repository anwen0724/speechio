"""百炼预设音色目录；随发行包更新，运行时不抓取网页。"""

import json
from importlib.resources import files
from urllib.parse import urlsplit

from speechio.errors import TTSError
from speechio.tts.voices import VoiceInfo


def preset_voices(config):
    """按模型及已知官方服务地域筛选，不能确定地域时明确报错。"""
    host = urlsplit(config.base_url or "https://dashscope.aliyuncs.com").hostname
    if host == "dashscope.aliyuncs.com" or host.endswith(".cn-beijing.maas.aliyuncs.com"):
        region = "cn-beijing"
    elif host == "dashscope-intl.aliyuncs.com" or host.endswith(
        ".ap-southeast-1.maas.aliyuncs.com"
    ):
        region = "ap-southeast-1"
    else:
        raise TTSError(
            "CAPABILITY_UNSUPPORTED", "list_voices", "无法确定该服务地址的地域，不能列举预设音色"
        )
    catalog = json.loads(
        files("speechio").joinpath("assets/bailian-tts-voices.json").read_text(encoding="utf-8")
    )
    if config.model != catalog["model"]:
        raise TTSError("CAPABILITY_UNSUPPORTED", "list_voices", "未收录该模型的预设音色")
    return tuple(
        VoiceInfo(
            id=item["id"],
            name=item["name"],
            languages=None if item["languages"] is None else tuple(item["languages"]),
            instruction_control=item["instruction_control"],
            text_timestamps=tuple(item["text_timestamps"]),
        )
        for item in catalog["voices"]
        if region in item["regions"]
    )
