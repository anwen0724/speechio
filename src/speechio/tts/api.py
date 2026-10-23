"""一次性合成入口，负责关闭自己创建的合成对象。"""

from speechio.tts.synthesizer import Synthesizer


def synthesize(text, *, config, options=None, reference_audio=None, reference_text=None):
    """完整文本合成，可传参考音频先克隆；重复调用宜复用 Synthesizer。"""
    with Synthesizer(config) as synthesizer:
        return synthesizer.synthesize(
            text, options=options, reference_audio=reference_audio, reference_text=reference_text
        )


def clone_voice(reference_audio, *, config, reference_text=None):
    """一次性使用指定模型克隆声音，结果独立于模型实例。"""
    with Synthesizer(config) as synthesizer:
        return synthesizer.clone_voice(reference_audio, reference_text=reference_text)
