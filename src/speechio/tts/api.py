"""一次性合成入口，负责关闭自己创建的合成对象。"""

from speechio.tts.synthesizer import Synthesizer


def synthesize(text, *, options, config=None, engine=None):
    """完整文本合成；重复调用宜显式复用 Synthesizer。"""
    with Synthesizer(config, engine=engine) as synthesizer:
        return synthesizer.synthesize(text, options=options)
