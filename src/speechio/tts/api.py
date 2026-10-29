"""一次性合成入口，负责关闭自己创建的合成对象。"""

from speechio._async import cancellable, run_operation
from speechio.errors import TTSError
from speechio.tts.synthesizer import Synthesizer


@cancellable(TTSError)
def synthesize(
    text, *, config, options=None, reference_audio=None, reference_text=None, cancellation=None
):
    """完整文本合成，可传参考音频先克隆；重复调用宜复用 Synthesizer。"""
    with Synthesizer(config) as synthesizer:
        return synthesizer.synthesize(
            text, options=options, reference_audio=reference_audio, reference_text=reference_text
        )


@cancellable(TTSError)
def clone_voice(reference_audio, *, config, reference_text=None, cancellation=None):
    """一次性使用指定模型克隆声音，结果独立于模型实例。"""
    with Synthesizer(config) as synthesizer:
        return synthesizer.clone_voice(reference_audio, reference_text=reference_text)


async def synthesize_async(
    text, *, config, options=None, reference_audio=None, reference_text=None, cancellation=None
):
    """一次性异步合成；即使等待者取消，内部拥有者仍负责关闭模型。"""
    return await run_operation(
        synthesize,
        text,
        config=config,
        options=options,
        reference_audio=reference_audio,
        reference_text=reference_text,
        cancellation=cancellation,
    )


async def clone_voice_async(reference_audio, *, config, reference_text=None, cancellation=None):
    """异步克隆并关闭自有模型，返回可保存的 Voice。"""
    return await run_operation(
        clone_voice,
        reference_audio,
        config=config,
        reference_text=reference_text,
        cancellation=cancellation,
    )
