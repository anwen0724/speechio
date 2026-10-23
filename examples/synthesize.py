"""使用百炼合成并保存 WAV；仅显式运行脚本时才调用服务。"""

import argparse
import os
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from speechio import RemoteModelConfig, SynthesisOptions, Synthesizer


def save_stream(stream, path):
    """应用逐块保存，不在内存中积累整段音频，也不覆盖已有文件。"""
    with Path(path).open("xb") as file:
        with wave.open(file, "wb") as output:
            output.setnchannels(stream.format.channels)
            output.setsampwidth(2)
            output.setframerate(stream.format.sample_rate)
            for chunk in stream:
                output.writeframesraw(chunk.data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text", nargs="+", help="完整文本或按顺序提供的文本片段")
    parser.add_argument("--voice", required=True, help="服务支持的音色 ID")
    parser.add_argument("--language", default=None)
    parser.add_argument("--mode", choices=("full", "stream", "incremental"), default="full")
    parser.add_argument("--output", default="speech.wav")
    parser.add_argument("--base-url", default=None)
    args = parser.parse_args()
    # 环境变量由示例读取；核心包只使用调用方显式传入的凭据。
    config = RemoteModelConfig(
        provider="bailian",
        model="qwen-audio-3.1-tts-flash",
        api_key=os.environ["DASHSCOPE_API_KEY"],
        base_url=args.base_url,
    )
    options = SynthesisOptions(voice=args.voice, language=args.language)
    with Synthesizer(config) as synthesizer:
        if args.mode == "full":
            audio = synthesizer.synthesize("".join(args.text), options=options)
            audio.save(args.output)
        elif args.mode == "stream":
            with synthesizer.stream("".join(args.text), options=options) as stream:
                save_stream(stream, args.output)
        else:
            # 接收持续进行；发送异常时先关闭流，再退出应用自己的线程池。
            with ThreadPoolExecutor(max_workers=1) as pool:
                with synthesizer.open_stream(options=options) as stream:
                    receiving = pool.submit(save_stream, stream, args.output)
                    for part in args.text:
                        stream.send_text(part)
                    stream.finish_input()
                    receiving.result()


if __name__ == "__main__":
    main()
