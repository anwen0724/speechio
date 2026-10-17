"""加载调用方的引擎工厂，通过公开同步 SDK 识别文件。"""

import argparse
from importlib import import_module

from speechio import Transcriber


def main():
    """工厂负责模型、地址与凭据；SDK 接管返回实例的生命周期。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("factory", help="调用方模块与无参数工厂，例如 company_asr:create_engine")
    parser.add_argument("file", help="待识别音频文件；非标准 WAV 需安装 audio 依赖")
    parser.add_argument("--language")
    args = parser.parse_args()
    module_name, function_name = args.factory.split(":", 1)
    factory = getattr(import_module(module_name), function_name)
    with Transcriber(engine=factory()) as recognizer:
        result = recognizer.transcribe_file(args.file, language=args.language)
        print(result.text)


if __name__ == "__main__":
    main()
