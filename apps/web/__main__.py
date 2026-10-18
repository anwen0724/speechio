"""在仓库中运行：python -m apps.web --enable-debug。"""

import argparse

import uvicorn

from .server.app import create_app
from .server.settings import Settings


def main():
    """只绑定本机地址；模型目录、密钥和预设由本地配置提供。"""
    parser = argparse.ArgumentParser(description="SpeechIO 本地 Web 示例与调试台")
    parser.add_argument("--config", help="本机 TOML 配置文件，默认读取 Web 目录的 config.toml")
    parser.add_argument("--port", type=int, default=8767)
    parser.add_argument("--enable-debug", action="store_true", help="开放 /debug 调试台")
    args = parser.parse_args()
    try:
        settings = Settings.load(args.config, debug=args.enable_debug)
    except ValueError as exc:
        parser.error(str(exc))
    uvicorn.run(create_app(settings), host="127.0.0.1", port=args.port, ws_max_size=1048576)


if __name__ == "__main__":
    main()
