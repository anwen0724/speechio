"""python -m realtime_asr 与安装命令使用同一入口。"""

from realtime_asr.cli.runner import main

if __name__ == "__main__":
    raise SystemExit(main())
