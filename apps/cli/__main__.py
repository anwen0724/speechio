"""python -m apps.cli 与仓库内命令行入口使用同一实现。"""

from apps.cli.runner import main

if __name__ == "__main__":
    raise SystemExit(main())
