"""兼容入口；实现位于 speechio.stt.text.endpoint。"""

import sys
from importlib import import_module

sys.modules[__name__] = import_module("speechio.stt.text.endpoint")
