"""兼容入口；实现位于 speechio.stt.runtime.tasks。"""

import sys
from importlib import import_module

sys.modules[__name__] = import_module("speechio.stt.runtime.tasks")
