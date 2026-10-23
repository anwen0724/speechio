"""兼容入口；实现位于 speechio.adapters.local.engine。"""

import sys
from importlib import import_module

sys.modules[__name__] = import_module("speechio.adapters.local.engine")
