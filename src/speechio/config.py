"""兼容入口；实现位于 speechio.models.config。"""

import sys
from importlib import import_module

sys.modules[__name__] = import_module("speechio.models.config")
