"""兼容原适配器契约导入。"""

import sys
from importlib import import_module

sys.modules[__name__] = import_module("speechio.adapters.contracts")
