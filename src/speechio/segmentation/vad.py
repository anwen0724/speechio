"""兼容入口；实现位于 speechio.stt.segmentation.vad。"""

import sys
from importlib import import_module

sys.modules[__name__] = import_module("speechio.stt.segmentation.vad")
