"""在项目外用隔离解释器检查发布包和可选依赖边界。"""

import importlib.util
import json
import sys

import realtime_asr
from realtime_asr.runtime.asr import ASRRuntime

report = {
    "package": realtime_asr.__file__,
    "isolated": bool(sys.flags.isolated),
    "vad_loaded_on_import": "onnxruntime" in sys.modules or "webrtcvad" in sys.modules,
    "optional_dependencies": {
        name: importlib.util.find_spec(name) is not None
        for name in ("onnxruntime", "webrtcvad", "soundfile", "soxr", "faster_whisper")
    },
    "automatic_api": hasattr(ASRRuntime, "create_auto_session"),
}
assert report["isolated"] and not report["vad_loaded_on_import"] and report["automatic_api"]
assert not report["optional_dependencies"]["soundfile"]
assert not report["optional_dependencies"]["soxr"]
print(json.dumps(report, ensure_ascii=False))
