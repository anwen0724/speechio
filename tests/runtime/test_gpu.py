"""GPU 专用库缺失是独立失败，不能退回 CPU 并保留 GPU 名称。"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from realtime_asr.errors import ASRError
from realtime_asr.runtime.native.gpu import GPUContext


class GPUFailureTests(unittest.TestCase):
    def test_empty_native_manifest_is_rejected_before_any_dll_load(self):
        with tempfile.TemporaryDirectory() as folder:
            Path(folder, "bin").mkdir()
            Path(folder, "native-manifest.json").write_text(
                json.dumps({"files": []}), encoding="utf-8"
            )
            with patch("ctypes.WinDLL", side_effect=AssertionError("不应加载 DLL")):
                with self.assertRaises(ASRError) as caught:
                    GPUContext(folder).__enter__()
            self.assertEqual(caught.exception.code, "DEPENDENCY_MISSING")

    def test_missing_project_runtime_is_explicit_and_not_loaded(self):
        with tempfile.TemporaryDirectory() as folder:
            context = GPUContext(folder)
            with self.assertRaises(ASRError) as caught:
                context.__enter__()
            self.assertEqual(caught.exception.code, "DEPENDENCY_MISSING")
