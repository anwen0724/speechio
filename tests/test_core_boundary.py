"""核心层不依赖接入层；用源码导入关系守住目录边界。"""

import ast
import unittest
from pathlib import Path

CORE = Path(__file__).resolve().parents[1] / "src" / "realtime_asr" / "core"
FORBIDDEN = (
    "realtime_asr.cli",
    "realtime_asr.server",
    "realtime_asr.client",
    "realtime_asr.protocol",
)


class CoreBoundaryTests(unittest.TestCase):
    def test_core_does_not_import_entry_layers(self):
        violations = []
        for path in CORE.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                names = []
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    names = [node.module]
                for name in names:
                    if any(name == prefix or name.startswith(prefix + ".") for prefix in FORBIDDEN):
                        violations.append(f"{path.relative_to(CORE)}:{node.lineno} {name}")
        self.assertEqual(violations, [])
