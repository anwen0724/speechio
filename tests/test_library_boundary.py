"""库不依赖仓库内的应用层；用源码导入关系守住目录边界。"""

import ast
import unittest
from pathlib import Path

LIBRARY = Path(__file__).resolve().parents[1] / "src" / "realtime_asr"


class LibraryBoundaryTests(unittest.TestCase):
    def test_library_does_not_import_apps(self):
        violations = []
        for path in LIBRARY.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                names = []
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    names = [node.module]
                for name in names:
                    if name == "apps" or name.startswith("apps."):
                        violations.append(f"{path.relative_to(LIBRARY)}:{node.lineno} {name}")
        self.assertEqual(violations, [])
