"""配置声明、模板生成与错误提示的一致性检查。"""

import io
import json
import tempfile
import unittest
from pathlib import Path

from apps.cli.options import COMMANDS, LOCAL, OPTIONS
from apps.cli.parser import _options_for, parse_args
from apps.cli.settings import configure, main, template
from speechio import ASRError


def _template_keys(text):
    """从生成模板中取出 (分组, 键)，忽略说明文字。"""
    section, keys = None, set()
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1]
            continue
        if section is None or "=" not in line:
            continue
        key = line.removeprefix("# ").split("=", 1)[0].strip()
        keys.add((section, key))
    return keys


class ConfigDeclarationTests(unittest.TestCase):
    def test_table_drives_parser_and_template(self):
        expected = {(option.section, option.config_key) for option in OPTIONS}
        self.assertEqual(_template_keys(template()), expected)
        for command in COMMANDS:
            flags = {option.flag for option in _options_for(command.name)}
            for option in OPTIONS:
                if command.name in option.commands:
                    self.assertIn(option.flag, flags, f"{command.name} 缺少 --{option.flag}")

    def test_generated_template_loads_for_every_command(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.toml"
            path.write_text(template(), encoding="utf-8")
            for command in (*LOCAL, "devices"):
                invocation = configure(["--config", str(path)], command)
                if command in LOCAL:
                    self.assertEqual(invocation.sources["model_path"], "config:model.model_path")
                    self.assertIn("--model-path", invocation.argv)
                else:
                    self.assertNotIn("model", invocation.sources)

    def test_all_model_paths_resolve_next_to_config(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.toml"
            path.write_text('[model]\nmodel_path = "tiny"\n', encoding="utf-8")
            invocation = configure(["--config", str(path)], "file")
            self.assertIn(str((path.parent / "tiny").resolve()), invocation.argv)
            path.write_text('[model]\nmodel_path = "models/tiny"\n', encoding="utf-8")
            invocation = configure(["--config", str(path)], "file")
            self.assertIn(str((path.parent / "models/tiny").resolve()), invocation.argv)

    def test_engine_key_and_flag_share_one_name(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.toml"
            path.write_text('[model]\nengine = "nemotron"\n', encoding="utf-8")
            args = parse_args(["stdin", "--model-path", "tiny", "--config", str(path)])
            self.assertEqual(args.engine, "nemotron")

    def test_unknown_config_key_names_the_field(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.toml"
            path.write_text('[model]\nbackend = "whisper"\n', encoding="utf-8")
            with self.assertRaises(ASRError) as raised:
                configure(["--config", str(path)], "mic")
            self.assertEqual(raised.exception.code, "INVALID_CONFIGURATION")
            self.assertIn("model.backend", raised.exception.message)

    def test_config_init_writes_template_that_show_can_read(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.toml"
            created = io.StringIO()
            errors = io.StringIO()
            self.assertEqual(main(["init"], path=path, stdout=created, stderr=errors), 0)
            self.assertEqual(json.loads(created.getvalue())["status"], "created")
            report = io.StringIO()
            self.assertEqual(main(["show", "file"], path=path, stdout=report, stderr=errors), 0)
            data = json.loads(report.getvalue())
            self.assertEqual(data["command"], "file")
            self.assertEqual(data["sources"]["model_path"], "config:model.model_path")
            self.assertEqual(data["values"]["engine"], "whisper")


if __name__ == "__main__":
    unittest.main()
