"""固定语言应进入有效参数，而不只保留运行库的 None 默认值。"""

import json
import tempfile
import unittest
from pathlib import Path

from evals.p0.diagnostics.records import RecordWriter
from evals.p0.matrix import MatrixTask
from evals.p0.storage import BaselineStorage
from tests.support import whisper_engine


class StorageTests(unittest.TestCase):
    def test_fixed_snapshot_and_final_parameters_have_actual_language(self):
        corpus = {"corpus_version": "v1", "samples": [{"sample_id": "cn", "language": "zh"}]}
        task = MatrixTask("M02", "whisper", "cn", "snapshot", "paced")
        with tempfile.TemporaryDirectory() as folder, whisper_engine() as engine:
            writer = RecordWriter(Path(folder) / "run", "test-run")
            storage = BaselineStorage(writer, "env", corpus)
            config_id = storage.configuration(task, engine)
            record = json.loads(writer.path.read_text(encoding="utf-8"))
        self.assertEqual(record["record_type"], "configuration")
        self.assertEqual(record["config_id"], config_id)
        parameters = record["resolved_parameters"]
        self.assertEqual(parameters["snapshot_parameters"]["language"], "zh")
        self.assertEqual(parameters["final_parameters"]["language"], "zh")
        self.assertEqual(parameters["snapshot_parameters"]["beam_size"], 3)
        self.assertEqual(parameters["final_parameters"]["beam_size"], 5)
