"""固定语言应进入有效参数，而不只保留运行库的 None 默认值。"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from evals.p0.matrix import MatrixTask
from evals.p0.storage import BaselineStorage


class StorageTests(unittest.TestCase):
    def test_fixed_snapshot_and_final_parameters_have_actual_language(self):
        records = []
        writer = SimpleNamespace(write=lambda kind, **fields: records.append((kind, fields)))
        corpus = {"corpus_version": "v1", "samples": [{"sample_id": "cn", "language": "zh"}]}
        storage = BaselineStorage(writer, "env", corpus)
        engine = SimpleNamespace(
            manifest={"model_record_id": "model", "model_id": "tiny"},
            resolved_parameters=lambda role="final": {
                "language": None,
                "beam_size": 3 if role == "snapshot" else 5,
            },
        )
        task = MatrixTask("M02", "whisper", "cn", "snapshot", "paced")
        with patch("importlib.metadata.version", return_value="1.2.1"):
            config_id = storage.configuration(task, engine)
        parameters = storage.configurations[config_id]["resolved_parameters"]
        self.assertEqual(parameters["snapshot_parameters"]["language"], "zh")
        self.assertEqual(parameters["final_parameters"]["language"], "zh")
