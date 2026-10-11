"""正式语料不能以未经听音核实的标注启动评分。"""

import json
import tempfile
import unittest
import wave
from pathlib import Path

from evals.p0.corpus.manifest import load_corpus
from realtime_asr.core.errors import ASRError
from realtime_asr.runtime.models.manifest import sha256_file


class CorpusTests(unittest.TestCase):
    def test_unreviewed_annotation_and_changed_audio_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            with wave.open(str(path / "sample.wav"), "wb") as wav:
                wav.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                wav.writeframes(bytes(32000))
            item = {
                "sample_id": "one",
                "audio_path": "sample.wav",
                "audio_sha256": sha256_file(path / "sample.wav"),
                "sample_count": 16000,
                "sample_rate": 16000,
                "channels": 1,
                "sample_width_bytes": 2,
                "language": "none",
                "categories": ["C08"],
                "reference_text": "",
                "speech_intervals": [],
                "source": {"source_id": "silence"},
                "annotation_checked": False,
            }
            manifest = path / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {"corpus_version": "draft", "samples": [item], "representative_sample_id": {}}
                )
            )
            with self.assertRaises(ASRError) as caught:
                load_corpus(manifest)
            self.assertEqual(caught.exception.code, "RECORD_INCOMPLETE")
            self.assertEqual(caught.exception.operation, "validate_annotation")
            item["annotation_checked"] = True
            item["audio_sha256"] = "0" * 64
            manifest.write_text(
                json.dumps(
                    {"corpus_version": "draft", "samples": [item], "representative_sample_id": {}}
                )
            )
            with self.assertRaises(ASRError) as caught:
                load_corpus(manifest)
            self.assertEqual(caught.exception.code, "RECORD_INCOMPLETE")
            self.assertEqual(caught.exception.operation, "validate_audio_hash")
