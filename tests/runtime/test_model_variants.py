"""模型准备的公开配置应区分固定块长，不把权重身份混用。"""

import tempfile
import unittest
from pathlib import Path

from realtime_asr import ASRError
from realtime_asr.runtime.models.prepare import prepare_nemotron


class NemotronVariantTests(unittest.TestCase):
    def test_new_variant_still_requires_explicit_preparation_before_offline_use(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "model"
            with self.assertRaises(ASRError) as raised:
                prepare_nemotron(target, variant="80ms")
            self.assertEqual(raised.exception.code, "MODEL_INCOMPLETE")
            self.assertFalse(target.exists())
