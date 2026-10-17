"""测试共用的外部模型替身；使用真实引擎和真实模型清单。"""

import tempfile
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from realtime_asr import ModelConfig
from realtime_asr.engines.whisper import WhisperEngine
from realtime_asr.runtime.models.manifest import create_manifest


@contextmanager
def whisper_engine():
    """替换 CTranslate2 模型加载，保留清单校验、参数展开和结果转换。"""
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder)
        for name in ("model.bin", "config.json", "tokenizer.json"):
            (path / name).write_bytes(b"test-model")
        create_manifest(path, backend="whisper", model_id="tiny", source="test", revision="abc")
        model = SimpleNamespace(feature_extractor=SimpleNamespace(chunk_length=30))
        with patch("faster_whisper.WhisperModel", return_value=model):
            engine = WhisperEngine(ModelConfig(model=path))
            try:
                yield engine
            finally:
                engine.close()
