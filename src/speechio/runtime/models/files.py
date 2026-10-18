"""本地 ASR 文件要求；加载只检查运行必需文件，不下载或读写哈希记录。"""

from pathlib import Path

from speechio.errors import ASRError

MODEL_FILES = {
    "whisper": ("model.bin", "config.json", "tokenizer.json"),
    "nemotron": ("encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt"),
}


def check_model_files(engine, folder):
    """检查指定目录中的必需文件；格式是否可执行由对应运行库确认。"""
    folder = Path(folder).expanduser().resolve()
    if not folder.is_dir():
        raise ASRError("MODEL_INCOMPLETE", "load_model", f"模型目录不存在：{folder}")
    for name in MODEL_FILES[engine]:
        path = folder / name
        if not path.is_file() or path.stat().st_size == 0:
            raise ASRError("MODEL_INCOMPLETE", "load_model", f"模型文件缺失或为空：{path}")
    return folder
