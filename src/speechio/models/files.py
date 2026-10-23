"""本地 ASR 文件要求；加载只检查运行必需文件，不下载或读写哈希记录。"""

from pathlib import Path

from speechio.errors import STTError

MODEL_FILES = {
    "whisper": ("model.bin", "config.json", "tokenizer.json"),
    "nemotron": ("encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt"),
}


def check_model_files(engine, folder):
    """检查指定目录中的必需文件；格式是否可执行由对应运行库确认。"""
    folder = Path(folder).expanduser().resolve()
    if not folder.is_dir():
        raise STTError("MODEL_INCOMPLETE", "load_model", f"模型目录不存在：{folder}")
    for name in MODEL_FILES[engine]:
        path = folder / name
        if not path.is_file() or path.stat().st_size == 0:
            raise STTError("MODEL_INCOMPLETE", "load_model", f"模型文件缺失或为空：{path}")
    return folder


TTS_REPOSITORIES = {
    "qwen_tts": ("Qwen/Qwen3-TTS-12Hz-0.6B-Base", "Qwen/Qwen3-TTS-12Hz-1.7B-Base"),
    "kokoro": ("hexgrad/Kokoro-82M-v1.1-zh",),
}


def check_tts_model_files(adapter, folder):
    """核对本地合成模型的必要文件；不下载，不重算已有模型哈希。"""
    import json

    from speechio.errors import TTSError

    folder = Path(folder).expanduser().resolve()
    required = ["config.json"]
    if adapter == "qwen_tts":
        required += [
            "tokenizer_config.json",
            "vocab.json",
            "merges.txt",
            "preprocessor_config.json",
            "generation_config.json",
            "speech_tokenizer/config.json",
            "speech_tokenizer/preprocessor_config.json",
        ]
        for location in (folder, folder / "speech_tokenizer"):
            index = location / "model.safetensors.index.json"
            try:
                weights = (
                    set(json.loads(index.read_text(encoding="utf-8"))["weight_map"].values())
                    if index.is_file()
                    else {"model.safetensors"}
                )
            except (OSError, ValueError, KeyError, TypeError) as exc:
                raise TTSError("MODEL_INCOMPLETE", "load_model", "模型分片索引无效") from exc
            for weight in weights:
                path = (location / weight).resolve()
                if not path.is_relative_to(folder):
                    raise TTSError("MODEL_INCOMPLETE", "load_model", "模型分片不在所选目录中")
                required.append(str(path.relative_to(folder)))
    elif adapter == "kokoro":
        required += ["kokoro-v1_1-zh.pth", "english/config.cfg"]
        if not (folder / "voices").is_dir() or not any((folder / "voices").glob("*.pt")):
            raise TTSError("MODEL_INCOMPLETE", "load_model", "Kokoro 目录缺少 voices 音色文件")
    else:
        raise TTSError("CAPABILITY_UNSUPPORTED", "load_model", "未知本地合成适配器")
    for name in required:
        path = folder / name
        if not path.is_file() or not path.stat().st_size:
            raise TTSError("MODEL_INCOMPLETE", "load_model", f"模型文件缺失或为空：{path}")
    return folder
