"""显式下载 ASR 模型；加载只接收本地目录。"""

from os import PathLike
from pathlib import Path

from speechio.runtime.models.prepare import download_asr_model


def download_model(*, engine: str, model_id: str, destination: str | PathLike) -> Path:
    """把模型保存到调用方指定的目录，返回绝对路径。"""
    return download_asr_model(engine, model_id, destination)
