"""显式下载识别或合成模型；加载只接收本地目录。"""

from os import PathLike
from pathlib import Path

from speechio.models.prepare import download_model_files


def download_model(*, engine: str, model_id: str, destination: str | PathLike) -> Path:
    """把模型保存到调用方指定的目录，返回绝对路径。"""
    return download_model_files(engine, model_id, destination)
