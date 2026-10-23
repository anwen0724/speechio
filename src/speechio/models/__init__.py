"""模型来源配置与显式下载；导入不加载模型。"""

from speechio.models.config import LocalModelConfig, ModelConfig, RemoteModelConfig
from speechio.models.download import download_model

__all__ = ["LocalModelConfig", "RemoteModelConfig", "ModelConfig", "download_model"]
