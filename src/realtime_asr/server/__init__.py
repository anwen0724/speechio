"""服务入口按需导入；基础 SDK 不加载 ASGI 框架。"""

from realtime_asr.server.config import ServerConfig

__all__ = ["ServerConfig", "create_app"]


def create_app(config):
    """创建应用，由 ASGI lifespan 负责模型准备和关闭。"""
    from realtime_asr.server.app import create_app as factory

    return factory(config)
