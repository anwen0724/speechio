"""远程接入接口；调用方不需要安装本地模型后端。"""

from apps.client.connection import RemoteClient, RemoteSession

__all__ = ["RemoteClient", "RemoteSession"]
