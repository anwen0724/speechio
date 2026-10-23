"""模型来源配置；配置对象不读取密钥文件，不加载模型或发起网络请求。"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from math import isfinite
from os import PathLike
from urllib.parse import urlsplit

from speechio.errors import STTError


@dataclass(frozen=True, kw_only=True)
class LocalModelConfig:
    """加载已有本地目录；下载由调用方单独执行。"""

    adapter: str
    model_path: str | PathLike
    device: str = "cpu"
    compute_type: str | None = None
    device_index: int = 0
    adapter_options: Mapping[str, object] | None = None


@dataclass(frozen=True, kw_only=True)
class RemoteModelConfig:
    """调用远程模型；base_url 为服务根地址，不含具体 API 路径。"""

    provider: str
    model: str
    api_key: str | None = field(default=None, repr=False)
    base_url: str | None = None
    timeout_seconds: float = 30.0

    def __post_init__(self):
        for name in ("provider", "model"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise STTError("INVALID_CONFIGURATION", "configure", f"{name} 不能为空")
        if self.api_key is not None and (
            not isinstance(self.api_key, str) or not self.api_key.strip()
        ):
            raise STTError("INVALID_CONFIGURATION", "configure", "api_key 应为非空字符串或 None")
        if (
            isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, (int, float))
            or not isfinite(self.timeout_seconds)
            or self.timeout_seconds <= 0
        ):
            raise STTError("INVALID_CONFIGURATION", "configure", "timeout_seconds 应为有限正数")
        if self.base_url is not None:
            try:
                url = urlsplit(self.base_url)
                # 访问 port 触发 urllib 的端口格式校验。
                url.port
                valid = (
                    url.scheme in ("http", "https")
                    and url.hostname
                    and not url.username
                    and not url.password
                    and url.path in ("", "/")
                    and not url.query
                    and not url.fragment
                )
            except (TypeError, ValueError):
                valid = False
            if not valid:
                raise STTError(
                    "INVALID_CONFIGURATION", "configure", "base_url 应为 HTTP(S) 服务根地址"
                )
