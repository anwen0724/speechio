"""服务拥有的启动和容量配置，不读取客户端提供的模型路径。"""

from dataclasses import dataclass
from ipaddress import ip_address
from math import isfinite

from realtime_asr import ASRError, RecognizerConfig
from realtime_asr.backends.base import validate_engine


def is_loopback(host):
    if host == "localhost":
        return True
    try:
        return ip_address(host).is_loopback
    except ValueError:
        return False


@dataclass(frozen=True, kw_only=True)
class ServerConfig:
    """启动字段固定；运行字段通过明确的配置更新重新建立值。"""

    recognizer: RecognizerConfig | None = None
    engine: object = None
    host: str = "127.0.0.1"
    port: int = 8000
    token: str | None = None
    ssl_certfile: str | None = None
    ssl_keyfile: str | None = None
    proxy_tls: bool = False
    trusted_proxy_ips: str | None = None
    allowed_origins: tuple[str, ...] = ()
    realtime: bool = True
    acoustic_boundaries: bool = False
    vad_model: str | None = None
    max_sessions: int = 16
    max_tasks: int = 32
    max_task_audio_samples: int = 9600000
    max_audio_seconds: float = 30
    max_pending_units: int = 8
    max_http_bytes: int = 16 * 1024 * 1024
    max_packet_bytes: int = 1024 * 1024
    max_metadata_bytes: int = 64 * 1024
    max_pending_messages: int = 256
    open_timeout: float = 10
    idle_timeout: float = 300
    body_timeout: float = 30
    drain_timeout: float = 30
    send_timeout: float = 5
    close_timeout: float = 5

    @property
    def capabilities(self):
        """配置只使用提供方声明，不能根据后端名称再推导能力。"""
        return self.engine.capabilities if self.engine is not None else self.recognizer.capabilities

    def __post_init__(self):
        if (self.recognizer is None) == (self.engine is None) or (
            self.recognizer is not None and not isinstance(self.recognizer, RecognizerConfig)
        ):
            raise ASRError(
                "INVALID_CONFIGURATION", "configure_server", "本地配置与引擎应且仅应提供一个"
            )
        if self.engine is not None:
            validate_engine(self.engine)
        positive_ints = (
            self.port,
            self.max_sessions,
            self.max_tasks,
            self.max_task_audio_samples,
            self.max_pending_units,
            self.max_http_bytes,
            self.max_packet_bytes,
            self.max_metadata_bytes,
            self.max_pending_messages,
        )
        seconds = (
            self.max_audio_seconds,
            self.open_timeout,
            self.idle_timeout,
            self.body_timeout,
            self.drain_timeout,
            self.send_timeout,
            self.close_timeout,
        )
        if (
            any(type(value) is not int or value <= 0 for value in positive_ints)
            or any(
                type(value) not in (int, float) or not isfinite(value) or value <= 0
                for value in seconds
            )
            or self.port > 65535
            or self.max_audio_seconds > 600
            or int(self.max_audio_seconds * 16000) == 0
            or self.max_pending_messages < 4
            or self.max_metadata_bytes > self.max_packet_bytes
            or not isinstance(self.host, str)
            or not self.host.strip()
            or (
                self.token is not None
                and (not isinstance(self.token, str) or not self.token.strip())
            )
            or type(self.realtime) is not bool
            or type(self.acoustic_boundaries) is not bool
            or (self.acoustic_boundaries and (not self.realtime or not self.capabilities.snapshot))
            or type(self.proxy_tls) is not bool
        ):
            raise ASRError("INVALID_CONFIGURATION", "configure_server", "服务配置无效")
        if bool(self.ssl_certfile) != bool(self.ssl_keyfile):
            raise ASRError("INVALID_CONFIGURATION", "configure_server", "TLS 证书与私钥须同时提供")
        if self.proxy_tls and (
            not self.trusted_proxy_ips
            or "*" in {value.strip() for value in self.trusted_proxy_ips.split(",")}
        ):
            raise ASRError("INVALID_CONFIGURATION", "configure_server", "代理须明确可信地址")
        if not is_loopback(self.host) and (
            not self.token or not ((self.ssl_certfile and self.ssl_keyfile) or self.proxy_tls)
        ):
            raise ASRError("INVALID_CONFIGURATION", "configure_server", "远程监听需要 Token 与 TLS")
