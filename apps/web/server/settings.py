"""本机配置负责路径与密钥，浏览器只选择公开的预设 ID。"""

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import Field, SecretStr, create_model
from pydantic_settings import BaseSettings, SettingsConfigDict

from speechio import ASRError, LocalModelConfig, RemoteModelConfig, WakeConfig

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.toml"
DEFAULT_ENV_PATH = Path(__file__).resolve().parents[1] / ".env"


class CredentialSettings(BaseSettings):
    """按直填、进程环境、.env 的顺序取密钥；空白值不遮挡下一来源。"""

    model_config = SettingsConfigDict(
        extra="ignore", case_sensitive=True, hide_input_in_errors=True
    )

    @classmethod
    def settings_customise_sources(
        cls, settings_cls, init_settings, env_settings, dotenv_settings, file_secret_settings
    ):
        def without_blank(source):
            def read():
                return {
                    key: value
                    for key, value in source().items()
                    if value is not None and (not isinstance(value, str) or value.strip())
                }

            return read

        return tuple(
            without_blank(source) for source in (init_settings, env_settings, dotenv_settings)
        )


def model_summary(key, model):
    """公开本次请求采用的模型参数，不暴露服务器路径。"""
    if isinstance(model, RemoteModelConfig):
        return {
            "id": key,
            "source": "remote",
            "provider": model.provider,
            "model": model.model,
            "timeout_seconds": model.timeout_seconds,
        }
    return {
        "id": key,
        "source": "local",
        "engine": model.engine,
        "device": model.device,
        "device_index": model.device_index,
        "compute_type": model.compute_type,
    }


def load_model(value):
    """应用读取密钥并转换为核心配置；密钥来源不进入核心包。"""
    options = dict(value)
    source = options.pop("source", "local")
    if source == "local":
        return LocalModelConfig(**options)
    if source != "remote":
        raise ValueError("模型 source 应为 local 或 remote")
    api_key = options.pop("apikey", None)
    env_name = options.pop("api_key_env", None)
    if env_name is not None:
        if not isinstance(env_name, str) or not env_name.strip():
            raise ValueError("api_key_env 应为非空环境变量名称")
        # 每个模型可指定不同变量；Pydantic Settings 负责解析 .env 和合并来源。
        credential_type = create_model(
            "ModelCredential",
            __base__=CredentialSettings,
            apikey=(SecretStr | None, Field(default=None, validation_alias=env_name)),
        )
        credential = credential_type(
            **{env_name: api_key},
            _env_file=DEFAULT_ENV_PATH,
            _env_file_encoding="utf-8-sig",
        )
        api_key = credential.apikey.get_secret_value() if credential.apikey else None
    elif isinstance(api_key, str) and not api_key.strip():
        api_key = None
    # 未配置密钥时仍可打开页面和使用本地模型，远程任务在开始边界拒绝。
    return RemoteModelConfig(api_key=api_key or None, **options)


@dataclass
class Settings:
    """一个服务进程的配置；Web 依赖不进入 speechio 发布包。"""

    models: dict[str, LocalModelConfig | RemoteModelConfig]
    debug: bool = False
    upload_limit_bytes: int = 100 * 1024 * 1024
    upload_dir: Path = Path(".runtime/web-uploads")
    wakes: dict[str, WakeConfig] = field(default_factory=dict)

    def __post_init__(self):
        if not self.models:
            raise ValueError("配置中至少需要一个模型，请填写 [models.模型名称]")
        if self.upload_limit_bytes <= 0:
            raise ValueError("上传大小限制必须为正数")
        for model in self.models.values():
            model.capabilities

    @classmethod
    def load(cls, path=None, *, debug=False):
        """默认读取 Web 目录的配置；配置内相对路径仍按启动目录解释。"""
        path = (DEFAULT_CONFIG_PATH if path is None else Path(path)).expanduser().resolve()
        try:
            content = path.read_text(encoding="utf-8-sig")
        except FileNotFoundError as exc:
            example = DEFAULT_CONFIG_PATH.with_name("config.example.toml")
            raise ValueError(
                f"配置文件不存在：{path}。请复制 {example} 到该路径并填写模型目录。"
            ) from exc
        except (OSError, UnicodeError) as exc:
            raise ValueError(f"无法读取配置文件 {path}：{exc}") from exc
        try:
            data = tomllib.loads(content)
            options = data.get("web", {})
            return cls(
                debug=debug,
                upload_limit_bytes=int(options.get("upload_limit_mb", 100) * 1024 * 1024),
                upload_dir=Path(options.get("upload_dir", ".runtime/web-uploads")),
                models={key: load_model(value) for key, value in data.get("models", {}).items()},
                wakes={
                    key: WakeConfig(
                        **{
                            name: tuple(value) if name in ("model_paths", "keywords") else value
                            for name, value in options.items()
                        }
                    )
                    for key, options in data.get("wakes", {}).items()
                },
            )
        except (ValueError, TypeError, AttributeError, ASRError) as exc:
            # TOML 语法和外部字段错误带上来源，不退回另一份模型列表。
            raise ValueError(f"配置文件无效 {path}：{exc}") from exc

    def public(self):
        """能力声明可提前读取；不向浏览器返回本地路径或 AccessKey。"""
        return {
            "debug": self.debug,
            "upload_limit_bytes": self.upload_limit_bytes,
            "models": [
                {
                    **model_summary(key, model),
                    "devices": []
                    if isinstance(model, RemoteModelConfig)
                    else (["cpu", "gpu"] if model.engine == "whisper" else ["cpu"]),
                    "compute_types": []
                    if isinstance(model, RemoteModelConfig)
                    else (
                        [None, "int8", "float16", "float32", "int8_float16"]
                        if model.engine == "whisper"
                        else [None, "int8"]
                    ),
                    **(
                        {"api_key_configured": bool(model.api_key)}
                        if isinstance(model, RemoteModelConfig)
                        else {}
                    ),
                    "snapshot": model.capabilities.snapshot,
                    "streaming": model.capabilities.streaming,
                    "word_timestamps": model.capabilities.word_timestamps,
                }
                for key, model in self.models.items()
            ],
            "wakes": [{"id": key, "backend": wake.backend} for key, wake in self.wakes.items()],
        }
