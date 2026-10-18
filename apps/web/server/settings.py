"""本机配置负责路径与密钥，浏览器只选择公开的预设 ID。"""

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from speechio import ModelConfig, WakeConfig

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.toml"


def model_summary(key, model):
    """公开本次请求采用的模型参数，不暴露服务器路径。"""
    return {
        "id": key,
        "engine": model.engine,
        "device": model.device,
        "device_index": model.device_index,
        "compute_type": model.compute_type,
    }


@dataclass
class Settings:
    """一个服务进程的配置；Web 依赖不进入 speechio 发布包。"""

    models: dict[str, ModelConfig]
    debug: bool = False
    upload_limit_bytes: int = 100 * 1024 * 1024
    upload_dir: Path = Path(".runtime/web-uploads")
    wakes: dict[str, WakeConfig] = field(default_factory=dict)

    def __post_init__(self):
        if not self.models:
            raise ValueError("配置中至少需要一个模型，请填写 [models.模型名称]")
        if self.upload_limit_bytes <= 0:
            raise ValueError("上传大小限制必须为正数")

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
                models={key: ModelConfig(**value) for key, value in data.get("models", {}).items()},
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
        except (ValueError, TypeError, AttributeError) as exc:
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
                    "devices": ["cpu", "gpu"] if model.capabilities.snapshot else ["cpu"],
                    "compute_types": [None, "int8", "float16", "float32", "int8_float16"]
                    if model.capabilities.snapshot
                    else [None, "int8"],
                    "snapshot": model.capabilities.snapshot,
                    "streaming": model.capabilities.streaming,
                    "word_timestamps": model.capabilities.word_timestamps,
                }
                for key, model in self.models.items()
            ],
            "wakes": [{"id": key, "backend": wake.backend} for key, wake in self.wakes.items()],
        }
