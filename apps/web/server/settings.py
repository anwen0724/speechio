"""本机配置负责路径与密钥，浏览器只选择公开的预设 ID。"""

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from speechio import ModelConfig, WakeConfig


@dataclass
class Settings:
    """一个服务进程的配置；Web 依赖不进入 speechio 发布包。"""

    debug: bool = False
    upload_limit_bytes: int = 100 * 1024 * 1024
    upload_dir: Path = Path(".runtime/web-uploads")
    models: dict[str, ModelConfig] = field(
        default_factory=lambda: {
            "whisper-tiny": ModelConfig(
                engine="whisper", model_path=".runtime/models/whisper-tiny"
            ),
            "whisper-base": ModelConfig(
                engine="whisper", model_path=".runtime/models/whisper-base"
            ),
        }
    )
    wakes: dict[str, WakeConfig] = field(default_factory=dict)

    def __post_init__(self):
        if self.upload_limit_bytes <= 0 or not self.models:
            raise ValueError("必须配置至少一个模型，上传大小限制必须为正数")

    @classmethod
    def load(cls, path=None, *, debug=False):
        """从本机 TOML 读取模型与唤醒预设；相对路径按启动目录解释。"""
        if path is None:
            return cls(debug=debug)
        data = tomllib.loads(Path(path).read_text(encoding="utf-8-sig"))
        options = data.get("web", {})
        return cls(
            debug=debug,
            upload_limit_bytes=int(options.get("upload_limit_mb", 100) * 1024 * 1024),
            upload_dir=Path(options.get("upload_dir", ".runtime/web-uploads")),
            models={key: ModelConfig(**value) for key, value in data["models"].items()},
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

    def public(self):
        """能力声明可提前读取；不向浏览器返回本地路径或 AccessKey。"""
        return {
            "debug": self.debug,
            "upload_limit_bytes": self.upload_limit_bytes,
            "models": [
                {
                    "id": key,
                    "engine": model.engine,
                    "device": model.device,
                    "snapshot": model.capabilities.snapshot,
                    "streaming": model.capabilities.streaming,
                    "word_timestamps": model.capabilities.word_timestamps,
                }
                for key, model in self.models.items()
            ],
            "wakes": [{"id": key, "backend": wake.backend} for key, wake in self.wakes.items()],
        }
