"""本机配置负责路径与密钥，浏览器只选择公开的预设 ID。"""

import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path

from pydantic import Field, SecretStr, create_model
from pydantic_settings import BaseSettings, SettingsConfigDict

from speechio import (
    LocalModelConfig,
    RemoteModelConfig,
    SpeechioError,
    SynthesisOptions,
    Synthesizer,
    WakeWordConfig,
)
from speechio.models.factory import engine_capabilities

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.toml"
DEFAULT_ENV_PATH = Path(__file__).resolve().parents[1] / ".env"


def transcription_capabilities(model):
    """读取核心适配器能力声明，不因缺少密钥阻止页面打开。"""
    return engine_capabilities(model)


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


def model_summary(key, model, *, synthesis=False):
    """公开本次请求采用的模型参数，不暴露服务器路径。"""
    if isinstance(model, RemoteModelConfig):
        return {
            "id": key,
            "source": "remote",
            "provider": model.provider,
            "model": model.model,
            "timeout_seconds": model.timeout_seconds,
            **(
                {"task_timeout_seconds": 3600}
                if not synthesis and transcription_capabilities(model).file_transcription
                else {}
            ),
        }
    return {
        "id": key,
        "source": "local",
        "engine": model.adapter,
        "device": model.device,
        "device_index": model.device_index,
        "compute_type": model.compute_type,
    }


def load_model(value):
    """应用读取密钥并转换为核心配置；密钥来源不进入核心包。"""
    options = dict(value)
    source = options.pop("source", "local")
    if source == "local":
        # 兼容现有 Web 配置拼写；核心包统一使用 adapter。
        if "engine" in options:
            options["adapter"] = options.pop("engine")
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
class SynthesisPreset:
    """Web 的合成默认值与音色建议；不把应用字段传给核心模型配置。"""

    config: LocalModelConfig | RemoteModelConfig
    options: SynthesisOptions
    voices: tuple[str, ...] = ()
    max_buffer_seconds: float = 10.0

    def __post_init__(self):
        synth = Synthesizer(self.config, max_buffer_seconds=self.max_buffer_seconds)
        self.capabilities = synth.capabilities
        synth.close()

    def public(self, key):
        return {
            **model_summary(key, self.config, synthesis=True),
            "defaults": asdict(self.options),
            "voices": self.voices,
            "max_buffer_seconds": self.max_buffer_seconds,
            "capabilities": asdict(self.capabilities),
            "api_key_configured": bool(getattr(self.config, "api_key", None)),
            "voice_inputs": {
                "preset": isinstance(self.config, RemoteModelConfig)
                or self.config.adapter != "qwen_tts",
                "reference_text": getattr(self.config, "adapter", None) == "qwen_tts",
                "reference_hint": (
                    "参考录音需 5–60 秒、采样率至少 16 kHz；解码后的单声道 WAV 不超过 10 MiB。"
                    if isinstance(self.config, RemoteModelConfig)
                    else "上传清晰的单人录音；参考文字可选，应与录音内容一致。"
                ),
            },
        }


@dataclass
class Settings:
    """一个服务进程的配置；Web 依赖不进入 speechio 发布包。"""

    models: dict[str, LocalModelConfig | RemoteModelConfig]
    debug: bool = False
    upload_limit_bytes: int = 1024 * 1024 * 1024
    upload_dir: Path = Path(".runtime/web-uploads")
    wakes: dict[str, WakeWordConfig] = field(default_factory=dict)
    synthesis_models: dict[str, SynthesisPreset] = field(default_factory=dict)
    voices_dir: Path = Path("apps/web/data/voices")
    task_timeouts: dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        if not self.models and not self.synthesis_models:
            raise ValueError("配置中至少需要一个模型，请填写 [models.模型名称]")
        if self.upload_limit_bytes <= 0:
            raise ValueError("上传大小限制必须为正数")
        for model in self.models.values():
            transcription_capabilities(model)

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
            models, synthesis_models, task_timeouts = {}, {}, {}
            for key, value in data.get("models", {}).items():
                value = dict(value)
                task = value.pop("task", "transcription")
                if task == "synthesis":
                    defaults = SynthesisOptions(**value.pop("synthesis", {}))
                    voices = tuple(value.pop("voices", ()))
                    buffer_seconds = value.pop("max_buffer_seconds", 10.0)
                    synthesis_models[key] = SynthesisPreset(
                        load_model(value), defaults, voices, buffer_seconds
                    )
                elif task == "transcription":
                    task_timeouts[key] = value.pop("task_timeout_seconds", 3600)
                    models[key] = load_model(value)
                else:
                    raise ValueError("task 应为 transcription 或 synthesis")
            return cls(
                debug=debug,
                upload_limit_bytes=int(options.get("upload_limit_mb", 1024) * 1024 * 1024),
                upload_dir=Path(options.get("upload_dir", ".runtime/web-uploads")),
                models=models,
                synthesis_models=synthesis_models,
                voices_dir=Path(options.get("voices_dir", "apps/web/data/voices")),
                task_timeouts=task_timeouts,
                wakes={
                    key: WakeWordConfig(
                        **{
                            name: tuple(value) if name in ("model_paths", "keywords") else value
                            for name, value in options.items()
                        }
                    )
                    for key, options in data.get("wakes", {}).items()
                },
            )
        except (ValueError, TypeError, AttributeError, SpeechioError) as exc:
            # TOML 语法和外部字段错误带上来源，不退回另一份模型列表。
            raise ValueError(f"配置文件无效 {path}：{exc}") from exc

    def public(self):
        """能力声明可提前读取；不向浏览器返回本地路径或 AccessKey。"""
        return {
            "debug": self.debug,
            "upload_limit_bytes": self.upload_limit_bytes,
            "synthesis_models": [
                preset.public(key) for key, preset in self.synthesis_models.items()
            ],
            "models": [
                {
                    **model_summary(key, model),
                    **(
                        {"task_timeout_seconds": self.task_timeouts.get(key, 3600)}
                        if transcription_capabilities(model).file_transcription
                        else {}
                    ),
                    "devices": []
                    if isinstance(model, RemoteModelConfig)
                    else (["cpu", "gpu"] if model.adapter == "whisper" else ["cpu"]),
                    "compute_types": []
                    if isinstance(model, RemoteModelConfig)
                    else (
                        [None, "int8", "float16", "float32", "int8_float16"]
                        if model.adapter == "whisper"
                        else [None, "int8"]
                    ),
                    **(
                        {"api_key_configured": bool(model.api_key)}
                        if isinstance(model, RemoteModelConfig)
                        else {}
                    ),
                    "snapshot": transcription_capabilities(model).snapshot,
                    "transcription": transcription_capabilities(model).transcription,
                    "file_transcription": transcription_capabilities(model).file_transcription,
                    "url_transcription": transcription_capabilities(model).url_transcription,
                    "streaming": transcription_capabilities(model).streaming,
                    "word_timestamps": transcription_capabilities(model).word_timestamps,
                }
                for key, model in self.models.items()
            ],
            "wakes": [{"id": key, "backend": wake.backend} for key, wake in self.wakes.items()],
        }
