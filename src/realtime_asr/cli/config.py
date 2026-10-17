"""启动配置边界：读取 TOML、解释相对路径并交给既有参数解析器校验。"""

import argparse
import importlib.util
import json
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from realtime_asr import ASRError

LOCAL = ("mic", "file", "stdin")
REMOTE = ("remote-mic", "remote-file", "remote-stdin")
HOST = (*LOCAL, "serve")
INPUT = (*LOCAL, *REMOTE)

# 字段只定义映射、类型和适用入口；具体默认值与取值校验继续由原解析器负责。
FIELDS = {
    "model": {
        "backend": ("backend", str),
        "model_path": ("model", Path),
        "device": ("device", str),
        "device_index": ("device_index", int),
    },
    "gpu": {
        "cudnn_folder": ("cudnn_folder", Path),
        "cuda_directory": ("cuda_directory", Path),
    },
    "recognition": {
        "language": ("language", str),
        "mode": ("mode", str),
        "word_timestamps": ("word_timestamps", bool),
    },
    "microphone": {
        "device_index": ("input_device", int),
        "sample_rate": ("sample_rate", int),
        "channels": ("channels", int),
        "chunk_frames": ("chunk_frames", int),
        "queue_seconds": ("queue_seconds", float),
        "duration": ("duration", float),
    },
    "input": {
        "sample_rate": ("sample_rate", int),
        "channels": ("channels", int),
        "sample_type": ("sample_type", str),
    },
    "realtime": {
        "enabled": ("realtime", bool),
        "model_path": ("realtime_model", Path),
        "backend": ("realtime_backend", str),
        "device": ("realtime_device", str),
        "interval": ("realtime_interval", float),
        "initial_delay": ("realtime_initial_delay", float),
        "stable_text": ("stable_text", bool),
        "dual_realtime": ("dual_realtime", bool),
        "acoustic_boundaries": ("acoustic_boundaries", bool),
    },
    "ultrafast": {
        "model_path": ("ultrafast_model", Path),
        "backend": ("ultrafast_backend", str),
        "device": ("ultrafast_device", str),
    },
    "vad": {
        "model_path": ("vad_model", Path),
        "webrtc_mode": ("webrtc_mode", int),
        "silero_sensitivity": ("silero_sensitivity", float),
        "silero_end_detection": ("silero_end_detection", bool),
    },
    "segmentation": {
        "silence_confirmation": ("silence_confirmation", float),
        "post_speech_silence": ("post_speech_silence", float),
        "min_recording_duration": ("min_recording_duration", float),
        "min_gap": ("min_gap", float),
        "pre_roll_duration": ("pre_roll_duration", float),
        "trim_pre_roll": ("trim_pre_roll", bool),
        "backdate_duration": ("backdate_seconds", float),
    },
    "endpoint": {
        "early_transcription_silence": ("early_transcription_silence", float),
        "dynamic_silence": ("dynamic_silence", bool),
        "punctuation_split": ("punctuation_split", bool),
    },
    "candidate": {
        "enabled": ("candidates", bool),
        "model_path": ("candidate_model", Path),
        "backend": ("candidate_backend", str),
        "device": ("candidate_device", str),
        "only": ("candidate_only", bool),
        "kind": ("candidate_kind", str),
        "every": ("candidate_every", float),
        "hint": ("candidate_hint", bool),
        "completion_mode": ("completion_mode", str),
        "retain_for_correction": ("retain_for_correction", bool),
        "correct_after_final": ("correct_after_final", bool),
    },
    "wake": {
        "enabled": ("wake", bool),
        "backend": ("wake_backend", str),
        "keywords": ("wake_keywords", list),
        "model_paths": ("wake_models", list),
        "model_path": ("wake_parameter_model", Path),
        "feature_model_dir": ("wake_feature_dir", Path),
        "sensitivity": ("wake_sensitivity", float),
        "activation_delay": ("wake_activation_delay", float),
        "speech_timeout": ("wake_speech_timeout", float),
        "trim_duration": ("wake_trim_duration", float),
        "followup_window": ("wake_followup_window", float),
    },
    "server": {
        "host": ("host", str),
        "port": ("port", int),
        "token_env": ("token_env", str),
        "ssl_certfile": ("ssl_certfile", Path),
        "ssl_keyfile": ("ssl_keyfile", Path),
        "proxy_tls": ("proxy_tls", bool),
        "trusted_proxy_ips": ("trusted_proxy_ips", str),
        "allowed_origins": ("allowed_origin", list),
        **{
            name: (name, int)
            for name in (
                "max_sessions",
                "max_tasks",
                "max_task_audio_samples",
                "max_pending_units",
                "max_http_bytes",
                "max_packet_bytes",
                "max_metadata_bytes",
                "max_pending_messages",
            )
        },
        **{
            name: (name, float)
            for name in (
                "max_audio_seconds",
                "open_timeout",
                "idle_timeout",
                "body_timeout",
                "drain_timeout",
                "send_timeout",
                "close_timeout",
            )
        },
    },
    "remote": {
        "url": ("url", str),
        "token_env": ("token_env", str),
        "timeout": ("timeout", float),
    },
    "output": {"format": ("output", str)},
    "limits": {
        "max_audio_seconds": ("max_audio_seconds", float),
        "drain_timeout": ("drain_timeout", float),
        "close_timeout": ("close_timeout", float),
    },
}
SCOPES = {
    "model": HOST,
    "gpu": HOST,
    "recognition": INPUT,
    "microphone": ("mic", "remote-mic", "devices"),
    "input": ("stdin", "remote-stdin"),
    "realtime": (*HOST, *REMOTE),
    "ultrafast": HOST,
    "vad": HOST,
    "segmentation": LOCAL,
    "endpoint": INPUT,
    "candidate": (*HOST, *REMOTE),
    "wake": (*HOST, *REMOTE),
    "server": ("serve",),
    "remote": REMOTE,
    "output": (*INPUT, "devices"),
    "limits": LOCAL,
}
OVERRIDES = {
    ("microphone", "device_index"): ("mic", "remote-mic"),
    ("microphone", "sample_rate"): ("mic", "remote-mic"),
    ("microphone", "chunk_frames"): ("mic",),
    ("microphone", "queue_seconds"): ("mic",),
    ("microphone", "duration"): ("mic", "remote-mic"),
    **{
        ("realtime", key): HOST
        for key in (
            "model_path",
            "backend",
            "device",
            "interval",
            "initial_delay",
        )
    },
    ("realtime", "stable_text"): INPUT,
    ("realtime", "dual_realtime"): INPUT,
    ("realtime", "acoustic_boundaries"): HOST,
    **{
        ("vad", key): LOCAL
        for key in (
            "webrtc_mode",
            "silero_sensitivity",
            "silero_end_detection",
        )
    },
    **{("candidate", key): HOST for key in ("model_path", "backend", "device", "only")},
    **{
        ("candidate", key): INPUT
        for key in (
            "kind",
            "every",
            "hint",
            "completion_mode",
            "retain_for_correction",
            "correct_after_final",
        )
    },
    **{("wake", key): HOST for key in FIELDS["wake"] if key != "enabled"},
}
REMOTE_GLOBAL = {"url", "token_env", "timeout", "language", "output", "word_timestamps"}


def split_config(argv):
    """全局选择器可放在子命令前后；不读取或修改任何配置。"""
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--config", type=Path)
    args, remaining = parser.parse_known_args(argv)
    return args.config, remaining


def forward_config(argv, path):
    return [*argv, "--config", str(path)] if path is not None else argv


def remote_source(argv):
    """跳过远程全局选项的值，定位来源子命令，不把值误认为命令。"""
    index = 0
    while index < len(argv):
        token = argv[index]
        name = token.split("=", 1)[0].removeprefix("--").replace("-", "_")
        if name not in REMOTE_GLOBAL and name != "no_word_timestamps":
            return token, index
        index += 1 if "=" in token or name in ("word_timestamps", "no_word_timestamps") else 2
    return None, index


def order_remote_arguments(arguments):
    """全局选项回到来源命令之前，保持有无配置文件时的覆盖语法一致。"""
    global_options, remaining = [], []
    index = 0
    while index < len(arguments):
        token = arguments[index]
        option = token.split("=", 1)[0].removeprefix("--").removeprefix("no-").replace("-", "_")
        if token.startswith("--") and option in REMOTE_GLOBAL:
            size = 1 if "=" in token or option == "word_timestamps" else 2
            global_options.extend(arguments[index : index + size])
            index += size
        else:
            remaining.append(token)
            index += 1
    return [*global_options, *remaining]


@dataclass
class ConfigArguments:
    """同一份启动结果保存最终参数的来源，不把配置元数据交给核心。"""

    argv: list[str]
    path: Path | None = None
    sources: dict = field(default_factory=dict)
    explicit: set = field(default_factory=set)

    def describe(self, args, target):
        values = {name: value for name, value in vars(args).items() if name != "launch_config"}
        return {
            "command": target,
            "config_file": str(self.path) if self.path else None,
            "values": values,
            "sources": {
                name: "command_line" if name in self.explicit else self.sources.get(name, "default")
                for name in values
            },
        }


def configure(argv, target):
    """配置转为既有命令参数；显式参数整项覆盖，最终仍使用原参数校验。"""
    selected, original = split_config(list(argv))
    if target in REMOTE:
        original = order_remote_arguments(original)
    invocation = ConfigArguments(original)
    if "--help" in original or "-h" in original:
        return invocation
    invocation.explicit = {
        token.split("=", 1)[0].removeprefix("--").removeprefix("no-").replace("-", "_")
        for token in original
        if token.startswith("--")
    }
    invocation.explicit.update(("command", "input", "paths"))
    path = (selected or Path("config.toml")).resolve()
    if not path.exists() and selected is None:
        return invocation
    try:
        with path.open("rb") as source:
            data = tomllib.load(source)
    except (OSError, ValueError) as exc:
        raise ASRError(
            "INVALID_CONFIGURATION", "load_config", f"无法读取配置 {path}：{exc}"
        ) from exc
    invocation.path = path
    explicit = invocation.explicit
    configured, global_remote = [], []
    for section, values in data.items():
        if section not in FIELDS or not isinstance(values, dict):
            raise ASRError("INVALID_CONFIGURATION", "load_config", f"未知配置分组：{section}")
        for key, value in values.items():
            if key not in FIELDS[section]:
                raise ASRError(
                    "INVALID_CONFIGURATION", "load_config", f"未知配置字段：{section}.{key}"
                )
            option, kind = FIELDS[section][key]
            expected = str if kind is Path else kind
            valid = type(value) is expected or (kind is float and type(value) is int)
            if kind is list:
                valid = valid and all(isinstance(item, str) and item for item in value)
            if not valid:
                raise ASRError(
                    "INVALID_CONFIGURATION", "load_config", f"配置类型无效：{section}.{key}"
                )
            if target not in OVERRIDES.get((section, key), SCOPES[section]) or option in explicit:
                continue
            if kind is Path:
                value = str((path.parent / value).resolve())
            elif option == "wake_models":
                value = [str((path.parent / item).resolve()) for item in value]
            flag = "--" + option.replace("_", "-")
            if kind is bool:
                arguments = [flag if value else "--no-" + option.replace("_", "-")]
            elif kind is list:
                if not value:
                    continue
                arguments = (
                    [part for item in value for part in (flag, item)]
                    if option == "allowed_origin"
                    else [flag, *value]
                )
            else:
                arguments = [flag, str(value)]
            destination = (
                global_remote if target in REMOTE and option in REMOTE_GLOBAL else configured
            )
            destination.extend(arguments)
            invocation.sources[option] = f"config:{section}.{key}"
    # CPU 模式忽略文件中的 GPU 路径；命令行显式不兼容参数仍由原契约拒绝。
    device = data.get("model", {}).get("device", "cpu")
    for index, token in enumerate(original):
        if token == "--device" and index + 1 < len(original):
            device = original[index + 1]
        elif token.startswith("--device="):
            device = token.partition("=")[2]
    if target in HOST and device != "cuda":
        for option in ("cudnn_folder", "cuda_directory"):
            if option in invocation.sources:
                flag = "--" + option.replace("_", "-")
                index = configured.index(flag)
                del configured[index : index + 2]
                del invocation.sources[option]
    invocation.argv = [*global_remote, *original, *configured]
    return invocation


def parse_target(target, argv):
    """管理命令复用实际启动解析，不维护第二套默认值或功能规则。"""
    if target == "serve":
        from realtime_asr.server.__main__ import parse_options

        return parse_options(argv)
    if target in REMOTE:
        from realtime_asr.client.__main__ import parse_options

        return parse_options([target.removeprefix("remote-"), *argv], diagnostic=True)
    from realtime_asr.cli.args import parse_args

    return parse_args([target, *argv], diagnostic=True)


def check_paths(args, target):
    """只核对选中入口的文件、目录与依赖；完整性及模型加载另行执行。"""
    modules = set()
    if target in REMOTE:
        from urllib.parse import urlsplit

        parts = urlsplit(args.url)
        if parts.scheme not in ("http", "https") or not parts.netloc:
            raise ASRError("INVALID_CONFIGURATION", "check_config", "remote.url 需要 HTTP(S) 地址")
        modules.update(("httpx", "websocket"))
    else:
        from realtime_asr.backends.local import RecognizerConfig, resolve_config

        model = RecognizerConfig(
            model_path=args.model,
            backend=args.backend,
            device=args.device,
            device_index=args.device_index,
            cudnn_folder=args.cudnn_folder,
            cuda_directory=args.cuda_directory,
        )
        from realtime_asr.cli.args import candidate_role, realtime_roles, wake_role

        resolve_config(model)
        realtime, ultrafast = realtime_roles(args, model)
        candidate = candidate_role(args, model)
        selected_models = [model]
        for role in (realtime, ultrafast, candidate):
            if role is not None and role.model is not None:
                resolve_config(role.model)
                selected_models.append(role.model)
        for selected in selected_models:
            modules.add("faster_whisper" if selected.backend == "whisper" else "sherpa_onnx")
        if target in LOCAL and args.mode == "auto":
            from realtime_asr import EndpointConfig, SegmentationConfig, VadConfig

            VadConfig(
                model_path=args.vad_model,
                webrtc_mode=args.webrtc_mode,
                silero_sensitivity=args.silero_sensitivity,
                silero_end_detection=args.silero_end_detection,
            )
            SegmentationConfig(
                silence_confirmation=args.silence_confirmation,
                post_speech_silence=args.post_speech_silence,
                min_recording_duration=args.min_recording_duration,
                min_gap=args.min_gap,
                pre_roll_duration=args.pre_roll_duration,
                trim_pre_roll=args.trim_pre_roll,
                backdate_duration=args.backdate_seconds,
            )
            EndpointConfig(
                early_transcription_silence=args.early_transcription_silence,
                dynamic_silence=args.dynamic_silence,
                punctuation_split=args.punctuation_split,
            )
        if args.wake:
            wake_role(args)
        for name in (
            "model",
            "realtime_model",
            "ultrafast_model",
            "candidate_model",
            "cudnn_folder",
            "cuda_directory",
        ):
            value = getattr(args, name, None)
            if value and not Path(value).is_dir():
                raise ASRError(
                    "INVALID_CONFIGURATION", "check_config", f"目录不存在：{name}={value}"
                )
        if target == "serve":
            for name in ("ssl_certfile", "ssl_keyfile"):
                value = getattr(args, name)
                if value and not Path(value).is_file():
                    raise ASRError(
                        "INVALID_CONFIGURATION", "check_config", f"TLS 文件不存在：{value}"
                    )
            from realtime_asr.server.__main__ import build_config

            build_config(args)
            modules.update(("fastapi", "uvicorn", "websockets"))
        if target == "serve" or args.mode == "auto":
            if args.vad_model:
                modules.update(("webrtcvad", "onnxruntime"))
                if not Path(args.vad_model).is_file():
                    raise ASRError("INVALID_CONFIGURATION", "check_config", "VAD 文件不存在")
        if args.wake:
            modules.add("pvporcupine" if args.wake_backend == "porcupine" else "openwakeword")
            for value in (*args.wake_models, args.wake_parameter_model):
                if value and not Path(value).is_file():
                    raise ASRError(
                        "INVALID_CONFIGURATION", "check_config", f"唤醒文件不存在：{value}"
                    )
            if args.wake_feature_dir and not Path(args.wake_feature_dir).is_dir():
                raise ASRError("INVALID_CONFIGURATION", "check_config", "唤醒特征目录不存在")
    if target in ("mic", "remote-mic"):
        modules.add("pyaudio")
    if target in ("file", "remote-file"):
        modules.update(("soundfile", "soxr"))
    missing = [name for name in sorted(modules) if importlib.util.find_spec(name) is None]
    if missing:
        raise ASRError(
            "OPTIONAL_DEPENDENCY_MISSING", "check_config", "缺少依赖：" + ", ".join(missing)
        )


def main(argv, *, path=None, stdout=None, stderr=None):
    """配置管理输出结构化结果，始终不加载模型或创建设备、连接。"""
    output, errors = stdout or sys.stdout, stderr or sys.stderr
    parser = argparse.ArgumentParser(prog="asr config")
    commands = parser.add_subparsers(dest="action", required=True)
    commands.add_parser("init", help="生成本机 config.toml，不覆盖已有文件")
    for name in ("check", "show"):
        command = commands.add_parser(name)
        command.add_argument("target", nargs="?", default="mic", choices=(*LOCAL, "serve", *REMOTE))
    options, overrides = parser.parse_known_args(argv)
    try:
        if options.action == "init":
            if overrides:
                parser.error("init 不接收识别参数")
            destination = (path or Path("config.toml")).resolve()
            with destination.open("x", encoding="utf-8") as target:
                target.write(
                    Path(__file__).with_name("config.example.toml").read_text(encoding="utf-8")
                )
            output.write(
                json.dumps(
                    {"status": "created", "config_file": str(destination)}, ensure_ascii=False
                )
                + "\n"
            )
            return 0
        if options.action == "check" and not (path or Path("config.toml")).is_file():
            raise ASRError(
                "INVALID_CONFIGURATION", "check_config", "配置文件不存在，先运行 asr config init"
            )
        args = parse_target(options.target, forward_config(overrides, path))
        report = args.launch_config.describe(args, options.target)
        if options.action == "check":
            check_paths(args, options.target)
            report = {
                "status": "valid",
                "command": options.target,
                "config_file": report["config_file"],
                "model_loaded": False,
            }
        output.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        return 0
    except (ASRError, OSError, ValueError) as exc:
        errors.write(f"[CONFIG_ERROR] {exc}\n")
        return 2
