"""启动配置边界：读取 TOML、解释相对路径并生成配置模板。"""

import argparse
import importlib.util
import json
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from apps.cli.options import (
    OPTIONS,
    SECTION_COMMENTS,
    TARGETS,
    candidate_role,
    realtime_roles,
    wake_role,
)
from speechio import ASRError
from speechio.bundled import default_vad_model

SCALARS = {
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
    "model": str,
    "path": str,
}


def _grouped():
    """按声明顺序返回配置分组；模板和读取共用同一份分组。"""
    result = {}
    for option in OPTIONS:
        result.setdefault(option.section, []).append(option)
    return result


SECTIONS = _grouped()


def split_config(argv):
    """全局选择器可放在子命令前后；不读取或修改任何配置。"""
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--config", type=Path)
    args, remaining = parser.parse_known_args(argv)
    return args.config, remaining


def forward_config(argv, path):
    return [*argv, "--config", str(path)] if path is not None else argv


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
    """配置转为命令参数；显式参数整项覆盖，最终仍由解析器校验。"""
    selected, original = split_config(list(argv))
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
    data = _load(path)
    invocation.path = path
    configured = []
    for section, values in data.items():
        options = SECTIONS.get(section)
        if options is None or not isinstance(values, dict):
            raise ASRError("INVALID_CONFIGURATION", "load_config", f"未知配置分组：{section}")
        names = {option.config_key: option for option in options}
        for key, value in values.items():
            option = names.get(key)
            if option is None:
                raise ASRError(
                    "INVALID_CONFIGURATION", "load_config", f"未知配置字段：{section}.{key}"
                )
            _check(option, section, key, value)
            if target not in option.commands or option.dest in invocation.explicit:
                continue
            _extend(configured, invocation, option, value, path)
    invocation.argv = [*original, *configured]
    return invocation


def _load(path):
    try:
        with path.open("rb") as source:
            return tomllib.load(source)
    except (OSError, ValueError) as exc:
        raise ASRError(
            "INVALID_CONFIGURATION", "load_config", f"无法读取配置 {path}：{exc}"
        ) from exc


def _check(option, section, key, value):
    """配置值只核对类型；取值范围由同一选项声明在解析时校验。"""
    kind = option.kind
    if kind == "float":
        valid = type(value) in (int, float)
    elif kind == "int":
        valid = type(value) is int
    elif kind in ("strlist", "pathlist"):
        valid = isinstance(value, list) and all(isinstance(item, str) and item for item in value)
    else:
        valid = type(value) is SCALARS[kind]
    if not valid:
        raise ASRError("INVALID_CONFIGURATION", "load_config", f"配置类型无效：{section}.{key}")


def _extend(configured, invocation, option, value, path):
    flag = option.config_flag
    if option.kind == "bool":
        configured.append(flag if value else "--no-" + option.flag)
    elif option.kind == "strlist":
        if not value:
            return
        configured.extend([flag, *value])
    elif option.kind == "pathlist":
        if not value:
            return
        configured.extend([flag, *(str((path.parent / item).resolve()) for item in value)])
    elif option.kind == "path":
        configured.extend([flag, str((path.parent / value).resolve())])
    elif option.kind == "model":
        configured.extend([flag, _model_value(path.parent, value)])
    else:
        configured.extend([flag, str(value)])
    invocation.sources[option.dest] = f"config:{option.section}.{option.config_key}"


def _model_value(base, value):
    """模型名保持原样，看起来是路径的值按配置文件目录解析。"""
    if Path(value).is_absolute() or value.startswith(".") or "/" in value or "\\" in value:
        return str((base / value).resolve())
    return value


def template():
    """按选项声明生成配置模板；示例值只作提示。"""
    lines = [
        "# 本机启动配置。修改后重新启动命令生效。",
        "# 本文件由 asr config init 生成；相对路径以本文件目录为基准。",
        "# 命令行参数只覆盖当次运行。",
        "",
    ]
    for section, options in SECTIONS.items():
        lines.append("# " + SECTION_COMMENTS.get(section, section))
        lines.append(f"[{section}]")
        for option in options:
            comment = f"  # {option.comment}" if option.comment else ""
            if option.default is not None:
                lines.append(f"{option.config_key} = {_toml(option.default)}{comment}")
            elif option.sample is not None and option.active_sample:
                lines.append(f"{option.config_key} = {option.sample}{comment}")
            elif option.sample is not None:
                lines.append(f"# {option.config_key} = {option.sample}{comment}")
            else:
                lines.append(f"# {option.config_key} = <值>{comment}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _toml(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_toml(item) for item in value) + "]"
    return json.dumps(str(value), ensure_ascii=False)


def parse_target(target, argv):
    """管理命令复用实际启动解析，不维护第二套默认值或功能规则。"""
    from apps.cli.parser import parse_args

    return parse_args([target, *argv], diagnostic=True)


def check_paths(args, target):
    """只核对选中入口的文件、目录与依赖；完整性及模型加载另行执行。"""
    modules = set()
    from speechio.engines.local import ModelConfig, resolve_config

    model = ModelConfig(
        model=args.model,
        engine=args.engine,
        device=args.device,
        device_index=args.device_index,
    )
    resolve_config(model)
    realtime, ultrafast = realtime_roles(args, model)
    candidate = candidate_role(args, model)
    selected_models = [model]
    for role in (realtime, ultrafast, candidate):
        if role is not None and role.model is not None:
            resolve_config(role.model)
            selected_models.append(role.model)
    for selected in selected_models:
        modules.add("faster_whisper" if selected.engine == "whisper" else "sherpa_onnx")
    if args.mode == "auto":
        from speechio import EndpointConfig, SegmentationConfig, VadConfig

        VadConfig(
            model_path=args.vad_model or default_vad_model(),
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
    ):
        value = getattr(args, name, None)
        if value and not Path(value).is_dir():
            raise ASRError("INVALID_CONFIGURATION", "check_config", f"目录不存在：{name}={value}")
    if args.mode == "auto":
        modules.update(("webrtcvad", "onnxruntime"))
        if args.vad_model and not Path(args.vad_model).is_file():
            raise ASRError("INVALID_CONFIGURATION", "check_config", "VAD 文件不存在")
    if args.wake:
        modules.add("pvporcupine" if args.wake_backend == "porcupine" else "openwakeword")
        for value in (*args.wake_models, args.wake_parameter_model):
            if value and not Path(value).is_file():
                raise ASRError("INVALID_CONFIGURATION", "check_config", f"唤醒文件不存在：{value}")
        if args.wake_feature_dir and not Path(args.wake_feature_dir).is_dir():
            raise ASRError("INVALID_CONFIGURATION", "check_config", "唤醒特征目录不存在")
    if target == "mic":
        modules.add("pyaudio")
    if target == "file":
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
        command.add_argument("target", nargs="?", default="mic", choices=TARGETS)
    options, overrides = parser.parse_known_args(argv)
    try:
        if options.action == "init":
            if overrides:
                parser.error("init 不接收识别参数")
            destination = (path or Path("config.toml")).resolve()
            with destination.open("x", encoding="utf-8") as target:
                target.write(template())
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
