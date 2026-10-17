"""由选项声明生成 argparse，并执行跨参数校验。"""

import argparse
import sys
from math import isfinite

from apps.cli.options import COMMANDS, OPTIONS
from realtime_asr import ASRError


def build_parser(*, diagnostic=False):
    """按命令声明组装解析器；diagnostic 允许文件命令不带路径。"""
    parser = argparse.ArgumentParser(prog="asr", description="本地实时音频识别")
    parser.add_argument(
        "--config", default=argparse.SUPPRESS, help="指定 config.toml，默认读取当前目录"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("config", help="创建、检查和显示本机配置")
    commands.add_parser("models", help="显式准备与校验模型，不启动识别")
    commands.add_parser("gpu-runtime", help="显式准备与校验 GPU 原生库")
    for command in COMMANDS:
        sub = commands.add_parser(command.name, help=command.help)
        if command.name == "file":
            sub.add_argument("paths", nargs="*" if diagnostic else "+")
        for option in _options_for(command.name):
            _add_option(sub, option, command.defaults.get(option.dest, option.default))
    return parser


def _options_for(name):
    """同一命令只添加一次同名选项；配置分组之间允许重复声明。"""
    seen, result = set(), []
    for option in OPTIONS:
        if name in option.commands and option.flag not in seen:
            seen.add(option.flag)
            result.append(option)
    return result


def _add_option(parser, option, default):
    fields = {"dest": option.dest, "default": default, "help": option.help}
    if option.required:
        fields["required"] = True
    if option.choices:
        fields["choices"] = option.choices
    if option.kind == "bool":
        fields["action"] = argparse.BooleanOptionalAction
    elif option.kind in ("strlist", "pathlist"):
        fields["nargs"] = option.nargs or "+"
        fields["metavar"] = option.flag.upper().replace("-", "_")
    elif option.kind in ("int", "float"):
        fields["type"] = _number(option)
    parser.add_argument(option.config_flag, **fields)


def _number(option):
    """数值转换同时执行声明中的下限；配置文件的值也经过同一入口。"""

    def convert(text):
        try:
            value = int(text) if option.kind == "int" else float(text)
        except ValueError:
            raise argparse.ArgumentTypeError("需要数值") from None
        if option.kind == "float" and not isfinite(value):
            raise argparse.ArgumentTypeError("必须为有限数值")
        if option.minimum is not None:
            if option.exclusive and value <= option.minimum:
                raise argparse.ArgumentTypeError(f"必须大于 {option.minimum:g}")
            if not option.exclusive and value < option.minimum:
                raise argparse.ArgumentTypeError(f"不能小于 {option.minimum:g}")
        return value

    return convert


def parse_args(argv=None, *, diagnostic=False):
    """读取配置并解析命令；配置错误统一转为参数错误。"""
    from apps.cli.settings import configure, split_config

    parser = build_parser(diagnostic=diagnostic)
    selected = list(sys.argv[1:] if argv is None else argv)
    _, cleaned = split_config(selected)
    try:
        invocation = configure(selected, cleaned[0] if cleaned else "")
    except ASRError as exc:
        parser.error(exc.message)
    args = parser.parse_args(invocation.argv)
    args.launch_config = invocation
    _validate(parser, args)
    return args


def _validate(parser, args):
    """只检查跨参数关系；取值本身已由声明和 argparse 校验。"""
    if args.command == "devices":
        return
    if args.wake and args.mode != "auto":
        parser.error("唤醒要求 --mode auto")
    args.candidates = args.candidates or args.candidate_model is not None
    args.completion_mode = args.completion_mode or ("candidate" if args.candidate_only else "final")
    if args.candidate_only and args.completion_mode != "candidate":
        parser.error("候选专用模式要求候选完成策略")
    if args.correct_after_final:
        args.retain_for_correction = True
        if args.mode != "manual":
            parser.error("结束后修正只支持手动输入")
    if (args.completion_mode == "candidate" or args.retain_for_correction) and not (
        args.candidates or args.candidate_model or args.candidate_only
    ):
        parser.error("候选完成或修正要求配置候选角色")
    if args.dual_realtime and (not args.realtime or args.ultrafast_model is None):
        parser.error("双实时要求 --realtime 和 --ultrafast-model")
    if args.mode != "auto" and (
        args.early_transcription_silence or args.dynamic_silence or args.punctuation_split
    ):
        parser.error("结束优化要求 --mode auto")
    if (
        args.stable_text
        or args.dynamic_silence
        or args.acoustic_boundaries
        or args.punctuation_split
    ) and not args.realtime:
        parser.error("文本或声学实时策略要求 --realtime")
    if args.punctuation_split and not args.stable_text:
        parser.error("句末切段要求 --stable-text")
    if args.mode == "auto" and not args.vad_model:
        parser.error("auto 模式要求 --vad-model")
    if args.max_audio_seconds > 600 or int(args.max_audio_seconds * 16000) == 0:
        parser.error("--max-audio-seconds 应大于零且不超过 600 秒")
    if args.command == "mic" and args.input_device is not None and args.input_device < 0:
        parser.error("--input-device 应为非负索引")
