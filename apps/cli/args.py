"""命令语法与使用前可判断的配置，不创建模型或设备。"""

import argparse
import sys
from math import isfinite

from realtime_asr import ASRError


def _positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("必须为正整数")
    return number


def _seconds(value):
    number = float(value)
    if not isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("必须为有限非负秒数")
    return number


def _positive_seconds(value):
    number = _seconds(value)
    if number == 0:
        raise argparse.ArgumentTypeError("必须大于零")
    return number


def add_realtime_timing_args(parser):
    """本地和服务使用同一角色周期，默认值来自现有实时配置。"""
    parser.add_argument("--realtime-interval", type=_positive_seconds, default=0.2)
    parser.add_argument("--realtime-initial-delay", type=_seconds, default=0.2)


def add_segmentation_args(parser):
    """本地高级时间沿用分句默认；全部以秒为单位。"""
    for name, default in (
        ("silence-confirmation", 0.16),
        ("post-speech-silence", 0.6),
        ("min-recording-duration", 0.5),
        ("min-gap", 0),
        ("pre-roll-duration", 1),
    ):
        parser.add_argument("--" + name, type=_seconds, default=default)


def add_realtime_model_args(parser):
    """本地与服务启动器共用角色参数，外部语法只在 CLI 层维护。"""
    for role in ("realtime", "ultrafast"):
        parser.add_argument(f"--{role}-model")
        parser.add_argument(f"--{role}-backend", choices=("whisper", "nemotron"))
        parser.add_argument(f"--{role}-device", choices=("cpu", "cuda"))


def add_candidate_model_args(parser):
    """模型角色只由本地与服务宿主选择，远程客户端不接收模型路径。"""
    parser.add_argument("--candidates", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--candidate-model")
    parser.add_argument("--candidate-backend", choices=("whisper", "nemotron"))
    parser.add_argument("--candidate-device", choices=("cpu", "cuda"))
    parser.add_argument("--candidate-only", action=argparse.BooleanOptionalAction, default=False)


def add_candidate_request_args(parser):
    """本地与远程共用候选请求选项，不包含模型组装字段。"""
    parser.add_argument("--candidate-kind", choices=("tail", "whole"), default="whole")
    parser.add_argument("--candidate-every", type=_positive_seconds, default=2.0)
    parser.add_argument("--candidate-hint", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--completion-mode", choices=("final", "candidate"))
    parser.add_argument(
        "--retain-for-correction", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--correct-after-final", action=argparse.BooleanOptionalAction, default=False
    )


def add_wake_args(parser):
    """本地与服务宿主使用同一唤醒语法，密钥不通过命令行递交。"""
    parser.add_argument("--wake", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--wake-backend", choices=("porcupine", "openwakeword"), default="openwakeword"
    )
    parser.add_argument("--wake-keywords", nargs="+", default=[])
    parser.add_argument("--wake-models", nargs="+", default=[])
    parser.add_argument("--wake-parameter-model")
    parser.add_argument("--wake-feature-dir")
    parser.add_argument("--wake-sensitivity", type=float, default=0.5)
    parser.add_argument("--wake-activation-delay", type=_seconds, default=0)
    parser.add_argument("--wake-speech-timeout", type=_positive_seconds, default=5)
    parser.add_argument("--wake-trim-duration", type=_seconds, default=0.1)
    parser.add_argument("--wake-followup-window", type=_seconds, default=0)


def wake_role(args):
    """只组装配置；模型准备由自动会话或宿主预检负责。"""
    from pathlib import Path

    from realtime_asr import WakeConfig

    if not args.wake:
        return None
    return WakeConfig(
        backend=args.wake_backend,
        keywords=tuple(args.wake_keywords),
        model_paths=tuple(Path(path) for path in args.wake_models),
        model_path=Path(args.wake_parameter_model) if args.wake_parameter_model else None,
        feature_model_dir=Path(args.wake_feature_dir) if args.wake_feature_dir else None,
        sensitivity=args.wake_sensitivity,
        activation_delay=args.wake_activation_delay,
        speech_timeout=args.wake_speech_timeout,
        trim_duration=args.wake_trim_duration,
        followup_window=args.wake_followup_window,
    )


def _local_role_model(args, role, primary):
    """统一角色的路径、设备继承和 GPU 运行库复用，不加载模型。"""
    from realtime_asr import ASRError, RecognizerConfig

    path = getattr(args, role + "_model")
    backend = getattr(args, role + "_backend")
    device = getattr(args, role + "_device")
    if path is None:
        if backend is not None or device is not None:
            raise ASRError("INVALID_CONFIGURATION", "configure_cli", "角色后端或设备需要模型路径")
        return None
    selected_device = device or primary.device
    return RecognizerConfig(
        model=path,
        engine=backend or primary.engine,
        device=selected_device,
        device_index=primary.device_index if selected_device == "cuda" else 0,
        cudnn_folder=primary.cudnn_folder if selected_device == "cuda" else None,
        cuda_directory=primary.cuda_directory if selected_device == "cuda" else None,
    )


def candidate_role(args, primary):
    """候选专用时主路径属于候选；正常模式缺省共用最终引擎。"""
    from realtime_asr import CandidateConfig

    model = _local_role_model(args, "candidate", primary)
    if not (args.candidates or args.candidate_only or model is not None):
        return None
    if args.candidate_only and model is None:
        model = primary
    return CandidateConfig(model=model)


def realtime_roles(args, final):
    """外部角色参数转为 SDK 配置，模式只从声明能力选择。"""
    from realtime_asr import ASRError, RealtimeConfig

    roles = {}
    for role in ("realtime", "ultrafast"):
        path = getattr(args, role + "_model")
        model = _local_role_model(args, role, final)
        if path is not None and not args.realtime:
            raise ASRError(
                "INVALID_CONFIGURATION", "configure_cli", "角色选项需要模型路径并启用实时"
            )
        if role == "ultrafast" and path is None or not args.realtime:
            roles[role] = None
            continue
        selected = model if model is not None else final
        roles[role] = RealtimeConfig(
            mode=selected.capabilities.realtime_mode,
            model=model,
            acoustic_boundaries=args.acoustic_boundaries if role == "realtime" else False,
            interval=args.realtime_interval,
            initial_delay=args.realtime_initial_delay,
        )
    return roles["realtime"], roles["ultrafast"]


def parse_args(argv=None, *, diagnostic=False):
    """子命令分别声明源格式，避免麦克风设备与 ASR 设备混用。"""
    parser = argparse.ArgumentParser(prog="asr", description="本地实时音频识别")
    parser.add_argument(
        "--config", default=argparse.SUPPRESS, help="指定 config.toml，默认读取当前目录"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("config", help="创建、检查和显示本机配置")
    commands.add_parser("models", help="显式准备与校验模型，不启动识别")
    commands.add_parser("gpu-runtime", help="显式准备与校验 GPU 原生库")
    devices = commands.add_parser("devices", help="列出输入设备，不加载模型")
    devices.add_argument("--channels", type=_positive_int, default=1)
    devices.add_argument("--output", choices=("text", "jsonl"), default="text")
    for name in ("mic", "file", "stdin"):
        command = commands.add_parser(name)
        command.add_argument("--model", required=True)
        command.add_argument("--backend", choices=("whisper", "nemotron"), default="whisper")
        command.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
        command.add_argument("--device-index", type=int, default=0)
        command.add_argument("--cudnn-folder")
        command.add_argument("--cuda-directory")
        add_realtime_model_args(command)
        add_realtime_timing_args(command)
        add_candidate_model_args(command)
        add_candidate_request_args(command)
        add_wake_args(command)
        command.add_argument("--language")
        command.add_argument(
            "--mode", choices=("auto", "manual"), default="auto" if name == "mic" else "manual"
        )
        command.add_argument(
            "--realtime", action=argparse.BooleanOptionalAction, default=name == "mic"
        )
        command.add_argument("--output", choices=("text", "jsonl"), default="text")
        command.add_argument("--stable-text", action=argparse.BooleanOptionalAction, default=False)
        command.add_argument(
            "--dual-realtime", action=argparse.BooleanOptionalAction, default=False
        )
        command.add_argument(
            "--word-timestamps", action=argparse.BooleanOptionalAction, default=False
        )
        command.add_argument(
            "--acoustic-boundaries", action=argparse.BooleanOptionalAction, default=False
        )
        command.add_argument("--early-transcription-silence", type=_seconds, default=0)
        command.add_argument(
            "--dynamic-silence", action=argparse.BooleanOptionalAction, default=False
        )
        command.add_argument(
            "--punctuation-split", action=argparse.BooleanOptionalAction, default=False
        )
        command.add_argument("--max-audio-seconds", type=_positive_seconds, default=30)
        command.add_argument("--drain-timeout", type=_seconds, default=30)
        command.add_argument("--close-timeout", type=_seconds, default=5)
        command.add_argument("--vad-model")
        command.add_argument("--webrtc-mode", type=int, choices=range(4), default=3)
        command.add_argument("--silero-sensitivity", type=float, default=0.4)
        add_segmentation_args(command)
        command.add_argument(
            "--silero-end-detection", action=argparse.BooleanOptionalAction, default=False
        )
        command.add_argument(
            "--trim-pre-roll", action=argparse.BooleanOptionalAction, default=False
        )
        command.add_argument("--backdate-seconds", type=_seconds, default=0)
        if name in ("mic", "stdin"):
            command.add_argument(
                "--sample-rate", type=_positive_int, default=None if name == "mic" else 16000
            )
            command.add_argument("--channels", type=_positive_int, default=1)
        if name == "mic":
            command.add_argument("--input-device", type=int)
            command.add_argument("--chunk-frames", type=_positive_int, default=1024)
            command.add_argument("--queue-seconds", type=_positive_seconds, default=5)
            command.add_argument("--duration", type=_positive_seconds)
        elif name == "file":
            command.add_argument("paths", nargs="*" if diagnostic else "+")
        else:
            command.add_argument("--sample-type", choices=("int16", "float32"), default="int16")
    from apps.cli.config import configure, split_config

    selected = list(sys.argv[1:] if argv is None else argv)
    _, cleaned = split_config(selected)
    target = cleaned[0] if cleaned else ""
    try:
        invocation = configure(selected, target)
    except ASRError as exc:
        parser.error(exc.message)
    args = parser.parse_args(invocation.argv)
    args.launch_config = invocation
    if args.command != "devices":
        if args.wake and args.mode != "auto":
            parser.error("唤醒要求 --mode auto")
        args.candidates = args.candidates or args.candidate_model is not None
        args.completion_mode = args.completion_mode or (
            "candidate" if args.candidate_only else "final"
        )
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
    return args
