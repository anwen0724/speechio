"""命令语法与使用前可判断的配置，不创建模型或设备。"""

import argparse
from math import isfinite


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


def add_realtime_model_args(parser):
    """本地与服务启动器共用角色参数，外部语法只在 CLI 层维护。"""
    for role in ("realtime", "ultrafast"):
        parser.add_argument(f"--{role}-model")
        parser.add_argument(f"--{role}-backend", choices=("whisper", "nemotron"))
        parser.add_argument(f"--{role}-device", choices=("cpu", "cuda"))


def realtime_roles(args, final):
    """外部角色参数转为 SDK 配置，模式只从声明能力选择。"""
    from realtime_asr import ASRError, RealtimeConfig, RecognizerConfig

    roles = {}
    for role in ("realtime", "ultrafast"):
        path = getattr(args, role + "_model")
        backend = getattr(args, role + "_backend")
        device = getattr(args, role + "_device")
        if (path is None and (backend is not None or device is not None)) or (
            path is not None and not args.realtime
        ):
            raise ASRError(
                "INVALID_CONFIGURATION", "configure_cli", "角色选项需要模型路径并启用实时"
            )
        if role == "ultrafast" and path is None or not args.realtime:
            roles[role] = None
            continue
        model = None
        selected = final
        if path is not None:
            selected_device = device or final.device
            selected = RecognizerConfig(
                model_path=path,
                backend=backend or final.backend,
                device=selected_device,
                cudnn_folder=final.cudnn_folder if selected_device == "cuda" else None,
                cuda_directory=final.cuda_directory if selected_device == "cuda" else None,
            )
            model = selected
        roles[role] = RealtimeConfig(
            mode=selected.capabilities.realtime_mode,
            model=model,
            acoustic_boundaries=args.acoustic_boundaries if role == "realtime" else False,
        )
    return roles["realtime"], roles["ultrafast"]


def parse_args(argv=None):
    """子命令分别声明源格式，避免麦克风设备与 ASR 设备混用。"""
    parser = argparse.ArgumentParser(prog="realtime-asr", description="本地实时音频识别")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("serve", help="运行共享 HTTP / WebSocket 服务（需要 server 依赖）")
    commands.add_parser("remote", help="接入远程服务（需要 client 依赖）")
    devices = commands.add_parser("devices", help="列出输入设备，不加载模型")
    devices.add_argument("--channels", type=_positive_int, default=1)
    devices.add_argument("--output", choices=("text", "jsonl"), default="text")
    for name in ("mic", "file", "stdin"):
        command = commands.add_parser(name)
        command.add_argument("--model", required=True)
        command.add_argument("--backend", choices=("whisper", "nemotron"), default="whisper")
        command.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
        command.add_argument("--cudnn-folder")
        command.add_argument("--cuda-directory")
        add_realtime_model_args(command)
        command.add_argument("--language")
        command.add_argument(
            "--mode", choices=("auto", "manual"), default="auto" if name == "mic" else "manual"
        )
        command.add_argument(
            "--realtime", action=argparse.BooleanOptionalAction, default=name == "mic"
        )
        command.add_argument("--output", choices=("text", "jsonl"), default="text")
        command.add_argument("--stable-text", action="store_true")
        command.add_argument("--dual-realtime", action="store_true")
        command.add_argument("--word-timestamps", action="store_true")
        command.add_argument("--acoustic-boundaries", action="store_true")
        command.add_argument("--early-transcription-silence", type=_seconds, default=0)
        command.add_argument("--dynamic-silence", action="store_true")
        command.add_argument("--punctuation-split", action="store_true")
        command.add_argument("--max-audio-seconds", type=_positive_seconds, default=30)
        command.add_argument("--drain-timeout", type=_seconds, default=30)
        command.add_argument("--close-timeout", type=_seconds, default=5)
        command.add_argument("--vad-model")
        command.add_argument("--webrtc-mode", type=int, choices=range(4), default=3)
        command.add_argument("--silero-end-detection", action="store_true")
        command.add_argument("--trim-pre-roll", action="store_true")
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
            command.add_argument("paths", nargs="+")
        else:
            command.add_argument("--sample-type", choices=("int16", "float32"), default="int16")
    args = parser.parse_args(argv)
    if args.command != "devices":
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
