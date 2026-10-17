"""命令行选项与命令声明；argparse、配置读取和配置模板都从这里生成。"""

from dataclasses import dataclass, field
from pathlib import Path

LOCAL = ("mic", "file", "stdin")
TARGETS = (*LOCAL, "devices")
ENGINES = ("whisper", "nemotron")
DEVICES = ("cpu", "cuda")


@dataclass(frozen=True)
class Command:
    """输入命令及其与通用默认值不同的默认项。"""

    name: str
    help: str
    defaults: dict = field(default_factory=dict)


COMMANDS = (
    Command("devices", "列出输入设备，不加载模型"),
    Command("mic", "麦克风连续识别", {"mode": "auto", "realtime": True}),
    Command("file", "文件识别，按顺序处理", {"mode": "manual", "realtime": False}),
    Command("stdin", "原始管道输入识别", {"mode": "manual", "realtime": False}),
)


@dataclass(frozen=True)
class Option:
    """一个选项在命令行和配置文件中的完整声明。

    kind 决定取值校验和配置渲染方式：str、int、float、bool、model、path、
    strlist、pathlist。model 只把看起来像路径的值按配置文件目录解析，
    其余按模型名原样交给引擎。
    """

    flag: str
    kind: str
    section: str
    key: str | None = None
    default: object = None
    choices: tuple = ()
    required: bool = False
    commands: tuple = LOCAL
    help: str = ""
    comment: str = ""
    sample: str | None = None
    active_sample: bool = False
    minimum: float | None = None
    exclusive: bool = False
    nargs: str | None = None

    @property
    def dest(self):
        return self.flag.replace("-", "_")

    @property
    def config_key(self):
        return self.key or self.dest

    @property
    def config_flag(self):
        return "--" + self.flag


SECTION_COMMENTS = {
    "model": "模型与计算设备",
    "recognition": "识别语言与模式",
    "microphone": "麦克风采集",
    "input": "标准输入格式",
    "realtime": "实时识别",
    "ultrafast": "双实时的极速角色",
    "vad": "语音检测",
    "segmentation": "自动分句，单位秒",
    "endpoint": "结束优化",
    "candidate": "候选与同轮继续",
    "wake": "唤醒词",
    "output": "输出格式",
    "limits": "容量与超时",
}

# 声明顺序同时决定命令行帮助和配置模板的分组顺序。
OPTIONS = (
    Option(
        "model",
        "model",
        "model",
        required=True,
        help="模型名或已准备的目录",
        comment="模型名或目录；相对路径以本文件目录为基准",
        sample='".runtime/models/whisper-tiny"',
        active_sample=True,
    ),
    Option(
        "engine",
        "str",
        "model",
        choices=ENGINES,
        default="whisper",
        help="识别引擎",
        comment="whisper 或 nemotron",
    ),
    Option(
        "device",
        "str",
        "model",
        choices=DEVICES,
        default="cpu",
        help="模型计算设备",
        comment="需要本机 GPU 时改为 cuda",
    ),
    Option("device-index", "int", "model", default=0, help="CUDA 设备索引"),
    Option(
        "language",
        "str",
        "recognition",
        help="识别语言",
        comment="省略时由引擎判断",
        sample='"zh"',
    ),
    Option(
        "mode",
        "str",
        "recognition",
        choices=("auto", "manual"),
        help="输入模式",
        comment="auto 或 manual；不填写则保留各命令默认值",
        sample='"manual"',
    ),
    Option(
        "word-timestamps",
        "bool",
        "recognition",
        default=False,
        help="返回词时间",
    ),
    Option(
        "input-device",
        "int",
        "microphone",
        key="device_index",
        commands=("mic",),
        help="麦克风设备索引",
        comment="通过 asr devices 查询；省略时使用系统默认设备",
        sample="0",
    ),
    Option(
        "sample-rate",
        "int",
        "microphone",
        commands=("mic",),
        minimum=0,
        exclusive=True,
        help="采集采样率",
        comment="省略时由采集器选择设备支持的格式",
        sample="16000",
    ),
    Option(
        "channels",
        "int",
        "microphone",
        default=1,
        commands=("mic", "devices"),
        minimum=0,
        exclusive=True,
        help="采集声道数",
    ),
    Option(
        "chunk-frames",
        "int",
        "microphone",
        default=1024,
        commands=("mic",),
        minimum=0,
        exclusive=True,
        help="每次读取的源帧数",
    ),
    Option(
        "queue-seconds",
        "float",
        "microphone",
        default=5,
        commands=("mic",),
        minimum=0,
        exclusive=True,
        help="采集缓冲预算，单位秒",
    ),
    Option(
        "duration",
        "float",
        "microphone",
        commands=("mic",),
        minimum=0,
        exclusive=True,
        help="采集时长，单位秒",
        comment="省略时持续采集，Ctrl+C 停止",
        sample="20.0",
    ),
    Option(
        "sample-rate",
        "int",
        "input",
        default=16000,
        commands=("stdin",),
        minimum=0,
        exclusive=True,
        help="原始音频采样率",
    ),
    Option(
        "channels",
        "int",
        "input",
        default=1,
        commands=("stdin",),
        minimum=0,
        exclusive=True,
        help="原始音频声道数",
    ),
    Option(
        "sample-type",
        "str",
        "input",
        choices=("int16", "float32"),
        default="int16",
        commands=("stdin",),
        help="原始音频样本类型",
    ),
    Option(
        "realtime",
        "bool",
        "realtime",
        key="enabled",
        help="启用实时识别",
        comment="不填写则保留各命令默认值",
        sample="true",
    ),
    Option(
        "realtime-interval",
        "float",
        "realtime",
        key="interval",
        default=0.2,
        minimum=0,
        exclusive=True,
        help="实时更新间隔，单位秒",
    ),
    Option(
        "realtime-initial-delay",
        "float",
        "realtime",
        key="initial_delay",
        default=0.2,
        minimum=0,
        help="首个实时结果前的等待，单位秒",
    ),
    Option(
        "realtime-model",
        "model",
        "realtime",
        key="model",
        help="独立准确角色模型",
        comment="省略时沿用最终配置",
        sample='".runtime/models/nemotron"',
    ),
    Option(
        "realtime-engine",
        "str",
        "realtime",
        key="engine",
        choices=ENGINES,
        help="独立准确角色引擎",
        sample='"nemotron"',
    ),
    Option(
        "realtime-device",
        "str",
        "realtime",
        key="device",
        choices=DEVICES,
        help="独立准确角色设备",
        sample='"cpu"',
    ),
    Option("stable-text", "bool", "realtime", default=False, help="输出稳定文本"),
    Option("dual-realtime", "bool", "realtime", default=False, help="启用双实时"),
    Option(
        "acoustic-boundaries",
        "bool",
        "realtime",
        default=False,
        help="Whisper 快照的声学触发",
    ),
    Option(
        "ultrafast-model",
        "model",
        "ultrafast",
        key="model",
        help="极速角色模型",
        comment="双实时使用独立极速模型",
        sample='".runtime/models/nemotron-80ms"',
    ),
    Option(
        "ultrafast-engine",
        "str",
        "ultrafast",
        key="engine",
        choices=ENGINES,
        help="极速角色引擎",
        sample='"nemotron"',
    ),
    Option(
        "ultrafast-device",
        "str",
        "ultrafast",
        key="device",
        choices=DEVICES,
        help="极速角色设备",
        sample='"cpu"',
    ),
    Option(
        "vad-model",
        "path",
        "vad",
        key="model",
        help="VAD 模型文件",
        comment="自动分句使用",
        sample='".runtime/models/vad/silero-v6.onnx"',
        active_sample=True,
    ),
    Option(
        "webrtc-mode",
        "int",
        "vad",
        default=3,
        choices=range(4),
        help="WebRTC 检测模式",
        comment="0 到 3，越大越保守",
    ),
    Option(
        "silero-sensitivity",
        "float",
        "vad",
        default=0.4,
        help="Silero 检出灵敏度",
    ),
    Option(
        "silero-end-detection",
        "bool",
        "vad",
        default=False,
        help="用 Silero 判断句尾",
    ),
    Option(
        "silence-confirmation",
        "float",
        "segmentation",
        default=0.16,
        minimum=0,
        help="静音确认时长",
    ),
    Option(
        "post-speech-silence",
        "float",
        "segmentation",
        default=0.6,
        minimum=0,
        help="句尾静音时长",
    ),
    Option(
        "min-recording-duration",
        "float",
        "segmentation",
        default=0.5,
        minimum=0,
        help="最短录音时长",
    ),
    Option("min-gap", "float", "segmentation", default=0, minimum=0, help="段间最短间隔"),
    Option(
        "pre-roll-duration",
        "float",
        "segmentation",
        default=1,
        minimum=0,
        help="前置缓冲时长",
    ),
    Option("trim-pre-roll", "bool", "segmentation", default=False, help="裁掉前置缓冲"),
    Option(
        "backdate-seconds",
        "float",
        "segmentation",
        key="backdate_duration",
        default=0,
        minimum=0,
        help="回溯时长",
    ),
    Option(
        "early-transcription-silence",
        "float",
        "endpoint",
        default=0,
        minimum=0,
        help="提前推理等待，单位秒",
    ),
    Option("dynamic-silence", "bool", "endpoint", default=False, help="动态等待"),
    Option("punctuation-split", "bool", "endpoint", default=False, help="按句末标点切段"),
    Option("candidates", "bool", "candidate", key="enabled", default=False, help="启用候选"),
    Option(
        "candidate-model",
        "model",
        "candidate",
        key="model",
        help="候选角色模型",
        comment="省略时共享主模型",
        sample='".runtime/models/whisper-tiny"',
    ),
    Option(
        "candidate-engine",
        "str",
        "candidate",
        key="engine",
        choices=ENGINES,
        help="候选角色引擎",
        sample='"whisper"',
    ),
    Option(
        "candidate-device",
        "str",
        "candidate",
        key="device",
        choices=DEVICES,
        help="候选角色设备",
        sample='"cpu"',
    ),
    Option("candidate-only", "bool", "candidate", key="only", default=False, help="只用候选完成"),
    Option(
        "candidate-kind",
        "str",
        "candidate",
        key="kind",
        default="whole",
        choices=("tail", "whole"),
        help="候选范围",
    ),
    Option(
        "candidate-every",
        "float",
        "candidate",
        key="every",
        default=2.0,
        minimum=0,
        exclusive=True,
        help="候选请求周期，单位秒",
    ),
    Option(
        "candidate-hint",
        "bool",
        "candidate",
        key="hint",
        default=False,
        help="候选携带文本提示",
    ),
    Option(
        "completion-mode",
        "str",
        "candidate",
        choices=("final", "candidate"),
        help="完成策略",
        sample='"final"',
    ),
    Option(
        "retain-for-correction",
        "bool",
        "candidate",
        default=False,
        help="保留最近单元用于修正",
    ),
    Option(
        "correct-after-final",
        "bool",
        "candidate",
        default=False,
        help="结束后请求修正",
    ),
    Option("wake", "bool", "wake", key="enabled", default=False, help="启用唤醒"),
    Option(
        "wake-backend",
        "str",
        "wake",
        key="backend",
        default="openwakeword",
        choices=("porcupine", "openwakeword"),
        help="唤醒后端",
    ),
    Option(
        "wake-keywords",
        "strlist",
        "wake",
        key="keywords",
        default=(),
        help="Porcupine 内置关键词",
        sample='["porcupine"]',
    ),
    Option(
        "wake-models",
        "pathlist",
        "wake",
        key="model_paths",
        default=(),
        help="唤醒模型文件",
        sample='[".runtime/models/wake/alexa_v0.1.onnx"]',
    ),
    Option(
        "wake-parameter-model",
        "path",
        "wake",
        key="model",
        help="唤醒参数文件",
        sample='"models/porcupine-language.pv"',
    ),
    Option(
        "wake-feature-dir",
        "path",
        "wake",
        key="feature_model_dir",
        help="OpenWakeWord 特征目录",
        sample='".runtime/models/wake"',
    ),
    Option(
        "wake-sensitivity",
        "float",
        "wake",
        key="sensitivity",
        default=0.5,
        help="唤醒灵敏度",
    ),
    Option(
        "wake-activation-delay",
        "float",
        "wake",
        key="activation_delay",
        default=0,
        minimum=0,
        help="激活宽限，单位秒",
    ),
    Option(
        "wake-speech-timeout",
        "float",
        "wake",
        key="speech_timeout",
        default=5,
        minimum=0,
        exclusive=True,
        help="正文等待上限，单位秒",
    ),
    Option(
        "wake-trim-duration",
        "float",
        "wake",
        key="trim_duration",
        default=0.1,
        minimum=0,
        help="唤醒词裁剪时长，单位秒",
    ),
    Option(
        "wake-followup-window",
        "float",
        "wake",
        key="followup_window",
        default=0,
        minimum=0,
        help="继续窗口，单位秒",
    ),
    Option(
        "output",
        "str",
        "output",
        key="format",
        default="text",
        choices=("text", "jsonl"),
        commands=TARGETS,
        help="输出格式",
        comment="text 或 jsonl；不自动保存正文",
    ),
    Option(
        "max-audio-seconds",
        "float",
        "limits",
        default=30,
        minimum=0,
        exclusive=True,
        help="单段音频上限，单位秒",
    ),
    Option(
        "drain-timeout",
        "float",
        "limits",
        default=30,
        minimum=0,
        help="等待最终结果超时，单位秒",
    ),
    Option(
        "close-timeout",
        "float",
        "limits",
        default=5,
        minimum=0,
        help="关闭资源超时，单位秒",
    ),
)


def wake_role(args):
    """只组装唤醒配置；模型准备由自动会话负责。"""
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
    from realtime_asr import ASRError, ModelConfig

    path = getattr(args, role + "_model")
    engine = getattr(args, role + "_engine")
    device = getattr(args, role + "_device")
    if path is None:
        if engine is not None or device is not None:
            raise ASRError("INVALID_CONFIGURATION", "configure_cli", "角色引擎或设备需要模型路径")
        return None
    selected_device = device or primary.device
    return ModelConfig(
        model=path,
        engine=engine or primary.engine,
        device=selected_device,
        device_index=primary.device_index if selected_device == "cuda" else 0,
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
