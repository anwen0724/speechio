"""正式服务启动入口；一个 Uvicorn 进程拥有一个运行时。"""

import argparse
import logging
import os
import sys

from realtime_asr import ASRError, RecognizerConfig
from realtime_asr.cli.args import (
    add_candidate_model_args,
    add_realtime_model_args,
    add_realtime_timing_args,
    add_wake_args,
    candidate_role,
    realtime_roles,
    wake_role,
)
from realtime_asr.server import ServerConfig, create_app


def parser():
    command = argparse.ArgumentParser(prog="asr serve", description="共享本地 ASR 服务")
    command.add_argument(
        "--config", default=argparse.SUPPRESS, help="指定 config.toml，默认读取当前目录"
    )
    command.add_argument("--model", required=True)
    add_realtime_model_args(command)
    add_realtime_timing_args(command)
    add_candidate_model_args(command)
    add_wake_args(command)
    command.add_argument("--backend", choices=("whisper", "nemotron"), default="whisper")
    command.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    command.add_argument("--device-index", type=int, default=0)
    command.add_argument("--cudnn-folder")
    command.add_argument("--cuda-directory")
    command.add_argument("--host", default="127.0.0.1")
    command.add_argument("--port", type=int, default=8000)
    command.add_argument("--token-env", default="REALTIME_ASR_TOKEN")
    command.add_argument("--ssl-certfile")
    command.add_argument("--ssl-keyfile")
    command.add_argument("--proxy-tls", action=argparse.BooleanOptionalAction, default=False)
    command.add_argument("--trusted-proxy-ips")
    command.add_argument("--allowed-origin", action="append", default=[])
    command.add_argument("--vad-model")
    command.add_argument("--realtime", action=argparse.BooleanOptionalAction, default=True)
    command.add_argument(
        "--acoustic-boundaries", action=argparse.BooleanOptionalAction, default=False
    )
    for name in (
        "max-sessions",
        "max-tasks",
        "max-task-audio-samples",
        "max-pending-units",
        "max-http-bytes",
        "max-packet-bytes",
        "max-metadata-bytes",
        "max-pending-messages",
    ):
        default = ServerConfig.__dataclass_fields__[name.replace("-", "_")].default
        command.add_argument("--" + name, type=int, default=default)
    for name in (
        "max-audio-seconds",
        "open-timeout",
        "idle-timeout",
        "body-timeout",
        "drain-timeout",
        "send-timeout",
        "close-timeout",
    ):
        default = ServerConfig.__dataclass_fields__[name.replace("-", "_")].default
        command.add_argument("--" + name, type=float, default=default)
    return command


def parse_options(argv=None):
    """配置只形成实际参数；此步骤不监听或准备模型。"""
    from realtime_asr.cli.config import configure

    command = parser()
    try:
        invocation = configure(list(sys.argv[1:] if argv is None else argv), "serve")
    except ASRError as exc:
        command.error(exc.message)
    args = command.parse_args(invocation.argv)
    args.launch_config = invocation
    return args


def build_config(options):
    """启动和静态检查共用宿主组装，避免配置检查维护另一套规则。"""
    args = vars(options).copy()
    args.pop("launch_config")
    recognizer = RecognizerConfig(
        model=args.pop("model"),
        engine=args.pop("backend"),
        device=args.pop("device"),
        device_index=args.pop("device_index"),
        cudnn_folder=args.pop("cudnn_folder"),
        cuda_directory=args.pop("cuda_directory"),
    )
    realtime, ultrafast = realtime_roles(options, recognizer)
    candidate = candidate_role(options, recognizer)
    wake = wake_role(options)
    for name in tuple(args):
        if name == "wake" or name.startswith("wake_"):
            args.pop(name)
    for role in ("realtime", "ultrafast"):
        for field in ("model", "backend", "device"):
            args.pop(role + "_" + field)
    for name in (
        "candidates",
        "candidate_model",
        "candidate_backend",
        "candidate_device",
        "candidate_only",
    ):
        args.pop(name)
    args.pop("acoustic_boundaries")
    args.pop("realtime_interval")
    args.pop("realtime_initial_delay")
    args["realtime_config"] = realtime
    args["ultrafast_config"] = ultrafast
    args["token"] = os.environ.get(args.pop("token_env"))
    args["allowed_origins"] = tuple(args.pop("allowed_origin"))
    return ServerConfig(
        recognizer=None if options.candidate_only else recognizer,
        candidate_config=candidate,
        final_enabled=not options.candidate_only,
        wake_config=wake,
        **args,
    )


def main(argv=None):
    """启动器读取环境凭据；库和核心不读取隐藏配置。"""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    try:
        config = build_config(parse_options(argv))
        import uvicorn

        app = create_app(config)
        logger = logging.getLogger("realtime_asr.server.startup")
        handler = logging.StreamHandler(sys.stderr)
        logger.addHandler(handler)
        previous_level = logger.level
        logger.setLevel(logging.INFO)
        print("[preparing] 正在加载模型，请等待就绪提示。", file=sys.stderr, flush=True)
        try:
            uvicorn.run(
                app,
                host=config.host,
                port=config.port,
                workers=1,
                ssl_certfile=config.ssl_certfile,
                ssl_keyfile=config.ssl_keyfile,
                proxy_headers=config.proxy_tls,
                forwarded_allow_ips=config.trusted_proxy_ips or "127.0.0.1",
                ws_max_size=config.max_packet_bytes,
                ws_max_queue=16,
                timeout_graceful_shutdown=config.close_timeout,
                log_level="warning",
                access_log=False,
            )
        finally:
            logger.removeHandler(handler)
            logger.setLevel(previous_level)
        result = app.state.service.close_result
        if result is not None and result["status"] != "ok":
            print(f"[SERVICE_CLOSE_{result['status'].upper()}] {result}", file=sys.stderr)
            return 4
        if app.state.service.startup_error is not None:
            print(f"[SERVICE_PREPARE_FAILED] {app.state.service.startup_error}", file=sys.stderr)
            return 1
        return 0
    except ASRError as exc:
        print(f"[{exc.code}] {exc.message}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
