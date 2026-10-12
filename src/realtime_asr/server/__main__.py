"""正式服务启动入口；一个 Uvicorn 进程拥有一个运行时。"""

import argparse
import os
import sys

from realtime_asr import ASRError, RecognizerConfig
from realtime_asr.cli.args import (
    add_candidate_model_args,
    add_realtime_model_args,
    add_wake_args,
    candidate_role,
    realtime_roles,
    wake_role,
)
from realtime_asr.server import ServerConfig, create_app


def parser():
    command = argparse.ArgumentParser(prog="realtime-asr serve", description="共享本地 ASR 服务")
    command.add_argument("--model", required=True)
    add_realtime_model_args(command)
    add_candidate_model_args(command)
    add_wake_args(command)
    command.add_argument("--backend", choices=("whisper", "nemotron"), default="whisper")
    command.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    command.add_argument("--cudnn-folder")
    command.add_argument("--cuda-directory")
    command.add_argument("--host", default="127.0.0.1")
    command.add_argument("--port", type=int, default=8000)
    command.add_argument("--token-env", default="REALTIME_ASR_TOKEN")
    command.add_argument("--ssl-certfile")
    command.add_argument("--ssl-keyfile")
    command.add_argument("--proxy-tls", action="store_true")
    command.add_argument("--trusted-proxy-ips")
    command.add_argument("--allowed-origin", action="append", default=[])
    command.add_argument("--vad-model")
    command.add_argument("--realtime", action=argparse.BooleanOptionalAction, default=True)
    command.add_argument("--acoustic-boundaries", action="store_true")
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


def main(argv=None):
    """仅启动器读取明确指定的 Token 环境变量；应用工厂不读隐藏配置。"""
    options = parser().parse_args(argv)
    args = vars(options).copy()
    try:
        recognizer = RecognizerConfig(
            model_path=args.pop("model"),
            backend=args.pop("backend"),
            device=args.pop("device"),
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
        args["realtime_config"] = realtime
        args["ultrafast_config"] = ultrafast
        args["token"] = os.environ.get(args.pop("token_env"))
        args["allowed_origins"] = tuple(args.pop("allowed_origin"))
        config = ServerConfig(
            recognizer=None if options.candidate_only else recognizer,
            candidate_config=candidate,
            final_enabled=not options.candidate_only,
            wake_config=wake,
            **args,
        )
        import uvicorn

        app = create_app(config)
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
