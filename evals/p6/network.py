"""真实端口验证夹具，只有 eval 显式启动监听和模型。"""

import socket
from contextlib import contextmanager
from threading import Thread
from time import monotonic, sleep

import httpx
import uvicorn

from realtime_asr.server import create_app


@contextmanager
def running_server(config, *, verify=True):
    """绑定随机回环端口，退出时请求 Uvicorn 正常关闭而非杀进程。"""
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(32)
        app = create_app(config)
        server = uvicorn.Server(
            uvicorn.Config(
                app,
                host="127.0.0.1",
                port=0,
                log_level="warning",
                access_log=False,
                ws_max_size=config.max_packet_bytes,
                ssl_certfile=config.ssl_certfile,
                ssl_keyfile=config.ssl_keyfile,
            )
        )
        thread = Thread(target=server.run, kwargs={"sockets": [listener]}, name="ASREvalServer")
        thread.start()
        scheme = "https" if config.ssl_certfile else "http"
        url = f"{scheme}://127.0.0.1:{listener.getsockname()[1]}"
        try:
            with httpx.Client(
                base_url=url,
                verify=verify,
                trust_env=False,
                headers={"Authorization": f"Bearer {config.token}"} if config.token else {},
                timeout=5,
            ) as http:
                deadline = monotonic() + 60
                while monotonic() < deadline:
                    if server.started:
                        ready = http.get("/api/v1/ready")
                        if ready.status_code == 200:
                            break
                        if ready.json().get("phase") == "failed":
                            raise AssertionError(ready.json())
                    sleep(0.05)
                else:
                    raise AssertionError("真实服务没有就绪")
            yield url
        finally:
            server.should_exit = True
            thread.join(60)
            if thread.is_alive():
                raise AssertionError("服务线程未完成关闭")
            if app.state.service.close_result["status"] != "ok":
                raise AssertionError(app.state.service.close_result)
