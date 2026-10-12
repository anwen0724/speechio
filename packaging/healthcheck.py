"""镜像就绪探针：校验 TLS 与鉴权，不导入模型运行库。"""

import os
import ssl
from urllib.request import Request, urlopen


def main():
    url = os.environ.get("ASR_HEALTH_URL", "https://127.0.0.1:8000/api/v1/ready")
    context = ssl.create_default_context(cafile=os.environ.get("ASR_HEALTH_CA"))
    request = Request(url)
    token = os.environ.get("REALTIME_ASR_TOKEN")
    if token:
        request.add_header("Authorization", "Bearer " + token)
    try:
        with urlopen(request, context=context, timeout=4) as response:
            return 0 if response.status == 200 else 1
    except (OSError, ValueError):
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
