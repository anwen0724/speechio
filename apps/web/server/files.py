"""有界文件上传：客户端持有随机令牌，不选择服务器文件路径。"""

import re
from time import monotonic
from uuid import uuid4

from fastapi import HTTPException


class Uploads:
    """跟踪本进程创建的文件；消费、失败和过期时删除。"""

    def __init__(self, settings):
        self.settings = settings
        self.pending = {}
        self.receiving = False

    def expire(self, *, all_files=False):
        for token, (path, created) in list(self.pending.items()):
            if all_files or monotonic() - created > 300:
                path.unlink(missing_ok=True)
                del self.pending[token]

    async def receive(self, request):
        """逐块限制上传体积，超限立即删除已经写入的部分。"""
        self.expire()
        if self.receiving or len(self.pending) >= 4:
            raise HTTPException(409, "还有未处理的上传，请先处理或删除")
        # 仅接收扩展名，服务器始终生成文件名，不能让客户端指定路径。
        extension = request.headers.get("X-Audio-Extension", ".audio").lower()
        if not re.fullmatch(r"\.[a-z0-9]{1,10}", extension):
            raise HTTPException(400, "音频扩展名无效")
        self.receiving = True
        token = uuid4().hex
        directory = self.settings.upload_dir
        path = directory / f"{token}{extension}"
        try:
            directory.mkdir(parents=True, exist_ok=True)
            size = 0
            with path.open("xb") as stream:
                async for chunk in request.stream():
                    size += len(chunk)
                    if size > self.settings.upload_limit_bytes:
                        raise HTTPException(413, "文件超过本机设置的上传大小限制")
                    stream.write(chunk)
            if not size:
                raise HTTPException(400, "音频文件为空")
            self.pending[token] = (path, monotonic())
            return {"token": token, "bytes": size}
        except BaseException:
            path.unlink(missing_ok=True)
            raise
        finally:
            self.receiving = False

    def take(self, token):
        """令牌只能领取一次；领取后由运行任务负责清理。"""
        self.expire()
        record = self.pending.pop(token, None)
        if record is None:
            raise ValueError("上传不存在或已过期，请重新选择文件")
        return record[0]

    def discard(self, token):
        record = self.pending.pop(token, None)
        if record:
            record[0].unlink(missing_ok=True)
