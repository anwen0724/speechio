"""Web 音色文件与描述信息；模型克隆流程仍由核心包负责。"""

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from fastapi import HTTPException
from fastapi.responses import FileResponse

from speechio import RemoteModelConfig, TTSError, Voice


def model_source(model):
    """记录音色使用环境；只用于服务端匹配，不向浏览器公开目录。"""
    if isinstance(model, RemoteModelConfig):
        return {
            "adapter": model.provider,
            "model": model.model,
            "base_url": (model.base_url or "https://dashscope.aliyuncs.com").rstrip("/"),
        }
    return {"adapter": model.adapter, "model_path": str(Path(model.model_path).resolve())}


class Voices:
    """保存核心 Voice 与应用名称；记录不会随临时音频过期。"""

    def __init__(self, directory):
        self.directory = Path(directory)

    def path(self, voice_id, suffix=".voice"):
        if not re.fullmatch(r"[0-9a-f]{32}", voice_id):
            raise HTTPException(404, "音色不存在")
        return self.directory / f"{voice_id}{suffix}"

    def record(self, voice_id):
        try:
            return json.loads(self.path(voice_id, ".json").read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise HTTPException(404, "音色不存在") from None

    @staticmethod
    def public(record):
        return {key: value for key, value in record.items() if key != "source"}

    def save(self, voice, *, name, model_key, model):
        """只在克隆或导入成功后保存；失败时撤掉本次未完成的记录。"""
        self.directory.mkdir(parents=True, exist_ok=True)
        voice_id = uuid4().hex
        record = {
            "id": voice_id,
            "name": name,
            "model_key": model_key,
            "adapter": voice.adapter,
            "model": voice.model,
            "created_at": datetime.now(UTC).isoformat(),
            "source": model_source(model),
        }
        try:
            voice.save(self.path(voice_id))
            self.path(voice_id, ".json").write_text(
                json.dumps(record, ensure_ascii=False), encoding="utf-8"
            )
        except BaseException:
            self.path(voice_id).unlink(missing_ok=True)
            self.path(voice_id, ".json").unlink(missing_ok=True)
            raise
        return self.public(record)

    def import_file(self, path, *, name, model_key, model):
        """验证可见的模型身份；完整数据的兼容性仍由核心适配器检查。"""
        voice = Voice.load(path)
        source = model_source(model)
        if voice.adapter != source["adapter"] or (
            isinstance(model, RemoteModelConfig) and voice.model != model.model
        ):
            raise TTSError("VOICE_INCOMPATIBLE", "import_voice", "音色不属于所选模型")
        return self.save(voice, name=name, model_key=model_key, model=model)

    def list(self, model):
        source = model_source(model)
        records = [
            json.loads(path.read_text(encoding="utf-8")) for path in self.directory.glob("*.json")
        ]
        return [
            self.public(record)
            for record in sorted(records, key=lambda item: item["created_at"], reverse=True)
            if record["source"] == source
        ]

    def load(self, voice_id, model):
        if self.record(voice_id)["source"] != model_source(model):
            raise TTSError("VOICE_INCOMPATIBLE", "load_voice", "所选音色不属于当前模型或服务地址")
        return Voice.load(self.path(voice_id))

    def download(self, voice_id):
        self.record(voice_id)
        return FileResponse(
            self.path(voice_id), media_type="application/octet-stream", filename=f"{voice_id}.voice"
        )

    def remove(self, voice_id):
        """只移除应用的本地记录，不请求删除服务商的远端音色。"""
        self.record(voice_id)
        self.path(voice_id).unlink(missing_ok=True)
        self.path(voice_id, ".json").unlink()
