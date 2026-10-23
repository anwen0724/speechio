"""Web 音色文件与描述信息；模型克隆流程仍由核心包负责。"""

import json
import os
import re
import string
import tempfile
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
        self.directory = Path(directory).resolve()

    @property
    def save_dir(self):
        """应用记住上次成功保存的目录，配置目录仍用于存放音色索引。"""
        try:
            return json.loads((self.directory / ".preferences.json").read_text(encoding="utf-8"))[
                "save_dir"
            ]
        except FileNotFoundError:
            return str(self.directory)

    def prepare_directory(self, directory=None):
        """在模型调用前确认目标可写，避免克隆完成后才发现路径无效。"""
        path = Path(directory or self.save_dir).expanduser()
        if not path.is_absolute():
            raise ValueError("请填写保存目录的完整路径")
        try:
            path.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryFile(dir=path):
                pass
        except OSError as exc:
            raise ValueError("保存目录不可写，请重新选择") from exc
        return path.resolve()

    def directories(self, directory=None):
        """供本机应用选择保存位置；只浏览目录，不读取其中的文件。"""
        path = Path(directory or self.save_dir).expanduser()
        if not path.is_absolute():
            raise HTTPException(400, "请填写目录的完整路径")
        # 首次使用时保存目录可能尚未创建，打开最近的已有上级目录。
        if directory is None:
            while not path.exists() and path != path.parent:
                path = path.parent
        try:
            path = path.resolve()
            children = sorted(
                (item for item in path.iterdir() if item.is_dir()),
                key=lambda item: item.name.casefold(),
            )
        except OSError:
            raise HTTPException(400, "无法打开此目录，请检查路径和访问权限") from None
        roots = (
            [
                str(Path(f"{letter}:/"))
                for letter in string.ascii_uppercase
                if Path(f"{letter}:/").exists()
            ]
            if os.name == "nt"
            else ["/"]
        )
        return {
            "path": str(path),
            "parent": str(path.parent) if path != path.parent else None,
            "roots": roots,
            "directories": [{"name": item.name, "path": str(item)} for item in children],
        }

    def path(self, voice_id, suffix=".voice"):
        if not re.fullmatch(r"[0-9a-f]{32}", voice_id):
            raise HTTPException(404, "音色不存在")
        return self.directory / f"{voice_id}{suffix}"

    def record(self, voice_id):
        try:
            return json.loads(self.path(voice_id, ".json").read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise HTTPException(404, "音色不存在") from None

    def file_path(self, record):
        """旧记录仍从原目录读取；新记录指向用户选择的实际位置。"""
        return Path(record.get("file_path", self.path(record["id"])))

    def public(self, record):
        return {
            **{key: value for key, value in record.items() if key != "source"},
            "file_path": str(self.file_path(record)),
        }

    def save(self, voice, *, name, model_key, model, save_dir=None):
        """只在克隆或导入成功后保存；失败时撤掉本次未完成的记录。"""
        self.directory.mkdir(parents=True, exist_ok=True)
        voice_id = uuid4().hex
        destination = self.prepare_directory(save_dir) / f"{voice_id}.voice"
        record = {
            "id": voice_id,
            "name": name,
            "model_key": model_key,
            "adapter": voice.adapter,
            "model": voice.model,
            "created_at": datetime.now(UTC).isoformat(),
            "source": model_source(model),
            "file_path": str(destination),
        }
        try:
            voice.save(destination)
            self.path(voice_id, ".json").write_text(
                json.dumps(record, ensure_ascii=False), encoding="utf-8"
            )
            (self.directory / ".preferences.json").write_text(
                json.dumps({"save_dir": str(destination.parent)}, ensure_ascii=False),
                encoding="utf-8",
            )
        except BaseException:
            destination.unlink(missing_ok=True)
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
            json.loads(path.read_text(encoding="utf-8"))
            for path in self.directory.glob("*.json")
            if re.fullmatch(r"[0-9a-f]{32}", path.stem)
        ]
        return [
            self.public(record)
            for record in sorted(records, key=lambda item: item["created_at"], reverse=True)
            if record["source"] == source
        ]

    def load(self, voice_id, model):
        if self.record(voice_id)["source"] != model_source(model):
            raise TTSError("VOICE_INCOMPATIBLE", "load_voice", "所选音色不属于当前模型或服务地址")
        return Voice.load(self.file_path(self.record(voice_id)))

    def download(self, voice_id):
        record = self.record(voice_id)
        return FileResponse(
            self.file_path(record),
            media_type="application/octet-stream",
            filename=f"{voice_id}.voice",
        )

    def remove(self, voice_id):
        """只移除应用的本地记录，不请求删除服务商的远端音色。"""
        record = self.record(voice_id)
        self.file_path(record).unlink(missing_ok=True)
        self.path(voice_id, ".json").unlink()
