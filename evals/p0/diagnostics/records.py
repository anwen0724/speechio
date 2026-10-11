"""UTF-8 机器记录；同一次验证统一归属到 run_id。"""

import json
from pathlib import Path


class RecordWriter:
    """逐行落盘；调用方在计时区间之外写入，避免把 IO 算入推理。"""

    def __init__(self, folder, run_id):
        self.folder = Path(folder).resolve()
        self.folder.mkdir(parents=True, exist_ok=False)
        self.run_id = run_id
        self.path = self.folder / "records.jsonl"

    def write(self, record_type, **fields):
        """未知值使用 null，JSON 不允许隐含 NaN 或 Infinity。"""
        record = {"schema_version": 1, "record_type": record_type, "run_id": self.run_id, **fields}
        with self.path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        return record
