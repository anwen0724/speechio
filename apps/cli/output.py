"""终端和 JSONL 的唯一出口；多线程写入保留整行与核心归属。"""

import json
from dataclasses import asdict
from threading import Lock


class Output:
    """文本正文只写 stdout；状态和实时预览只写 stderr。"""

    def __init__(self, mode, stdout, stderr):
        self.mode = mode
        self.stdout = stdout
        self.stderr = stderr
        self.failed = False
        self._lock = Lock()

    def _write_stdout(self, text):
        """输出通道失败后停止交付；Windows 管道可能以 EINVAL 报错。"""
        try:
            self.stdout.write(text)
            self.stdout.flush()
        except OSError as exc:
            self.failed = True
            raise BrokenPipeError("标准输出不可写") from exc

    def record(self, kind, **fields):
        with self._lock:
            if self.mode == "jsonl":
                record = {"schema_version": 2, "record_type": kind, **fields}
                self._write_stdout(json.dumps(record, ensure_ascii=False) + "\n")
                if kind == "error":
                    self.stderr.write(f"[{fields['code']}] {fields['message']}\n")
                    self.stderr.flush()
            elif kind == "status":
                reason = " " + fields["reason"] if fields.get("reason") else ""
                self.stderr.write(f"[{fields['status']}] {fields.get('source', '')}{reason}\n")
                self.stderr.flush()
            elif kind == "error":
                self.stderr.write(f"[{fields['code']}] {fields['message']}\n")
                self.stderr.flush()
            elif kind == "devices":
                self._write_stdout(
                    "".join(
                        f"{device['index']}: {device['name']} ({device['host_api']})\n"
                        for device in fields["devices"]
                    )
                )

    def event(self, event, *, input_index, source):
        if self.mode == "jsonl":
            self.record("event", event=asdict(event), input_index=input_index, source=source)
            return
        with self._lock:
            if event.kind == "final":
                self._write_stdout(event.result.text + "\n")
            elif event.kind == "realtime":
                if event.merge is not None:
                    text = event.merge.text
                elif event.stabilization is not None:
                    text = event.stabilization.display_text
                else:
                    text = event.result.text
                self.stderr.write(f"[{event.source}:{event.turn.turn_id}] {text}\n")
                self.stderr.flush()
            elif event.candidate is not None:
                value = event.candidate
                self.stderr.write(
                    f"[{event.kind}:{value.status}:{event.turn.turn_id}] {value.text}\n"
                )
                if event.error is not None:
                    self.stderr.write(f"[{event.error.code}] {event.error.message}\n")
                self.stderr.flush()
            elif event.error is not None:
                self.stderr.write(f"[{event.error.code}] {event.error.message}\n")
                self.stderr.flush()
