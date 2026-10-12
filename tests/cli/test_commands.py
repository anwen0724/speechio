"""CLI 使用真实会话与转换，只替换外部模型和标准流。"""

import io
import json
import unittest
from _thread import interrupt_main
from threading import Event, Thread
from types import SimpleNamespace
from unittest.mock import patch

from realtime_asr.cli.runner import main
from tests.runtime import test_recognizer


class CommandTests(unittest.TestCase):
    def setUp(self):
        test_recognizer.RecognizerTests.setUp(self)

    def test_dual_cli_streams_both_roles_before_stdin_eof(self):
        received = Event()
        roles = set()

        class Input:
            first = True

            def read(self, count):
                if self.first:
                    self.first = False
                    return b"\x01\x00" * 1024
                if not received.wait(5):
                    raise OSError("两路实时未在输入结束前交付")
                return b""

        class Output(io.StringIO):
            def write(self, text):
                record = json.loads(text)
                if record["record_type"] == "event" and record["event"]["kind"] == "realtime":
                    roles.add(record["event"]["realtime_role"])
                    if roles == {"accurate", "ultrafast"}:
                        received.set()
                return super().write(text)

        def inference(audio, **options):
            return iter(
                [SimpleNamespace(text="speech system today", start=0, end=0.1)]
            ), SimpleNamespace(language="en", language_probability=1)

        stdout, stderr = Output(), io.StringIO()
        with patch("faster_whisper.BatchedInferencePipeline.transcribe", side_effect=inference):
            code = main(
                [
                    "stdin",
                    "--model",
                    str(self.path),
                    "--realtime",
                    "--dual-realtime",
                    "--ultrafast-model",
                    str(self.path),
                    "--output",
                    "jsonl",
                ],
                stdin=Input(),
                stdout=stdout,
                stderr=stderr,
            )
        self.assertEqual(code, 0, stderr.getvalue())
        self.assertEqual(roles, {"accurate", "ultrafast"})
        events = [
            record["event"]
            for line in stdout.getvalue().splitlines()
            if (record := json.loads(line))["record_type"] == "event"
        ]
        self.assertEqual(events[-1]["kind"], "completed")
        self.assertTrue(any(event["merge"] and event["merge"]["text"] for event in events))

    def test_json_input_error_is_structured_and_also_visible_on_stderr(self):
        stdout, stderr = io.StringIO(), io.StringIO()
        code = main(
            ["stdin", "--model", str(self.path), "--output", "jsonl"],
            stdin=io.BytesIO(b"\x01"),
            stdout=stdout,
            stderr=stderr,
        )
        records = [json.loads(line) for line in stdout.getvalue().splitlines()]
        self.assertEqual(code, 1)
        error = next(record for record in records if record["record_type"] == "error")
        self.assertEqual(error["code"], "AUDIO_INPUT_TRUNCATED")
        self.assertIn("AUDIO_INPUT_TRUNCATED", stderr.getvalue())

    def test_interrupt_drains_accepted_audio_and_reports_stop_once(self):
        class Input:
            def __init__(self):
                self.first = True

            def read(self, count):
                if self.first:
                    self.first = False
                    return b"\x01\x00" * 512
                raise KeyboardInterrupt()

        stdout, stderr = io.StringIO(), io.StringIO()
        result = (
            iter([SimpleNamespace(text="tail", start=0, end=0.032)]),
            SimpleNamespace(language="en", language_probability=1),
        )
        with patch("faster_whisper.BatchedInferencePipeline.transcribe", return_value=result):
            code = main(
                ["stdin", "--model", str(self.path), "--output", "jsonl"],
                stdin=Input(),
                stdout=stdout,
                stderr=stderr,
            )
        records = [json.loads(line) for line in stdout.getvalue().splitlines()]
        self.assertEqual(code, 130)
        events = [r["event"]["kind"] for r in records if r["record_type"] == "event"]
        self.assertEqual(events, ["final", "completed"])
        stops = [r for r in records if r.get("status") == "stopped"]
        self.assertEqual(len(stops), 1, "停止状态重复交付")
        self.assertEqual(stops[0]["reason"], "interrupt")

    def test_stdin_wait_does_not_block_realtime_output_and_text_stdout_is_final_only(self):
        preview = Event()

        class Input:
            def __init__(self):
                self.first = True

            def read(self, count):
                if self.first:
                    self.first = False
                    return b"\x01\x00" * 1024
                if not preview.wait(5):
                    raise OSError("实时输出没有在 stdin 等待期间交付")
                return b""

        class Diagnostic(io.StringIO):
            def write(self, text):
                if "[snapshot:" in text:
                    preview.set()
                return super().write(text)

        def transcribe(audio, **kwargs):
            return iter([SimpleNamespace(text="result", start=0, end=0.064)]), (
                SimpleNamespace(language="en", language_probability=1)
            )

        stdout, stderr = io.StringIO(), Diagnostic()
        with patch("faster_whisper.BatchedInferencePipeline.transcribe", side_effect=transcribe):
            code = main(
                ["stdin", "--model", str(self.path), "--realtime"],
                stdin=Input(),
                stdout=stdout,
                stderr=stderr,
            )
        self.assertEqual(code, 0)
        self.assertEqual(stdout.getvalue(), "result\n")
        self.assertTrue(preview.is_set())
        self.assertNotIn("\r", stdout.getvalue())

    def test_file_sequence_preserves_previous_result_when_next_file_fails(self):
        import wave

        file = self.path / "one.wav"
        with wave.open(str(file), "wb") as output:
            output.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
            output.writeframes(b"\x01\x00" * 515)
        stdout, stderr = io.StringIO(), io.StringIO()
        result = (
            iter([SimpleNamespace(text="one", start=0, end=0.032)]),
            SimpleNamespace(language="en", language_probability=1),
        )
        with patch("faster_whisper.BatchedInferencePipeline.transcribe", return_value=result):
            code = main(
                [
                    "file",
                    str(file),
                    str(self.path / "missing.wav"),
                    "--model",
                    str(self.path),
                    "--output",
                    "jsonl",
                ],
                stdout=stdout,
                stderr=stderr,
            )
        records = [json.loads(line) for line in stdout.getvalue().splitlines()]
        finals = [r for r in records if r.get("event", {}).get("kind") == "final"]
        self.assertEqual(code, 1)
        self.assertEqual(len(finals), 1)
        self.assertEqual((finals[0]["input_index"], finals[0]["event"]["audio_samples"]), (0, 515))
        error = next(r for r in records if r["record_type"] == "error")
        self.assertEqual(error["input_index"], 1)

    def test_broken_pipe_stops_command_and_does_not_close_borrowed_stream(self):
        class Broken(io.StringIO):
            def write(self, text):
                if '"record_type": "event"' in text:
                    self.pipe_closed = True
                return super().write(text)

            def flush(self):
                if getattr(self, "pipe_closed", False):
                    # Windows 的关闭管道在刷新时返回 EINVAL。
                    raise OSError(22, "Invalid argument")

        result = (
            iter([SimpleNamespace(text="result", start=0, end=0.032)]),
            SimpleNamespace(language="en", language_probability=1),
        )
        stdout, stderr = Broken(), io.StringIO()
        stdin = io.BytesIO(b"\x01\x00" * 515)
        with patch("faster_whisper.BatchedInferencePipeline.transcribe", return_value=result):
            code = main(
                ["stdin", "--model", str(self.path), "--output", "jsonl"],
                stdin=stdin,
                stdout=stdout,
                stderr=stderr,
            )
        self.assertEqual(code, 1)
        self.assertFalse(stdin.closed or stdout.closed or stderr.closed)

    def test_drain_deadline_reports_pending_cleanup_instead_of_success(self):
        entered, release = Event(), Event()

        def transcribe(audio, **kwargs):
            entered.set()
            if not release.wait(5):
                raise OSError("测试没有释放外部推理")
            return iter([SimpleNamespace(text="late", start=0, end=0.032)]), (
                SimpleNamespace(language="en", language_probability=1)
            )

        stdout, stderr = io.StringIO(), io.StringIO()
        codes = []

        def invoke():
            codes.append(
                main(
                    [
                        "stdin",
                        "--model",
                        str(self.path),
                        "--output",
                        "jsonl",
                        "--drain-timeout",
                        "1",
                        "--close-timeout",
                        "0.05",
                    ],
                    stdin=io.BytesIO(b"\x01\x00" * 515),
                    stdout=stdout,
                    stderr=stderr,
                )
            )

        with patch("faster_whisper.BatchedInferencePipeline.transcribe", side_effect=transcribe):
            command = Thread(target=invoke)
            try:
                command.start()
                self.assertTrue(entered.wait(5))
                command.join(5)
                self.assertFalse(command.is_alive())
                records = [json.loads(line) for line in stdout.getvalue().splitlines()]
                self.assertEqual(codes, [4])
                errors = [r for r in records if r["record_type"] == "error"]
                self.assertIn("DRAIN_TIMEOUT", [r["code"] for r in errors])
                cleanup = next(r for r in errors if r["code"] == "CLEANUP_FAILED")
                self.assertEqual(cleanup["cleanup"][0]["status"], "pending")
                self.assertFalse(any(r.get("event", {}).get("kind") == "final" for r in records))
            finally:
                release.set()
                command.join(5)

    def test_second_interrupt_cancels_final_and_keeps_user_exit_code(self):
        release = Event()

        class Input:
            first = True

            def read(self, count):
                if self.first:
                    self.first = False
                    return b"\x01\x00" * 515
                raise KeyboardInterrupt()

        def transcribe(audio, **kwargs):
            # 第一次来自输入端；第二次是推理期间送达主线程的用户中断。
            interrupt_main()
            if not release.wait(5):
                raise OSError("测试没有释放外部推理")
            return iter([SimpleNamespace(text="late", start=0, end=0.032)]), (
                SimpleNamespace(language="en", language_probability=1)
            )

        stdout, stderr = io.StringIO(), io.StringIO()
        with patch("faster_whisper.BatchedInferencePipeline.transcribe", side_effect=transcribe):
            try:
                code = main(
                    [
                        "stdin",
                        "--model",
                        str(self.path),
                        "--output",
                        "jsonl",
                        "--drain-timeout",
                        "2",
                        "--close-timeout",
                        "0.05",
                    ],
                    stdin=Input(),
                    stdout=stdout,
                    stderr=stderr,
                )
                records = [json.loads(line) for line in stdout.getvalue().splitlines()]
                self.assertEqual(code, 130)
                events = [r["event"]["kind"] for r in records if r["record_type"] == "event"]
                self.assertEqual(events, ["cancelled"])
                self.assertEqual(records[-1]["reason"], "interrupt")
            finally:
                release.set()


if __name__ == "__main__":
    unittest.main()
