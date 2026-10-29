"""长音频适配通过公开 SDK 与本机 HTTP 服务验证，不访问付费端点。"""

import json
import tempfile
import unittest
import wave
from contextlib import contextmanager
from email.parser import BytesParser
from email.policy import default
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from unittest.mock import patch

import speechio
from speechio import AudioFormat, RemoteModelConfig, STTError, Transcriber, transcribe_audio
from speechio.stt.options import TranscriptionOptions
from tests.audio.encoded_samples import encoded_audio

MODEL = "qwen-audio-3.1-asr-flash-filetrans"


@contextmanager
def file_service(*, states=("SUCCEEDED",), overrides=None):
    """模拟上传、任务和结果三个外部协议，保留收到的真实请求。"""
    requests = []
    status_index = 0

    class Handler(BaseHTTPRequestHandler):
        def handle_request(self):
            nonlocal status_index
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            requests.append((self.command, self.path, dict(self.headers), body))
            root = f"http://127.0.0.1:{self.server.server_port}"
            if self.path.startswith("/api/v1/uploads?"):
                data = {
                    "data": {
                        "upload_host": root + "/upload",
                        "upload_dir": "temporary",
                        "oss_access_key_id": "upload-key",
                        "signature": "upload-signature",
                        "policy": "upload-policy",
                        "x_oss_object_acl": "private",
                        "x_oss_forbid_overwrite": "true",
                    }
                }
            elif self.path == "/upload":
                data = {}
            elif self.path == "/api/v1/services/audio/asr/transcription":
                data = {"output": {"task_id": "task-1", "task_status": "PENDING"}}
            elif self.path == "/api/v1/tasks/task-1":
                status = states[min(status_index, len(states) - 1)]
                status_index += 1
                data = {
                    "output": {
                        "task_id": "task-1",
                        "task_status": status,
                        "results": [
                            {"subtask_status": "SUCCEEDED", "transcription_url": root + "/result"}
                        ],
                    }
                }
            elif self.path == "/result":
                data = {
                    "properties": {"original_duration_in_milliseconds": 1000},
                    "transcripts": [
                        {
                            "channel_id": 0,
                            "text": "你好 hello",
                            "sentences": [
                                {
                                    "words": [
                                        {"text": "你好 ", "begin_time": 0, "end_time": 400},
                                        {"text": "hello", "begin_time": 400, "end_time": 900},
                                    ]
                                }
                            ],
                        }
                    ],
                }
            else:
                self.send_error(404)
                return
            status_code = 200
            if overrides and self.path in overrides:
                status_code, data = overrides[self.path]
            encoded = json.dumps(data).encode()
            self.send_response(status_code)
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        do_GET = do_POST = handle_request

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01})
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


class RemoteFileTests(unittest.TestCase):
    def test_m4a_upload_failure_releases_file_without_submitting_task(self):
        with (
            tempfile.TemporaryDirectory() as folder,
            file_service(overrides={"/upload": (500, {"message": "upload-signature"})}) as (
                root,
                requests,
            ),
        ):
            path = Path(folder) / "recording.m4a"
            path.write_bytes(encoded_audio())
            with self.assertRaises(STTError) as caught:
                transcribe_audio(
                    path,
                    format=AudioFormat(),
                    config=RemoteModelConfig(
                        provider="bailian",
                        model=MODEL,
                        api_key="test-secret",
                        base_url=root,
                    ),
                )
            self.assertEqual(caught.exception.code, "REMOTE_SERVICE")
            self.assertNotIn("upload-signature", str(caught.exception))
            self.assertEqual([r[1].split("?")[0] for r in requests], ["/api/v1/uploads", "/upload"])
            path.unlink()

    def test_pending_running_and_total_wait_timeout(self):
        for states, timeout, expected_time in (
            (("PENDING", "RUNNING", "SUCCEEDED"), 10, 4),
            (("RUNNING",), 3, 3),
        ):
            with self.subTest(states=states), file_service(states=states) as (root, requests):
                clock = [0.0]

                def advance(seconds):
                    clock[0] += seconds

                config = RemoteModelConfig(
                    provider="bailian",
                    model=MODEL,
                    api_key="test-secret",
                    base_url=root,
                )
                with (
                    patch("speechio.adapters.remote.bailian_file.monotonic", lambda: clock[0]),
                    patch(
                        "speechio.cancellation.CancellationToken.wait_cancelled",
                        side_effect=advance,
                    ),
                    Transcriber(config) as recognizer,
                ):
                    if timeout == 3:
                        with self.assertRaises(STTError) as caught:
                            recognizer.transcribe_url(
                                "https://audio.example/long.m4a", task_timeout_seconds=timeout
                            )
                        self.assertEqual(caught.exception.code, "REMOTE_TIMEOUT")
                    else:
                        result = recognizer.transcribe_url(
                            "https://audio.example/long.m4a", task_timeout_seconds=timeout
                        )
                        self.assertEqual(result.text, "你好 hello")
                self.assertEqual(clock[0], expected_time)
                self.assertEqual(recognizer.state, "closed")
                self.assertEqual(sum(r[0] == "POST" for r in requests), 1)

    def test_task_and_result_failures_do_not_return_partial_success_or_resubmit(self):
        submit = "/api/v1/services/audio/asr/transcription"
        task = "/api/v1/tasks/task-1"
        for path, status, data, code in (
            (submit, 401, {"message": "test-secret"}, "REMOTE_AUTH"),
            (submit, 429, {"message": "test-secret"}, "REMOTE_RATE_LIMIT"),
            (
                task,
                200,
                {"output": {"task_id": "task-1", "task_status": "FAILED"}},
                "REMOTE_SERVICE",
            ),
            (
                task,
                200,
                {
                    "output": {
                        "task_id": "task-1",
                        "task_status": "SUCCEEDED",
                        "results": [{"subtask_status": "FAILED"}],
                    }
                },
                "REMOTE_SERVICE",
            ),
            (
                task,
                200,
                {"output": {"task_id": "wrong", "task_status": "RUNNING"}},
                "REMOTE_PROTOCOL",
            ),
            ("/result", 200, {"transcripts": [{"text": "partial"}]}, "REMOTE_PROTOCOL"),
            ("/result", 302, {"message": "test-secret"}, "REMOTE_SERVICE"),
        ):
            with (
                self.subTest(path=path, data=data),
                file_service(overrides={path: (status, data)}) as (root, requests),
            ):
                with self.assertRaises(STTError) as caught:
                    speechio.transcribe_url(
                        "https://audio.example/long.m4a",
                        config=RemoteModelConfig(
                            provider="bailian",
                            model=MODEL,
                            api_key="test-secret",
                            base_url=root,
                        ),
                    )
                self.assertEqual(caught.exception.code, code)
                self.assertNotIn("test-secret", str(caught.exception))
                self.assertEqual(sum(r[0] == "POST" for r in requests), 1)

    def test_invalid_inputs_and_limits_are_rejected_before_upload(self):
        with tempfile.TemporaryDirectory() as folder, file_service() as (root, requests):
            path = Path(folder) / "recording.wav"
            with wave.open(str(path), "wb") as output:
                output.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                output.writeframes(bytes(32000))
            config = RemoteModelConfig(
                provider="bailian",
                model=MODEL,
                api_key="test-secret",
                base_url=root,
            )
            for operation, code in (
                (
                    lambda: transcribe_audio(bytes(100), format=AudioFormat(), config=config),
                    "BACKEND_INCOMPATIBLE",
                ),
                (
                    lambda: transcribe_audio(
                        path, format=AudioFormat(), config=config, max_audio_seconds=0.5
                    ),
                    "AUDIO_CAPACITY_EXCEEDED",
                ),
                (
                    lambda: transcribe_audio(
                        path.with_name("missing.wav"), format=AudioFormat(), config=config
                    ),
                    "AUDIO_READ_FAILED",
                ),
                (
                    lambda: speechio.transcribe_url("file:///recording.wav", config=config),
                    "INVALID_CONFIGURATION",
                ),
                (
                    lambda: speechio.transcribe_url(
                        "https://audio.example/a.wav", config=config, max_audio_seconds=1
                    ),
                    "INVALID_CONFIGURATION",
                ),
                (
                    lambda: speechio.transcribe_url(
                        "https://audio.example/a.wav",
                        config=RemoteModelConfig(
                            provider="bailian",
                            model="qwen-audio-3.1-asr-flash",
                            api_key="test-secret",
                            base_url=root,
                        ),
                    ),
                    "BACKEND_INCOMPATIBLE",
                ),
            ):
                with self.subTest(code=code), self.assertRaises(STTError) as caught:
                    operation()
                self.assertEqual(caught.exception.code, code)
            self.assertEqual(requests, [])
            path.unlink()
        for invalid in (0, -1, True, float("inf"), float("nan")):
            with self.subTest(timeout=invalid), self.assertRaises(STTError) as caught:
                speechio.transcribe_url(
                    "https://audio.example/a.wav",
                    config=RemoteModelConfig(provider="bailian", model=MODEL),
                    task_timeout_seconds=invalid,
                )
            self.assertEqual(caught.exception.code, "INVALID_CONFIGURATION")

    def test_url_entrypoints_submit_address_without_download_or_upload(self):
        for entry in ("shortcut", "object"):
            with self.subTest(entry=entry), file_service() as (root, requests):
                url = "https://audio.example/recording.m4a?signature=private-value"
                config = RemoteModelConfig(
                    provider="bailian",
                    model=MODEL,
                    api_key="test-secret",
                    base_url=root,
                )
                if entry == "shortcut":
                    result = speechio.transcribe_url(url, config=config)
                else:
                    with Transcriber(config) as recognizer:
                        result = recognizer.transcribe_url(url)
                self.assertEqual(result.text, "你好 hello")
                self.assertEqual(result.timestamp_status, "not_requested")
                self.assertIsNone(result.words)
                self.assertEqual(len(requests), 3)
                self.assertEqual(json.loads(requests[0][3])["input"]["file_urls"], [url])
                self.assertNotIn("X-DashScope-OssResourceResolve", requests[0][2])
                self.assertNotIn("Authorization", requests[-1][2])

    def test_local_file_upload_returns_text_and_word_timestamps(self):
        with tempfile.TemporaryDirectory() as folder, file_service() as (url, requests):
            path = Path(folder) / "recording.wav"
            with wave.open(str(path), "wb") as output:
                output.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                output.writeframes(b"\x01\x00" * 16000 * 301)
            original = path.read_bytes()
            result = transcribe_audio(
                path,
                format=AudioFormat(),
                config=RemoteModelConfig(
                    provider="bailian",
                    model=MODEL,
                    api_key="test-secret",
                    base_url=url,
                ),
                options=TranscriptionOptions(language="zh-CN", word_timestamps=True),
            )
            self.assertEqual(result.text, "你好 hello")
            self.assertEqual(result.requested_language, "zh-CN")
            self.assertEqual(result.backend_metadata["task_id"], "task-1")
            self.assertEqual(result.timestamp_status, "reported")
            self.assertEqual(result.words[-1].end_sample, 14400)
            upload = next(r for r in requests if r[1] == "/upload")
            multipart = BytesParser(policy=default).parsebytes(
                ("Content-Type: " + upload[2]["Content-Type"] + "\r\n\r\n").encode() + upload[3]
            )
            fields = {
                part.get_param("name", header="content-disposition"): part.get_payload(decode=True)
                for part in multipart.iter_parts()
            }
            self.assertEqual(fields["file"], original)
            self.assertNotIn("Authorization", upload[2])
            submitted = next(r for r in requests if r[0] == "POST" and r[1] != "/upload")
            payload = json.loads(submitted[3])
            self.assertEqual(payload["model"], MODEL)
            self.assertEqual(payload["input"]["file_urls"], ["oss://" + fields["key"].decode()])
            self.assertEqual(payload["parameters"], {"channel_id": [0], "language_hints": ["zh"]})
            self.assertEqual(submitted[2]["X-DashScope-Async"], "enable")
            self.assertEqual(submitted[2]["X-DashScope-OssResourceResolve"], "enable")
            self.assertNotIn("Authorization", requests[-1][2])
            path.unlink()  # Windows 下确认上传后文件句柄已释放。
