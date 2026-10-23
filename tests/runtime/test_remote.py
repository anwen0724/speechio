"""通过公开 SDK 和本机协议服务验证远程适配，不访问真实百炼账户。"""
import base64
import io
import json
import unittest
import wave
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Event, Thread
from unittest.mock import patch

from speechio import (
    AudioFormat,
    LocalModelConfig,
    RealtimeConfig,
    RemoteModelConfig,
    SessionConfig,
    STTError,
    Transcriber,
    transcribe_audio,
)
from speechio.stt.options import TranscriptionOptions
from speechio.stt.runtime.asr import STTRuntime
from tests.runtime.test_engine_injection import RecordingEngine

FLASH = "qwen-audio-3.1-asr-flash"
STREAMING = "qwen-audio-3.1-asr-flash-streaming"


@contextmanager
def websocket_service(handler, **options):
    """真实本机 WebSocket，仅替代百炼服务端。"""
    from websockets.sync.server import serve

    with serve(handler, "127.0.0.1", 0, **options) as server:
        thread = Thread(target=server.serve_forever)
        thread.start()
        try:
            yield f"http://127.0.0.1:{server.socket.getsockname()[1]}"
        finally:
            server.shutdown()
            thread.join(timeout=2)


@contextmanager
def http_service(response, status=200):
    """仅替代外部 HTTP 服务，记录真实请求，退出时关闭服务线程。"""
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append((self.path, self.headers["Authorization"], body))
            encoded = json.dumps(response).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

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


class RemoteTests(unittest.TestCase):
    def test_finish_wait_extends_only_when_recognition_progresses(self):
        """替代外部连接和时钟，确定性验证跨越总时限的尾部处理。"""
        for outcome in ("progress", "heartbeat", "duplicate"):
            with self.subTest(outcome=outcome):
                now = [0.0]

                class Connection:
                    def __init__(self):
                        self.task_id = None
                        self.started = False
                        self.ending = False
                        self.closed = False
                        self.event_index = 0

                    def __enter__(self):
                        return self

                    def __exit__(self, *args):
                        self.closed = True

                    def send(self, message):
                        if isinstance(message, bytes):
                            return
                        header = json.loads(message)["header"]
                        self.task_id = header["task_id"]
                        self.ending = header["action"] == "finish-task"

                    def recv(self, timeout):
                        if not self.started:
                            self.started = True
                            event, output = "task-started", {}
                        elif not self.ending:
                            raise TimeoutError
                        else:
                            # 每隔 20 秒一个事件；结果完整结束共需 60 秒。
                            if timeout < 20:
                                now[0] += timeout
                                raise TimeoutError
                            now[0] += 20
                            self.event_index += 1
                            event, output = "task-finished", {}
                            if self.event_index < 3:
                                sentence = {
                                    "sentence_id": 1,
                                    "text": "你好" if self.event_index == 2 else "你",
                                    "sentence_end": self.event_index == 2,
                                }
                                if outcome == "heartbeat":
                                    sentence = {"heartbeat": True}
                                elif outcome == "duplicate":
                                    sentence = {
                                        "sentence_id": 1,
                                        "text": "你",
                                        "sentence_end": False,
                                    }
                                event = "result-generated"
                                output = {"sentence": sentence}
                        return json.dumps(
                            {
                                "header": {"event": event, "task_id": self.task_id},
                                "payload": {"output": output},
                            }
                        )

                connection = Connection()
                config = RemoteModelConfig(
                    provider="bailian",
                    model=STREAMING,
                    api_key="test-secret",
                    base_url="http://127.0.0.1:1",
                    timeout_seconds=30,
                )
                with (
                    patch(
                        "speechio.adapters.remote.bailian_stream._DirectConnection",
                        return_value=connection,
                    ),
                    patch(
                        "speechio.adapters.remote.bailian_stream.monotonic",
                        side_effect=lambda: now[0],
                    ),
                ):
                    if outcome == "progress":
                        result = transcribe_audio(bytes(3200), format=AudioFormat(), config=config)
                        self.assertEqual(result.text, "你好")
                        self.assertEqual(now[0], 60)
                    else:
                        with self.assertRaises(STTError) as caught:
                            transcribe_audio(bytes(3200), format=AudioFormat(), config=config)
                        self.assertEqual(caught.exception.code, "REMOTE_TIMEOUT")
                        self.assertEqual(now[0], 30 if outcome == "heartbeat" else 50)
                    self.assertTrue(connection.closed)

    def test_websocket_redirect_does_not_forward_credentials(self):
        forwarded = []

        def destination(ws):
            forwarded.append(ws.request.headers.get("Authorization"))
            ws.close()

        with websocket_service(destination) as target:

            def redirect(connection, request):
                response = connection.respond(307, "redirect")
                response.headers["Location"] = target.replace("http://", "ws://") + "/elsewhere"
                return response

            with websocket_service(lambda ws: None, process_request=redirect) as url:
                with self.assertRaises(STTError) as caught:
                    transcribe_audio(
                        bytes(1024),
                        format=AudioFormat(),
                        config=RemoteModelConfig(
                            provider="bailian",
                            model=STREAMING,
                            api_key="test-secret",
                            base_url=url,
                        ),
                    )
                self.assertEqual(forwarded, [])
                self.assertEqual(caught.exception.code, "REMOTE_SERVICE")

    def test_full_response_timestamps_are_mapped_but_partial_coverage_is_unavailable(self):
        for complete in (True, False):
            with self.subTest(complete=complete):
                sentence = {
                    "words": [
                        {
                            "text": "Hello",
                            "punctuation": ".",
                            "begin_time": 10,
                            "end_time": 50,
                            "fixed": True,
                        }
                    ]
                }
                response = {
                    "output": {
                        "text": "Hello." if complete else "你好 Hello.",
                        "sentence": sentence,
                    }
                }
                with http_service(response) as (url, _):
                    result = transcribe_audio(
                        bytes(3200),
                        format=AudioFormat(),
                        options=TranscriptionOptions(word_timestamps=True),
                        config=RemoteModelConfig(
                            provider="bailian", model=FLASH, api_key="test-secret", base_url=url
                        ),
                    )
                if complete:
                    self.assertEqual(result.timestamp_status, "reported")
                    self.assertEqual(
                        (result.words[0].start_sample, result.words[0].end_sample), (160, 800)
                    )
                else:
                    self.assertEqual(result.timestamp_status, "unavailable")
                    self.assertIsNone(result.words)

    def test_remote_http_failures_are_classified_without_retry_or_secret_disclosure(self):
        for status, response, expected in (
            (401, {"message": "test-secret"}, "REMOTE_AUTH"),
            (429, {"message": "test-secret"}, "REMOTE_RATE_LIMIT"),
            (200, {"output": {"text": None}}, "REMOTE_PROTOCOL"),
        ):
            with self.subTest(status=status), http_service(response, status) as (url, requests):
                with self.assertRaises(STTError) as caught:
                    transcribe_audio(
                        bytes(1024),
                        format=AudioFormat(),
                        config=RemoteModelConfig(
                            provider="bailian",
                            model=FLASH,
                            api_key="test-secret",
                            base_url=url,
                        ),
                    )
                self.assertEqual(caught.exception.code, expected)
                self.assertNotIn("test-secret", str(caught.exception))
                self.assertEqual(len(requests), 1)

    def test_optional_dependencies_and_config_validation_do_not_load_local_models(self):
        self.assertIs(LocalModelConfig, LocalModelConfig)
        config = RemoteModelConfig(provider="bailian", model=FLASH, api_key="test-secret")
        with patch.dict("sys.modules", {"httpx": None}):
            self.assertFalse(Transcriber(config).capabilities.streaming)
            local = Transcriber(LocalModelConfig(adapter="whisper", model_path="unused"))
            self.assertEqual(local.state, "unprepared")
            local.close()
            remote = Transcriber(config)
            try:
                with self.assertRaises(STTError) as caught:
                    remote.prepare()
                self.assertEqual(caught.exception.code, "DEPENDENCY_MISSING")
            finally:
                remote.close()
        with self.assertRaises(STTError) as caught:
            Transcriber(
                RemoteModelConfig(provider="bailian", model="unknown", api_key="test-secret")
            )
        self.assertEqual(caught.exception.code, "BACKEND_INCOMPATIBLE")
        with self.assertRaises(STTError) as caught:
            Transcriber(RemoteModelConfig(provider="bailian", model=FLASH))
        self.assertEqual(caught.exception.code, "INVALID_CONFIGURATION")

    def test_streaming_model_accepts_whole_audio_and_requires_finish_confirmation(self):
        from websockets.exceptions import ConnectionClosed

        for outcome, expected in (
            ("success", None),
            ("disconnect", "REMOTE_CONNECTION"),
            ("timeout", "REMOTE_TIMEOUT"),
            ("task-failed", "REMOTE_SERVICE"),
        ):
            with self.subTest(outcome=outcome):
                received, commands = [], []
                closed = Event()

                def handler(ws):
                    try:
                        start = json.loads(ws.recv(timeout=3))
                        commands.append(start)
                        task_id = start["header"]["task_id"]

                        def send(event, output=None):
                            ws.send(
                                json.dumps(
                                    {
                                        "header": {
                                            "event": event,
                                            "task_id": task_id,
                                            "error_message": "test-secret",
                                        },
                                        "payload": {"output": output or {}},
                                    }
                                )
                            )

                        send("task-started")
                        for message in ws:
                            if isinstance(message, bytes):
                                received.append(message)
                            else:
                                commands.append(json.loads(message))
                                send(
                                    "result-generated",
                                    {
                                        "sentence": {
                                            "sentence_id": 1,
                                            "text": "hello",
                                            "sentence_end": True,
                                        }
                                    },
                                )
                                if outcome == "success":
                                    send("task-finished")
                                elif outcome == "disconnect":
                                    ws.close()
                                    return
                                elif outcome == "task-failed":
                                    send("task-failed")
                                # timeout 分支不发确认，直到客户端主动关闭。
                    except ConnectionClosed:
                        pass
                    finally:
                        closed.set()

                with websocket_service(handler) as url:
                    config = RemoteModelConfig(
                        provider="bailian",
                        model=STREAMING,
                        api_key="test-secret",
                        base_url=url,
                        timeout_seconds=0.5,
                    )
                    if expected is None:
                        # 文件整段也可使用流式模型，超过五分钟仍应完整发送。
                        pcm = bytes(301 * 16000 * 2)
                        with self.assertRaises(STTError) as limited:
                            transcribe_audio(
                                pcm, format=AudioFormat(), config=config, max_audio_seconds=300
                            )
                        self.assertEqual(limited.exception.code, "AUDIO_CAPACITY_EXCEEDED")
                        result = transcribe_audio(pcm, format=AudioFormat(), config=config)
                        self.assertEqual(result.text, "hello")
                    else:
                        pcm = bytes(3210)
                        with self.assertRaises(STTError) as caught:
                            transcribe_audio(pcm, format=AudioFormat(), config=config)
                        self.assertEqual(caught.exception.code, expected)
                        self.assertNotIn("test-secret", str(caught.exception))
                    self.assertEqual(b"".join(received), pcm)
                    self.assertEqual(len(commands), 2)
                    self.assertTrue(closed.wait(3))

    def test_short_audio_limits_reject_before_any_upload(self):
        import numpy as np

        with http_service({}) as (url, requests):
            for seconds, code in ((301, "AUDIO_TOO_LONG"), (250, "AUDIO_TOO_LARGE")):
                with self.subTest(seconds=seconds), self.assertRaises(STTError) as caught:
                    transcribe_audio(
                        np.zeros(seconds * 16000, dtype=np.int16),
                        format=AudioFormat(),
                        config=RemoteModelConfig(
                            provider="bailian", model=FLASH, api_key="test-secret", base_url=url
                        ),
                    )
                self.assertEqual(caught.exception.code, code)
            self.assertEqual(requests, [])

    def test_stream_updates_without_new_audio_and_drains_tail_before_close(self):
        release, closed = Event(), Event()
        received, commands = [], []

        def handler(ws):
            command = json.loads(ws.recv(timeout=3))
            commands.append(command)
            task_id = command["header"]["task_id"]

            def send(event, sentence=None):
                ws.send(
                    json.dumps(
                        {
                            "header": {"event": event, "task_id": task_id},
                            "payload": {"output": {"sentence": sentence} if sentence else {}},
                        }
                    )
                )

            send("task-started")
            received.append(ws.recv(timeout=3))
            send("result-generated", {"sentence_id": 1, "text": "你", "sentence_end": False})
            if not release.wait(3):
                return
            send("result-generated", {"sentence_id": 1, "text": "你好", "sentence_end": True})
            for message in ws:
                if isinstance(message, bytes):
                    received.append(message)
                else:
                    commands.append(json.loads(message))
                    send(
                        "result-generated",
                        {
                            "sentence_id": 2,
                            "text": " hello",
                            "sentence_end": True,
                        },
                    )
                    send("task-finished")
                    ws.recv(timeout=3)
            closed.set()

        # 服务端确认任务结束后，客户端主动关闭；正常关闭异常也是预期退出。
        def connection(ws):
            from websockets.exceptions import ConnectionClosedOK

            try:
                handler(ws)
            except ConnectionClosedOK:
                closed.set()

        with websocket_service(connection) as url:
            config = RemoteModelConfig(
                provider="bailian",
                model=STREAMING,
                api_key="test-secret",
                base_url=url,
            )
            main = RecordingEngine()
            runtime = STTRuntime.from_adapter(
                adapter=main,
                realtime=RealtimeConfig(
                    mode="streaming", config=config, interval_seconds=0.01, initial_delay_seconds=0
                ),
            )
            try:
                runtime.prepare()
                session = runtime.create_session(
                    SessionConfig(realtime=True, options=TranscriptionOptions(language="zh"))
                )
                session.start()
                session.feed_audio(b"\x01\x00" * 512)
                previews = []
                for expected in ("你", "你好"):
                    while True:
                        event = session.next_event(timeout=3)
                        self.assertIsNotNone(event)
                        self.assertIsNone(event.error)
                        if event.result and event.result.text == expected:
                            previews.append(event.result.text)
                            break
                    release.set()
                session.feed_audio(b"\x02\x00" * 13)
                session.end_turn()
                drained = None
                final = None
                while True:
                    event = session.next_event(timeout=3)
                    self.assertIsNotNone(event)
                    self.assertIsNone(event.error)
                    if event.kind == "realtime" and event.stream_finished:
                        drained = event.result.text
                    if event.kind == "completed":
                        break
                    if event.kind == "final":
                        final = event.result.text
                self.assertEqual(drained, "你好 hello")
                self.assertEqual(final, "你好 hello")
                self.assertEqual(main.inputs, [])
                self.assertEqual(b"".join(received), b"\x01\x00" * 512 + b"\x02\x00" * 13)
                self.assertEqual(
                    [c["header"]["action"] for c in commands], ["run-task", "finish-task"]
                )
                self.assertTrue(closed.wait(3))
            finally:
                release.set()
                self.assertEqual(runtime.close().status, "ok")

    def test_short_audio_uses_remote_config_and_returns_unified_result(self):
        response = {"output": {"text": "你好，hello。"}, "request_id": "request-1"}
        pcm = bytes([0, 1]) * 1600
        with http_service(response) as (url, requests):
            config = RemoteModelConfig(
                provider="bailian",
                model=FLASH,
                api_key="test-secret",
                base_url=url,
            )
            with Transcriber(config) as recognizer:
                result = recognizer.transcribe(
                    pcm, format=AudioFormat(), options=TranscriptionOptions(language="zh")
                )
            self.assertEqual(result.text, "你好，hello。")
            self.assertEqual(result.requested_language, "zh")
            self.assertEqual(result.backend_metadata["request_id"], "request-1")
            self.assertEqual(len(requests), 1)
            path, authorization, body = requests[0]
            self.assertEqual(path, "/api/v1/services/aigc/multimodal-generation/generation")
            self.assertEqual(authorization, "Bearer test-secret")
            self.assertEqual(body["model"], FLASH)
            self.assertEqual(body["parameters"]["language_hints"], ["zh"])
            data = body["input"]["messages"][0]["content"][0]["input_audio"]["data"]
            with wave.open(io.BytesIO(base64.b64decode(data.split(",", 1)[1]))) as audio:
                self.assertEqual((audio.getframerate(), audio.getnchannels()), (16000, 1))
                self.assertEqual(audio.readframes(audio.getnframes()), pcm)
            self.assertNotIn("test-secret", repr(config))
