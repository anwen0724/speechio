"""本机 WebSocket 模拟百炼服务，SDK 执行真实协议流程。"""

import json
import socket
import unittest
from threading import Event, Thread
from unittest.mock import patch

from websockets.exceptions import ConnectionClosed
from websockets.sync.client import reconnect

from speechio import RemoteModelConfig, SynthesisOptions, Synthesizer, TTSError, synthesize
from tests.runtime.test_remote import websocket_service


class BailianSynthesisTests(unittest.TestCase):
    def test_send_timeout_releases_blocked_socket(self):
        """真实连接停止接收时，发送须自行超时；测试清理不能替代产品解除阻塞。"""
        release = Event()
        completed = Event()
        connections = []
        outcome = []

        class SmallBufferConnection(reconnect):
            def __init__(self, *args, **kwargs):
                # 禁用压缩并缩小缓冲，确保合法长度的文本足以阻塞真实发送。
                kwargs["compression"] = None
                super().__init__(*args, **kwargs)

            def __enter__(self):
                ws = super().__enter__()
                ws.socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 2048)
                connections.append(ws)
                return ws

        def service(ws):
            ws.socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 2048)
            run = json.loads(ws.recv())
            ws.send(
                json.dumps(
                    {"header": {"task_id": run["header"]["task_id"], "event": "task-started"}}
                )
            )
            release.wait()

        with websocket_service(
            service, max_queue=1, ping_interval=None, compression=None, close_timeout=0.2
        ) as url:
            synth = Synthesizer(
                RemoteModelConfig(
                    provider="bailian",
                    model="qwen-audio-3.1-tts-flash",
                    api_key="test-key",
                    base_url=url,
                    timeout_seconds=0.2,
                )
            )
            stream = sender = None
            try:
                with patch("websockets.sync.client.reconnect", SmallBufferConnection):
                    stream = synth.open_stream(options=SynthesisOptions(voice="test"))

                    def send():
                        try:
                            stream.send_text("a" * 200000)
                        except TTSError as exc:
                            outcome.append(exc.code)
                        finally:
                            completed.set()

                    sender = Thread(target=send, daemon=True)
                    sender.start()
                    self.assertTrue(completed.wait(2), "发送超时后仍未退出")
                    self.assertEqual(outcome, ["REMOTE_TIMEOUT"])
                    self.assertEqual(len(connections), 1, "超时后不应重连重发")
                    with self.assertRaises(TTSError) as caught:
                        stream.read(timeout=1)
                    self.assertEqual(caught.exception.code, "REMOTE_TIMEOUT")
                self.assertNotEqual(stream.close(timeout=2).status, "pending")
                self.assertNotEqual(synth.close(timeout=2).status, "pending")
            finally:
                # 旧实现失败时也能回收测试线程；断言必须发生在强制断开之前。
                for ws in connections:
                    ws.socket.close()
                release.set()
                if sender is not None:
                    sender.join(2)
                if stream is not None:
                    stream.close(timeout=2)
                synth.close(timeout=2)

    def test_cancel_interrupts_connection_without_waiting_for_send_lock(self):
        received = Event()

        def service(ws):
            run = json.loads(ws.recv())
            task_id = run["header"]["task_id"]
            ws.send(json.dumps({"header": {"task_id": task_id, "event": "task-started"}}))
            from websockets.exceptions import ConnectionClosed

            try:
                ws.recv()
            except ConnectionClosed:
                received.set()

        with websocket_service(service) as url:
            config = RemoteModelConfig(
                provider="bailian",
                model="qwen-audio-3.1-tts-flash",
                api_key="test-key",
                base_url=url,
            )
            with Synthesizer(config) as synth:
                stream = synth.open_stream(options=SynthesisOptions(voice="test"))
                stream.cancel()
                self.assertTrue(received.wait(1))
                self.assertEqual(stream.close(timeout=1).status, "ok")
                with self.assertRaises(TTSError) as caught:
                    stream.read()
                self.assertEqual(caught.exception.code, "CANCELLED")

    def test_long_text_is_split_within_one_task_and_rejects_total_overflow(self):
        parts = []

        def service(ws):
            run = json.loads(ws.recv())
            task_id = run["header"]["task_id"]
            ws.send(json.dumps({"header": {"task_id": task_id, "event": "task-started"}}))
            while True:
                from websockets.exceptions import ConnectionClosed

                try:
                    item = json.loads(ws.recv())
                except ConnectionClosed:
                    break
                self.assertEqual(item["header"]["task_id"], task_id)
                if item["header"]["action"] == "continue-task":
                    parts.append(item["payload"]["input"]["text"])
                else:
                    break

        with websocket_service(service) as url:
            config = RemoteModelConfig(
                provider="bailian",
                model="qwen-audio-3.1-tts-flash",
                api_key="test-key",
                base_url=url,
            )
            with Synthesizer(config) as synth:
                with synth.open_stream(options=SynthesisOptions(voice="test")) as stream:
                    stream.send_text("a" * 20001)
                    with self.assertRaises(TTSError) as caught:
                        stream.send_text("b" * 180000)
                    self.assertEqual(caught.exception.code, "INVALID_INPUT")
        self.assertEqual(list(map(len, parts)), [20000, 1])
        self.assertEqual("".join(parts), "a" * 20001)

    def test_incremental_stream_delivers_audio_before_finish_and_preserves_text(self):
        messages = []
        closed = Event()

        def service(ws):
            try:
                run = json.loads(ws.recv())
                task_id = run["header"]["task_id"]
                ws.send(json.dumps({"header": {"task_id": task_id, "event": "task-started"}}))
                while True:
                    message = json.loads(ws.recv())
                    messages.append(message)
                    if message["header"]["action"] == "continue-task":
                        ws.send(b"\x01\x00")
                    else:
                        ws.send(b"\x02\x00")
                        ws.send(
                            json.dumps({"header": {"task_id": task_id, "event": "task-finished"}})
                        )
                        break
            finally:
                closed.set()

        with websocket_service(service) as url:
            config = RemoteModelConfig(
                provider="bailian",
                model="qwen-audio-3.1-tts-flash",
                api_key="test-key",
                base_url=url,
            )
            with Synthesizer(config) as synth:
                with synth.open_stream(options=SynthesisOptions(voice="test")) as stream:
                    stream.send_text("Hello ")
                    self.assertEqual(stream.read(timeout=1).data, b"\x01\x00")
                    stream.send_text("world.")
                    stream.finish_input()
                    self.assertEqual(b"".join(c.data for c in stream), b"\x01\x00\x02\x00")
        self.assertTrue(closed.is_set())
        self.assertEqual(
            [m["payload"]["input"]["text"] for m in messages[:-1]], ["Hello ", "world."]
        )

    def test_service_failure_and_invalid_pcm_tail_remain_errors(self):
        for mode, code in (
            ("service", "REMOTE_RATE_LIMIT"),
            ("tail", "PROTOCOL_ERROR"),
            ("disconnect", "REMOTE_CONNECTION"),
        ):
            with self.subTest(mode=mode):

                def service(ws):
                    run = json.loads(ws.recv())
                    task_id = run["header"]["task_id"]
                    ws.send(json.dumps({"header": {"task_id": task_id, "event": "task-started"}}))
                    ws.recv()
                    ws.recv()
                    if mode == "disconnect":
                        return
                    if mode == "tail":
                        ws.send(b"\x01")
                        header = {"task_id": task_id, "event": "task-finished"}
                    else:
                        header = {
                            "task_id": task_id,
                            "event": "task-failed",
                            "error_code": "Throttling",
                            "error_message": "secret-test-key",
                        }
                    ws.send(json.dumps({"header": header}))

                with websocket_service(service) as url:
                    config = RemoteModelConfig(
                        provider="bailian",
                        model="qwen-audio-3.1-tts-flash",
                        api_key="test-key",
                        base_url=url,
                    )
                    with self.assertRaises(TTSError) as caught:
                        synthesize("Hello.", config=config, options=SynthesisOptions(voice="test"))
                    self.assertEqual(caught.exception.code, code)
                    self.assertNotIn("secret", str(caught.exception))

    def test_finish_deadline_tracks_audio_progress_but_not_heartbeats(self):
        """只替换网络和时钟；核心协议解析及截止计算按真实实现执行。"""
        for progress in (True, False):
            with self.subTest(progress=progress):
                now = [0.0]

                class Connection:
                    def __init__(self, *args, **kwargs):
                        self.task_id = None
                        self.received = 0

                    def __enter__(self):
                        return self

                    def __exit__(self, *args):
                        pass

                    def send(self, raw):
                        self.task_id = json.loads(raw)["header"]["task_id"]

                    def recv(self, timeout=None):
                        self.received += 1
                        if self.received == 1:
                            return json.dumps(
                                {"header": {"task_id": self.task_id, "event": "task-started"}}
                            )
                        now[0] += 0.4
                        if self.received < 6:
                            if progress:
                                return b"\x01\x00"
                            return json.dumps(
                                {
                                    "header": {
                                        "task_id": self.task_id,
                                        "event": "result-generated",
                                    },
                                    "payload": {"output": {"heartbeat": True}},
                                }
                            )
                        return json.dumps(
                            {"header": {"task_id": self.task_id, "event": "task-finished"}}
                        )

                    def close(self):
                        pass

                config = RemoteModelConfig(
                    provider="bailian",
                    model="qwen-audio-3.1-tts-flash",
                    api_key="test-key",
                    timeout_seconds=1,
                )
                with (
                    patch("websockets.sync.client.reconnect", Connection),
                    patch(
                        "speechio.adapters.remote.bailian_tts.monotonic", side_effect=lambda: now[0]
                    ),
                ):
                    if progress:
                        result = synthesize(
                            "Hello.", config=config, options=SynthesisOptions(voice="test")
                        )
                        self.assertEqual(len(result.data), 8)
                    else:
                        with self.assertRaises(TTSError) as caught:
                            synthesize(
                                "Hello.", config=config, options=SynthesisOptions(voice="test")
                            )
                        self.assertEqual(caught.exception.code, "REMOTE_TIMEOUT")

    def test_missing_start_confirmation_reports_network_timeout(self):
        def service(ws):
            ws.recv()
            try:
                ws.recv()
            except ConnectionClosed:
                pass

        with websocket_service(service) as url:
            config = RemoteModelConfig(
                provider="bailian",
                model="qwen-audio-3.1-tts-flash",
                api_key="test-key",
                base_url=url,
                timeout_seconds=0.1,
            )
            with Synthesizer(config) as synth:
                with self.assertRaises(TTSError) as caught:
                    synth.open_stream(options=SynthesisOptions(voice="test"))
                self.assertEqual(caught.exception.code, "REMOTE_TIMEOUT")

    def test_complete_synthesis_uses_one_task_and_collects_pcm_tail(self):
        requests = []

        def service(ws):
            run = json.loads(ws.recv())
            requests.append(run)
            task_id = run["header"]["task_id"]
            ws.send(json.dumps({"header": {"task_id": task_id, "event": "task-started"}}))
            while True:
                message = json.loads(ws.recv())
                requests.append(message)
                if message["header"]["action"] == "finish-task":
                    break
            # 网络帧可以切在半个 PCM 采样处；公开音频块不能。
            ws.send(b"\x01")
            ws.send(b"\x00\x02\x00")
            ws.send(json.dumps({"header": {"task_id": task_id, "event": "task-finished"}}))

        with websocket_service(service) as url:
            result = synthesize(
                "你好。 Hello.",
                config=RemoteModelConfig(
                    provider="bailian",
                    model="qwen-audio-3.1-tts-flash",
                    api_key="test-key",
                    base_url=url,
                ),
                options=SynthesisOptions(
                    voice="test", language="zh", speed=1.2, instruction="温和", sample_rate=24000
                ),
            )
        self.assertEqual(result.data, b"\x01\x00\x02\x00")
        self.assertEqual(
            [item["header"]["action"] for item in requests],
            ["run-task", "continue-task", "finish-task"],
        )
        self.assertEqual(len({item["header"]["task_id"] for item in requests}), 1)
        self.assertEqual(
            requests[0]["payload"]["parameters"],
            {
                "text_type": "PlainText",
                "voice": "test",
                "format": "pcm",
                "sample_rate": 24000,
                "rate": 1.2,
                "language_hints": ["zh"],
                "instruction": "温和",
            },
        )
